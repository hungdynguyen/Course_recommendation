#!/usr/bin/env python3
"""
Split dataset into train/test ensuring no skill overlap between sets.
Test set: 3000 samples
Train set: remaining samples
"""
import json
import random
from pathlib import Path
from collections import defaultdict

# Paths
INPUT_FILE = Path("data/processed/training_dataset/skill_finetuning_dataset_triplets.json")
TRAIN_OUTPUT = Path("data/processed/training_dataset/train_dataset.json")
TEST_OUTPUT = Path("data/processed/training_dataset/test_dataset.json")

# Parameters
TEST_SIZE = 3000
RANDOM_SEED = 42

def main():
    print("=" * 60)
    print("Splitting Dataset into Train/Test")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Total samples: {len(dataset)}")
    
    # Group samples by skill_uri
    skills_dict = defaultdict(list)
    for sample in dataset:
        skill_uri = sample['metadata']['skill_uri']
        skills_dict[skill_uri].append(sample)
    
    print(f"Total unique skills: {len(skills_dict)}")
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Shuffle skills and select for test set
    all_skills = list(skills_dict.keys())
    random.shuffle(all_skills)
    
    test_samples = []
    test_skills = []
    
    # Add skills to test set until we reach TEST_SIZE
    for skill_uri in all_skills:
        if len(test_samples) >= TEST_SIZE:
            break
        test_samples.extend(skills_dict[skill_uri])
        test_skills.append(skill_uri)
    
    # Trim to exactly TEST_SIZE if needed
    if len(test_samples) > TEST_SIZE:
        test_samples = test_samples[:TEST_SIZE]
    
    # All remaining samples go to train set
    test_skill_set = set(test_skills)
    train_samples = []
    for skill_uri in all_skills:
        if skill_uri not in test_skill_set:
            train_samples.extend(skills_dict[skill_uri])
    
    print(f"\n✅ Split completed:")
    print(f"  Train: {len(train_samples)} samples from {len(train_samples) // 3} skills (approx)")
    print(f"  Test:  {len(test_samples)} samples from {len(test_skills)} skills")
    
    # Verify no overlap
    train_skill_uris = set(s['metadata']['skill_uri'] for s in train_samples)
    test_skill_uris = set(s['metadata']['skill_uri'] for s in test_samples)
    overlap = train_skill_uris & test_skill_uris
    
    if overlap:
        print(f"\n⚠️ WARNING: Found {len(overlap)} overlapping skills!")
    else:
        print(f"\n✓ Verified: No skill overlap between train and test sets")
    
    # Save train set
    print(f"\nSaving train set to {TRAIN_OUTPUT}...")
    with open(TRAIN_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    # Save test set
    print(f"Saving test set to {TEST_OUTPUT}...")
    with open(TEST_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  📄 {TRAIN_OUTPUT}")
    print(f"  📄 {TEST_OUTPUT}")

if __name__ == "__main__":
    main()
