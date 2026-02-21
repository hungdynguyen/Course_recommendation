#!/usr/bin/env python3
"""
Split dataset into train/test ensuring no skill overlap between sets.
Test set: 3000 samples + human-labeled samples
Train set: remaining samples (excluding human-labeled skills)
"""
import json
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Paths
INPUT_FILE = Path("data/processed/training_dataset/skill_finetuning_dataset_triplets.json")
HUMAN_LABELS_FILE = Path("data/processed/training_dataset/human_labeled_skills.json")
ESCO_SKILLS_PATH = Path("data/raw/skill_taxonomy/skills_en.csv")
TRAIN_OUTPUT = Path("data/processed/training_dataset/train_dataset_80.json")
TEST_OUTPUT = Path("data/processed/training_dataset/test_dataset_20.json")

# Parameters
TEST_RATIO = 0.20  
RANDOM_SEED = 42

def load_esco_skills():
    """Load ESCO skills for generating negatives"""
    print(f"\nLoading ESCO skills from {ESCO_SKILLS_PATH}...")
    df = pd.read_csv(ESCO_SKILLS_PATH, encoding='utf-8')
    df = df.fillna('')
    
    esco_skills = []
    for _, row in df.iterrows():
        skill_name = row.get('preferredLabel', '')
        description = row.get('description', '')
        definition = row.get('definition', '')
        scope_note = row.get('scopeNote', '')
        
        # Combine text for full description
        desc_parts = [p for p in [description, definition, scope_note] if p]
        full_description = ' '.join(desc_parts).strip()
        if not full_description:
            full_description = skill_name
        
        esco_skills.append({
            'uri': row.get('conceptUri', ''),
            'name': skill_name,
            'description': description,
            'full_text': f"{skill_name}. {full_description}",
            'skill_type': row.get('skillType', ''),
            'alt_labels': row.get('altLabels', '')
        })
    
    print(f"Loaded {len(esco_skills)} ESCO skills")
    return esco_skills

def generate_human_label_triplets(human_labels, esco_skills):
    """Convert human-labeled data to triplet format (1 triplet per label)"""
    print(f"\n{'='*60}")
    print(f"Generating triplets from {len(human_labels)} human labels...")
    print(f"{'='*60}")
    
    triplets = []
    esco_by_uri = {s['uri']: s for s in esco_skills}
    
    # Set random seed for reproducible negative sampling
    random.seed(RANDOM_SEED)
    
    for label in human_labels:
        course_skill = label['course_skill_text']
        esco_uri = label['esco_uri']
        esco_name = label['esco_name']
        esco_desc = label.get('esco_description', '')
        
        # Get full ESCO skill info
        esco_skill = esco_by_uri.get(esco_uri, {})
        
        # Build positive text (same format as training data)
        if esco_desc:
            positive_text = f"{esco_name}. {esco_desc}"
        else:
            positive_text = esco_skill.get('full_text', esco_name)
        
        # Sample one random negative (different from positive)
        negative_skill = random.choice(esco_skills)
        while negative_skill['uri'] == esco_uri:
            negative_skill = random.choice(esco_skills)
        
        negative_text = negative_skill['full_text']
        
        triplet = {
            'query': course_skill,
            'positive': positive_text,
            'negative': negative_text,
            'metadata': {
                'skill_uri': esco_uri,
                'skill_name': esco_name,
                'skill_type': esco_skill.get('skill_type', ''),
                'alt_labels': esco_skill.get('alt_labels', ''),
                'source': 'human_labeled',
                'course_skill_id': label.get('course_skill_id', -1)
            }
        }
        triplets.append(triplet)
    
    print(f"Generated {len(triplets)} triplets from human labels (1 per label)")
    return triplets

def main():
    print("=" * 60)
    print("Splitting Dataset into Train/Test")
    print("with Human-Labeled Data Integration")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Total samples: {len(dataset)}")
    
    # Load human labels if available
    human_label_triplets = []
    human_labeled_course_skills = set()
    
    if HUMAN_LABELS_FILE.exists():
        print(f"\n✓ Human labels file found: {HUMAN_LABELS_FILE}")
        with open(HUMAN_LABELS_FILE, 'r', encoding='utf-8') as f:
            human_labels = json.load(f)
        
        print(f"Found {len(human_labels)} human-labeled pairs")
        
        if len(human_labels) > 0:
            # Load ESCO skills for generating negatives
            esco_skills = load_esco_skills()
            
            # Generate triplets from human labels
            human_label_triplets = generate_human_label_triplets(human_labels, esco_skills)
            
            # Track which course skills are human-labeled
            human_labeled_course_skills = set(label['course_skill_text'] for label in human_labels)
            print(f"\n✓ Will exclude {len(human_labeled_course_skills)} human-labeled skills from auto-generated train/test")
    else:
        print(f"\n⚠️  No human labels file found at {HUMAN_LABELS_FILE}")
        print("Proceeding with auto-generated dataset only")
    
    # Filter out human-labeled skills from auto-generated dataset
    print(f"\n{'='*60}")
    print("Filtering dataset...")
    print(f"{'='*60}")
    
    filtered_dataset = []
    removed_count = 0
    
    for sample in dataset:
        query = sample.get('query', '')
        # Check if this query matches any human-labeled course skill
        is_human_labeled = any(query == course_skill for course_skill in human_labeled_course_skills)
        
        if not is_human_labeled:
            filtered_dataset.append(sample)
        else:
            removed_count += 1
    
    print(f"Removed {removed_count} auto-generated samples that overlap with human labels")
    print(f"Remaining auto-generated samples: {len(filtered_dataset)}")
    
    # Calculate test size based on ratio (from filtered dataset)
    test_size = int(len(filtered_dataset) * TEST_RATIO)
    print(f"\nTarget test size from auto-generated ({TEST_RATIO*100:.0f}%): {test_size}")
    
    # Group samples by skill_uri
    skills_dict = defaultdict(list)
    for sample in filtered_dataset:
        skill_uri = sample['metadata']['skill_uri']
        skills_dict[skill_uri].append(sample)
    
    print(f"Total unique skills in filtered dataset: {len(skills_dict)}")
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Shuffle skills and select for test set
    all_skills = list(skills_dict.keys())
    random.shuffle(all_skills)
    
    test_samples = []
    test_skills = []
    
    # Add skills to test set until we reach test_size
    for skill_uri in all_skills:
        if len(test_samples) >= test_size:
            break
        test_samples.extend(skills_dict[skill_uri])
        test_skills.append(skill_uri)
    
    # Trim to exactly test_size if needed
    if len(test_samples) > test_size:
        test_samples = test_samples[:test_size]
    
    # All remaining samples go to train set
    test_skill_set = set(test_skills)
    train_samples = []
    for skill_uri in all_skills:
        if skill_uri not in test_skill_set:
            train_samples.extend(skills_dict[skill_uri])
    
    print(f"\n{'='*60}")
    print("Auto-generated split completed:")
    print(f"{'='*60}")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples from {len(test_skills)} skills")
    
    # Add human-labeled triplets to test set
    if human_label_triplets:
        print(f"\n{'='*60}")
        print(f"Adding {len(human_label_triplets)} human-labeled samples to test set...")
        print(f"{'='*60}")
        test_samples.extend(human_label_triplets)
        print(f"✓ Human-labeled samples added to test set")
    
    print(f"\n{'='*60}")
    print("FINAL DATASET STATISTICS:")
    print(f"{'='*60}")
    print(f"  Train: {len(train_samples)} samples (auto-generated only)")
    print(f"  Test:  {len(test_samples)} samples")
    print(f"    - Auto-generated: {len(test_samples) - len(human_label_triplets)}")
    print(f"    - Human-labeled:  {len(human_label_triplets)}")
    
    # Count unique skills in test
    test_skill_uris = set(s['metadata']['skill_uri'] for s in test_samples)
    print(f"  Total unique skills in test: {len(test_skill_uris)}")
    
    # Verify no overlap in skill URIs
    train_skill_uris = set(s['metadata']['skill_uri'] for s in train_samples)
    test_skill_uris = set(s['metadata']['skill_uri'] for s in test_samples)
    overlap = train_skill_uris & test_skill_uris
    
    if overlap:
        print(f"\n⚠️ WARNING: Found {len(overlap)} overlapping skill URIs between train and test!")
    else:
        print(f"\n✓ Verified: No skill URI overlap between train and test sets")
    
    # Verify no overlap in course skill queries (important for human labels)
    train_queries = set(s['query'] for s in train_samples)
    test_queries = set(s['query'] for s in test_samples)
    query_overlap = train_queries & test_queries
    
    if query_overlap:
        print(f"⚠️ WARNING: Found {len(query_overlap)} overlapping queries between train and test!")
        print(f"   First 3: {list(query_overlap)[:3]}")
    else:
        print(f"✓ Verified: No query overlap between train and test sets")
    
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
