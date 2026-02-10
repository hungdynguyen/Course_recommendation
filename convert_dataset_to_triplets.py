"""
Convert old fine-tuning dataset to new triplet format
- Old: Each sample has query, positive, and a list of negatives (duplicated)
- New: Each sample has query, positive, and exactly 1 negative (unique)
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd

# Configuration
INPUT_FILE = Path("/root/courses_rec/data/processed/training_dataset/skill_finetuning_dataset.json")
OUTPUT_FILE = Path("/root/courses_rec/data/processed/training_dataset/skill_finetuning_dataset_triplets.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def convert_to_triplets(input_path: Path, output_path: Path):
    """Convert dataset from multiple negatives to single negative per sample"""
    
    print("="*60)
    print("Converting Dataset to Triplet Format")
    print("="*60)
    
    # Load old dataset
    print(f"\n📂 Loading data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        old_dataset = json.load(f)
    
    print(f"  ✅ Loaded {len(old_dataset)} samples")
    
    # Group samples by skill_uri to understand the structure
    skills_grouped = defaultdict(list)
    for sample in old_dataset:
        skill_uri = sample['metadata']['skill_uri']
        skills_grouped[skill_uri].append(sample)
    
    print(f"  📊 Found {len(skills_grouped)} unique skills")
    
    # Convert to new format
    new_dataset = []
    total_negatives_used = 0
    
    print("\n🔄 Converting to triplet format...")
    print("  Strategy: Each query gets 1 unique negative from the list")
    
    for skill_uri, samples in skills_grouped.items():
        # All samples from same skill have same negatives list
        if not samples:
            continue
        
        # Get the list of negatives (same for all queries of this skill)
        negatives_pool = samples[0].get('negatives', [])
        
        # Assign one negative to each query variation
        for idx, sample in enumerate(samples):
            # Get a unique negative for this query (cycle through if needed)
            negative_idx = idx % len(negatives_pool) if negatives_pool else 0
            single_negative = negatives_pool[negative_idx] if negatives_pool else ""
            
            # Create new triplet with single negative
            new_sample = {
                'query': sample['query'],
                'positive': sample['positive'],
                'negative': single_negative,  # Changed from list to single string
                'metadata': sample['metadata']
            }
            
            new_dataset.append(new_sample)
            total_negatives_used += 1
    
    # Remove duplicates based on exact (query, positive, negative) match
    print("\n🔍 Removing duplicates...")
    seen = set()
    unique_dataset = []
    
    for sample in new_dataset:
        # Create a unique key
        key = (sample['query'], sample['positive'], sample['negative'])
        if key not in seen:
            seen.add(key)
            unique_dataset.append(sample)
    
    duplicates_removed = len(new_dataset) - len(unique_dataset)
    print(f"  ✅ Removed {duplicates_removed} duplicate triplets")
    
    # Save new dataset
    print(f"\n💾 Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_dataset, f, ensure_ascii=False, indent=2)
    
    # Save simplified Excel version
    print(f"\n📊 Creating simplified Excel file...")
    simplified_data = []
    for item in unique_dataset:
        simplified_data.append({
            'query': item['query'],
            'positive': item['positive'],
            'negative': item['negative']
        })
    
    df_simple = pd.DataFrame(simplified_data)
    excel_file = output_path.parent / "skill_finetuning_dataset_triplets_simple.xlsx"
    df_simple.to_excel(excel_file, index=False, engine='openpyxl')
    print(f"  ✅ Saved Excel: {excel_file}")
    
    # Generate statistics
    stats = {
        'conversion_info': {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'converted_at': datetime.now().isoformat()
        },
        'old_format': {
            'total_samples': len(old_dataset),
            'structure': 'query + positive + negatives (list)'
        },
        'new_format': {
            'total_samples': len(unique_dataset),
            'unique_triplets': len(unique_dataset),
            'duplicates_removed': duplicates_removed,
            'structure': 'query + positive + negative (single)'
        },
        'skills': {
            'unique_skills': len(skills_grouped),
            'avg_samples_per_skill': len(unique_dataset) / len(skills_grouped) if skills_grouped else 0
        }
    }
    
    stats_file = output_path.parent / "conversion_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETED!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"  Old format:")
    print(f"    - Total samples: {len(old_dataset):,}")
    print(f"    - Structure: query + positive + negatives (list)")
    print(f"\n  New format:")
    print(f"    - Total triplets: {len(unique_dataset):,}")
    print(f"    - Duplicates removed: {duplicates_removed:,}")
    print(f"    - Structure: query + positive + negative (single)")
    print(f"\n  Skills:")
    print(f"    - Unique skills: {len(skills_grouped):,}")
    print(f"    - Avg samples/skill: {stats['skills']['avg_samples_per_skill']:.1f}")
    print(f"\n📁 Output files:")
    print(f"  - Dataset: {output_path}")
    print(f"  - Simplified Excel: {excel_file}")
    print(f"  - Statistics: {stats_file}")
    print(f"\n🎯 Ready for fine-tuning with triplet loss!")
    
    # Show sample
    if unique_dataset:
        print(f"\n📝 Sample triplet:")
        sample = unique_dataset[0]
        print(f"  Query: {sample['query'][:80]}...")
        print(f"  Positive: {sample['positive'][:80]}...")
        print(f"  Negative: {sample['negative'][:80]}...")


def main():
    """Main execution"""
    print("\n🚀 Starting dataset conversion...")
    
    if not INPUT_FILE.exists():
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return
    
    try:
        convert_to_triplets(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
