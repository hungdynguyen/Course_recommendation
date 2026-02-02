import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import asyncio
from google import genai
from google.genai import types
import time
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# Configure Gemini API
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize client
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Paths configuration
BASE_DIR = Path(__file__).parent.parent
ESCO_SKILLS_PATH = BASE_DIR / "raw" / "skill_taxonomy"
OUTPUT_DIR = BASE_DIR / "processed" / "training_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Generation parameters
TARGET_SAMPLES = 41817
VARIATIONS_PER_SKILL = 3  # Number of positive variations per skill
NUM_HARD_NEGATIVES = 3  # Total hard negatives (1 per query variation)
NEGATIVES_PER_QUERY = 1  # Number of negatives per query
# LLM_MODEL = "gemini-3-flash-preview"
LLM_MODEL = "gemini-2.5-pro"
MAX_CONCURRENT_REQUESTS = 50  
CHECKPOINT_INTERVAL = 50  # Save progress every N skills
MAX_RETRIES = 3  # Retry failed requests


class SkillDatasetGenerator:
    """Generate skill dataset using concurrent API calls with combined positive/negative generation"""
    
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.esco_skills = []
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
    def load_esco_skills(self, skills_file: Optional[str] = None) -> List[Dict]:
        """Load ESCO skills from CSV"""
        print("Loading ESCO skills from CSV...")
        
        if skills_file is None:
            skills_file = ESCO_SKILLS_PATH / "skills_en.csv"
            if not skills_file.exists():
                raise FileNotFoundError(f"ESCO skills file not found: {skills_file}")
        
        df = pd.read_csv(skills_file, encoding='utf-8')
        # Replace NaN with empty strings
        df = df.fillna('')
        self.esco_skills = df.to_dict('records')
        
        print(f"Loaded {len(self.esco_skills)} ESCO skills from CSV")
        return self.esco_skills
    
    async def generate_variations_and_negatives(self, skill: Dict, skill_idx: int, semaphore: asyncio.Semaphore) -> Dict:
        """Generate both positive variations and hard negatives in a single prompt"""
        async with semaphore:
            skill_name = skill.get('preferredLabel', 'Unknown Skill')
            skill_description = str(skill.get('description', '')).strip() if pd.notna(skill.get('description')) else ''
            skill_definition = str(skill.get('definition', '')).strip() if pd.notna(skill.get('definition')) else ''
            skill_scope_note = str(skill.get('scopeNote', '')).strip() if pd.notna(skill.get('scopeNote')) else ''
            skill_uri = skill.get('conceptUri', '')
            skill_type = skill.get('skillType', '')
            alt_labels = skill.get('altLabels', '')
            
            # Build full_description by joining non-empty parts
            desc_parts = [p for p in [skill_description, skill_definition, skill_scope_note] if p]
            full_description = ' '.join(desc_parts).strip()
            if not full_description:
                full_description = skill_name
            
            prompt = f"""You are an expert in curriculum design and skill taxonomy.

Given this skill from ESCO:

Skill Name: {skill_name}
Description: {full_description}

Your task is to generate TWO types of content:

1. POSITIVE VARIATIONS ({VARIATIONS_PER_SKILL} samples): Generate {VARIATIONS_PER_SKILL} different ways this skill might be written in university course descriptions or syllabi. Each variation should:
   - Use natural language as in real educational contexts
   - Use synonyms or different phrasings
   - Can be more specific or more general
   - Vary in length (short phrases to full sentences)
   - IMPORTANT: Only generate variations of THIS exact skill. Do NOT generate similar but different skills.

2. HARD NEGATIVES ({NUM_HARD_NEGATIVES} samples): Generate {NUM_HARD_NEGATIVES} DIFFERENT skill descriptions that are:
   - Similar in wording or context but describe DIFFERENT skills
   - Confusing or easily mistaken for the given skill
   - Related but NOT the same (e.g., if skill is "Python programming", generate "Java programming", "Software debugging")
   - Subtly different (wrong level, wrong domain, wrong application)
   - Natural sounding but INCORRECT for this specific skill

Output format: Return ONLY a JSON object with this structure:
{{
  "variations": ["variation 1", "variation 2", "variation 3"],
  "hard_negatives": ["negative 1", "negative 2"]
}}
"""
            
            for attempt in range(MAX_RETRIES):
                try:
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=prompt
                    )
                    
                    response_text = response.text.strip()
                    
                    # Try to parse JSON response
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    response_data = json.loads(response_text)
                    variations = response_data.get('variations', [])
                    hard_negatives = response_data.get('hard_negatives', [])
                    
                    return {
                        'success': True,
                        'skill_idx': skill_idx,
                        'skill_uri': skill_uri,
                        'skill_name': skill_name,
                        'skill_type': skill_type,
                        'alt_labels': alt_labels,
                        'original_description': full_description,
                        'variations': variations[:VARIATIONS_PER_SKILL],
                        'hard_negatives': hard_negatives[:NUM_HARD_NEGATIVES]
                    }
                    
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        print(f"  ❌ Failed to generate data for skill {skill_idx}: {skill_name} - {e}")
                        return {
                            'success': False,
                            'skill_idx': skill_idx,
                            'skill_uri': skill_uri,
                            'skill_name': skill_name,
                            'error': str(e)
                        }
    
    async def generate_dataset_batch(self, skills: List[Dict], start_idx: int = 0) -> List[Dict]:
        """Generate both positive variations and hard negatives for all skills concurrently"""
        print(f"\n{'='*60}")
        print("Generating Skill Dataset (Positive + Negative)")
        print(f"{'='*60}")
        print(f"  - Total skills: {len(skills)}")
        print(f"  - Concurrent requests: {MAX_CONCURRENT_REQUESTS}")
        print(f"  - Variations per skill: {VARIATIONS_PER_SKILL}")
        print(f"  - Hard negatives per skill: {NUM_HARD_NEGATIVES}")
        
        checkpoint_file = OUTPUT_DIR / "dataset_checkpoint.json"
        final_dataset_file = OUTPUT_DIR / "skill_finetuning_dataset.json"
        
        # Load existing successful skills from final dataset
        results = []
        processed_uris = set()
        
        # Priority 1: Load from final dataset file (successful skills)
        if final_dataset_file.exists():
            try:
                with open(final_dataset_file, 'r', encoding='utf-8') as f:
                    existing_dataset = json.load(f)
                processed_uris = {item['metadata']['skill_uri'] for item in existing_dataset}
                print(f"  📂 Loaded {len(processed_uris)} successfully processed skills from final dataset")
            except Exception as e:
                print(f"  ⚠️ Could not load final dataset: {e}")
        
        # Priority 2: Load from checkpoint if final dataset doesn't exist
        if not processed_uris and checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                processed_uris = {r['skill_uri'] for r in results if r.get('success', False)}
                print(f"  📂 Loaded checkpoint: {len(results)} skills already processed")
            except Exception as e:
                print(f"  ⚠️ Could not load checkpoint: {e}")
                results = []
        
        # Filter out already processed skills
        skills_to_process = [s for s in skills if s.get('conceptUri', '') not in processed_uris]
        
        if not skills_to_process:
            print(f"  ✅ All skills already processed!")
            return results
        
        print(f"  - Remaining skills: {len(skills_to_process)}")
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = []
        
        for idx, skill in enumerate(skills_to_process, start=start_idx + len(results)):
            task = self.generate_variations_and_negatives(skill, idx, semaphore)
            tasks.append(task)
        
        # Process with real-time progress updates
        checkpoint_counter = 0
        
        with tqdm(total=len(tasks), desc="Generating dataset", unit="skill") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                checkpoint_counter += 1
                
                # Update progress bar
                pbar.update(1)
                success = result.get('success', False)
                pbar.set_postfix({
                    'total': len(results),
                    'current': 'OK' if success else 'FAIL'
                })
                
                # Save checkpoint every N skills
                if checkpoint_counter % CHECKPOINT_INTERVAL == 0:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Final checkpoint save
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        successful_results = [r for r in results if r.get('success', False)]
        print(f"\n✅ Completed dataset generation: {len(successful_results)}/{len(results)} successful")
        
        return successful_results
    
    def format_final_dataset(self, results: List[Dict], existing_dataset: List[Dict] = None) -> List[Dict]:
        """Format the generated data into final training dataset structure"""
        print(f"\n{'='*60}")
        print("Formatting Final Dataset")
        print(f"{'='*60}")
        
        # Start with existing dataset if provided
        final_dataset = existing_dataset if existing_dataset else []
        existing_count = len(final_dataset)
        
        if existing_count > 0:
            print(f"  📂 Starting with {existing_count} existing samples")
        
        # Format into final structure
        new_samples = []
        skipped_count = 0
        
        for result in results:
            if not result.get('success', False):
                skipped_count += 1
                continue
            
            variations = result.get('variations', [])
            hard_negatives = result.get('hard_negatives', [])
            
            # Skip if missing data
            if not variations or not hard_negatives:
                skipped_count += 1
                continue
            
            # Ensure we have enough negatives
            if len(hard_negatives) < NUM_HARD_NEGATIVES:
                skipped_count += 1
                continue
            
            # Create a sample for each variation with unique negative
            for idx, variation in enumerate(variations):
                # Mỗi query lấy 1 negative riêng
                if idx >= len(hard_negatives):
                    continue  # Skip nếu không đủ negatives
                
                query_negative = [hard_negatives[idx]]  # 1 negative duy nhất
                
                new_samples.append({
                    'query': variation,
                    'positive': f"{result['skill_name']}. {result['original_description']}",
                    'negatives': query_negative,
                    'metadata': {
                        'skill_uri': result['skill_uri'],
                        'skill_name': result['skill_name'],
                        'skill_type': result['skill_type'],
                        'alt_labels': result['alt_labels'],
                        'generated_at': datetime.now().isoformat(),
                        'model': self.model_name
                    }
                })
        
        # Add new samples to final dataset
        final_dataset.extend(new_samples)
        
        if skipped_count > 0:
            print(f"  ⚠️ Skipped {skipped_count} skills with missing data")
        
        print(f"  ✅ Added {len(new_samples)} new samples")
        
        unique_skills = len(set(s['metadata']['skill_uri'] for s in final_dataset))
        print(f"✅ Total: {len(final_dataset)} samples from {unique_skills} skills")
        
        return final_dataset
    
    async def generate_dataset(self, max_skills: Optional[int] = None):
        """Main generation function"""
        if not self.esco_skills:
            self.load_esco_skills()
        
        if max_skills is None:
            max_skills = TARGET_SAMPLES // VARIATIONS_PER_SKILL
        
        max_skills = min(max_skills, len(self.esco_skills))
        skills_to_process = self.esco_skills[:max_skills]
        
        print(f"\n💻 Concurrent Processing Mode (Combined Positive + Negative)")
        print(f"  - Skills to process: {len(skills_to_process)}")
        print(f"  - Variations per skill: {VARIATIONS_PER_SKILL}")
        print(f"  - Hard negatives per skill: {NUM_HARD_NEGATIVES}")
        print(f"  - Expected samples: ~{len(skills_to_process) * VARIATIONS_PER_SKILL}")
        print(f"  - Concurrent requests: {MAX_CONCURRENT_REQUESTS}")
        print(f"  - Model: {self.model_name}")
        
        start_time = time.time()
        
        # Load existing dataset if it exists
        output_file = OUTPUT_DIR / "skill_finetuning_dataset.json"
        existing_dataset = []
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_dataset = json.load(f)
                print(f"  📂 Loaded {len(existing_dataset)} existing samples from final dataset")
            except Exception as e:
                print(f"  ⚠️ Could not load existing dataset: {e}")
        
        # Generate both positive variations AND hard negatives in one go
        results = await self.generate_dataset_batch(skills_to_process)
        
        # Format into final dataset structure (merge with existing)
        final_dataset = self.format_final_dataset(results, existing_dataset)
        
        # Save final dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        
        # Save simplified version (no metadata) for easy viewing
        simplified_data = []
        for item in final_dataset:
            simplified_data.append({
                'query': item['query'],
                'positive': item['positive'],
                'negatives': ' | '.join(item['negatives'])
            })
        
        # Save as CSV
        df_simple = pd.DataFrame(simplified_data)
        
        # Save as Excel
        excel_file = OUTPUT_DIR / "skill_finetuning_dataset_simple.xlsx"
        df_simple.to_excel(excel_file, index=False, engine='openpyxl')
        
        # Save statistics
        elapsed_time = time.time() - start_time
        stats = {
            'total_samples': len(final_dataset),
            'unique_skills': len(set(s['metadata']['skill_uri'] for s in final_dataset)),
            'variations_per_skill': VARIATIONS_PER_SKILL,
            'hard_negatives_per_sample': NUM_HARD_NEGATIVES,
            'success_rate': len(results) / len(skills_to_process),
            'processing_time_seconds': elapsed_time,
            'created_at': datetime.now().isoformat(),
            'model': self.model_name,
            'concurrent_requests': MAX_CONCURRENT_REQUESTS
        }
        
        stats_file = OUTPUT_DIR / "finetuning_dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print("✅ COMPLETED!")
        print(f"{'='*60}")
        print(f"\n📊 Statistics:")
        print(f"  - Total samples: {len(final_dataset)}")
        print(f"  - Unique skills: {stats['unique_skills']}")
        print(f"  - Variations per skill: {VARIATIONS_PER_SKILL}")
        print(f"  - Hard negatives per sample: {NUM_HARD_NEGATIVES}")
        print(f"  - Success rate: {stats['success_rate']:.1%}")
        print(f"  - Processing time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
        print(f"\n📁 Output:")
        print(f"  - Dataset: {output_file}")
        print(f"  - Simplified Excel: {excel_file}")
        print(f"  - Statistics: {stats_file}")
        print(f"\n🎯 Ready for fine-tuning!")


def main():
    print("="*60)
    print("ESCO Skills Fine-tuning Dataset Generator")
    print("Using Gemini with Concurrent API Calls")
    print("="*60)
    
    if not GEMINI_API_KEY:
        print("\n⚠️ WARNING: GEMINI_API_KEY not set!")
        return
    
    generator = SkillDatasetGenerator()
    
    try:
        asyncio.run(generator.generate_dataset())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
