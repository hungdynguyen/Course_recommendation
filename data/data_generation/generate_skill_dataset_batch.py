"""
Generate skill fine-tuning dataset using Gemini Batch API
Generates positive and negative samples SEPARATELY to avoid bias
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from google import genai
from google.genai import types
import pandas as pd
import time

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
TARGET_SAMPLES = 300
VARIATIONS_PER_SKILL = 3  # Number of positive variations per skill
NUM_HARD_NEGATIVES = 2  # Number of hard negatives per skill
LLM_MODEL = "gemini-3-flash-preview"
BATCH_SIZE = 100  # Number of requests per batch
BATCH_POLL_INTERVAL = 20  # Seconds between status checks


class SkillDatasetGenerator:
    """Generate skill dataset using Batch API with separate positive/negative generation"""
    
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.esco_skills = []
        
    def load_esco_skills(self, skills_file: Optional[str] = None) -> List[Dict]:
        """Load ESCO skills from CSV"""
        print("Loading ESCO skills from CSV...")
        
        if skills_file is None:
            skills_file = ESCO_SKILLS_PATH / "skills_en.csv"
            if not skills_file.exists():
                raise FileNotFoundError(f"ESCO skills file not found: {skills_file}")
        
        df = pd.read_csv(skills_file, encoding='utf-8')
        self.esco_skills = df.to_dict('records')
        
        print(f"Loaded {len(self.esco_skills)} ESCO skills from CSV")
        return self.esco_skills
    
    def create_positive_batch_requests(self, skills: List[Dict]) -> List[Dict]:
        """Create batch requests for POSITIVE variations"""
        batch_requests = []
        
        for idx, skill in enumerate(skills):
            skill_name = skill.get('preferredLabel', 'Unknown Skill')
            skill_description = skill.get('description', '')
            skill_definition = skill.get('definition', '')
            skill_scope_note = skill.get('scopeNote', '')
            skill_uri = skill.get('conceptUri', '')
            skill_type = skill.get('skillType', '')
            alt_labels = skill.get('altLabels', '')
            
            full_description = f"{skill_description} {skill_definition} {skill_scope_note}".strip()
            if not full_description:
                full_description = skill_name
            
            prompt = f"""You are an expert in curriculum design and skill taxonomy.

Given this skill from ESCO:

Skill Name: {skill_name}
Description: {full_description}

Generate {VARIATIONS_PER_SKILL} different ways this skill might be written in university course descriptions or syllabi. Each variation should:
- Use natural language as in real educational contexts
- Use synonyms or different phrasings
- Can be more specific or more general
- Vary in length (short phrases to full sentences)

IMPORTANT: Only generate variations of THIS exact skill. Do NOT generate similar but different skills.

Output format: Return ONLY a JSON object with this structure:
{{"variations": ["variation 1", "variation 2", "variation 3"]}}
"""
            
            batch_requests.append({
                'key': f'pos_{idx}',
                'request': {
                    'contents': [{'parts': [{'text': prompt}]}]
                }
            })
        
        # Save metadata separately for later use
        metadata_file = OUTPUT_DIR / f"positive_batch_metadata.json"
        skill_metadata = [{
            'key': f'pos_{idx}',
            'skill_uri': skill.get('conceptUri', ''),
            'skill_name': skill.get('preferredLabel', ''),
            'skill_type': skill.get('skillType', ''),
            'alt_labels': skill.get('altLabels', ''),
            'original_description': f"{skill.get('description', '')} {skill.get('definition', '')} {skill.get('scopeNote', '')}".strip()
        } for idx, skill in enumerate(skills)]
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(skill_metadata, f, ensure_ascii=False, indent=2)
        
        return batch_requests
    
    def create_negative_batch_requests(self, skills: List[Dict]) -> List[Dict]:
        """Create batch requests for HARD NEGATIVES"""
        batch_requests = []
        
        for idx, skill in enumerate(skills):
            skill_name = skill.get('preferredLabel', 'Unknown Skill')
            skill_description = skill.get('description', '')
            skill_definition = skill.get('definition', '')
            skill_scope_note = skill.get('scopeNote', '')
            skill_uri = skill.get('conceptUri', '')
            
            full_description = f"{skill_description} {skill_definition} {skill_scope_note}".strip()
            if not full_description:
                full_description = skill_name
            
            prompt = f"""You are an expert in curriculum design and skill taxonomy.

Given this skill from ESCO:

Skill Name: {skill_name}
Description: {full_description}

Generate {NUM_HARD_NEGATIVES} DIFFERENT skill descriptions that are:
- Similar in wording or context but describe DIFFERENT skills
- Confusing or easily mistaken for the given skill
- Related but NOT the same (e.g., if skill is "Python programming", generate "Java programming", "Software debugging")
- Subtly different (wrong level, wrong domain, wrong application)
- Natural sounding but INCORRECT for this specific skill

These should be challenging hard negatives that could confuse an AI model.

Output format: Return ONLY a JSON object with this structure:
{{"hard_negatives": ["negative 1", "negative 2"]}}
"""
            
            batch_requests.append({
                'key': f'neg_{idx}',
                'request': {
                    'contents': [{'parts': [{'text': prompt}]}]
                }
            })
        
        # Save metadata separately
        metadata_file = OUTPUT_DIR / f"negative_batch_metadata.json"
        skill_metadata = [{
            'key': f'neg_{idx}',
            'skill_uri': skill.get('conceptUri', ''),
            'skill_name': skill.get('preferredLabel', ''),
            'original_description': f"{skill.get('description', '')} {skill.get('definition', '')} {skill.get('scopeNote', '')}".strip()
        } for idx, skill in enumerate(skills)]
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(skill_metadata, f, ensure_ascii=False, indent=2)
        
        return batch_requests
    
    def submit_batch_job(self, batch_requests: List[Dict], batch_name: str) -> str:
        """Submit a batch job to Gemini Batch API"""
        batch_file = OUTPUT_DIR / f"{batch_name}_requests.jsonl"
        with open(batch_file, 'w', encoding='utf-8') as f:
            for request in batch_requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        print(f"📤 Submitting batch job: {batch_name}")
        print(f"   Requests: {len(batch_requests)}")
        
        try:
            # Upload file
            uploaded_file = client.files.upload(
                file=str(batch_file),
                config=types.UploadFileConfig(
                    display_name=batch_name,
                    mime_type='application/jsonl'
                )
            )
            print(f"   Uploaded file: {uploaded_file.name}")
            
            # Create batch job
            batch_job = client.batches.create(
                model=f"models/{self.model_name}",
                src=uploaded_file.name,
                config={'display_name': batch_name}
            )
            
            print(f"✅ Batch job submitted: {batch_job.name}")
            return batch_job.name
            
        except Exception as e:
            print(f"❌ Error submitting batch job: {e}")
            raise
    
    def wait_for_batch_completion(self, job_id: str, max_wait_time: int = 3600) -> bool:
        """Wait for batch job to complete"""
        print(f"⏳ Waiting for batch job to complete: {job_id}")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                batch_job = client.batches.get(name=job_id)
                state = batch_job.state
                
                print(f"   Status: {state} (elapsed: {int(time.time() - start_time)}s)")
                
                if state == 'STATE_SUCCEEDED':
                    print(f"✅ Batch job completed successfully")
                    return True
                elif state in ['STATE_FAILED', 'STATE_CANCELLED']:
                    print(f"❌ Batch job failed with state: {state}")
                    return False
                
                time.sleep(BATCH_POLL_INTERVAL)
            except Exception as e:
                print(f"⚠️ Error checking status: {e}")
                time.sleep(BATCH_POLL_INTERVAL)
        
        print(f"⏱️ Timeout waiting for batch job")
        return False
    
    def process_positive_results(self, job_id: str, batch_name: str) -> List[Dict]:
        """Process positive variation results"""
        print(f"📥 Processing positive results: {batch_name}")
        
        try:
            # Load metadata
            metadata_file = OUTPUT_DIR / "positive_batch_metadata.json"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            metadata_by_key = {m['key']: m for m in metadata_list}
            
            batch_job = client.batches.get(name=job_id)
            if not batch_job.output_uri:
                print(f"❌ No output URI available")
                return []
            
            output_file = OUTPUT_DIR / f"{batch_name}_results.jsonl"
            
            # Download results file
            result_file = client.files.get(name=batch_job.output_uri)
            # Note: Need to implement download - for now read directly
            # Assuming output_uri points to a file we can read
            
            samples = []
            with open(output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        result = json.loads(line)
                        metadata = result.get('metadata', {})
                        response = result.get('response', {})
                        candidates = response.get('candidates', [])
                        
                        if candidates:
                            content = candidates[0].get('content', {})
                            parts = content.get('parts', [])
                            
                            if parts and 'text' in parts[0]:
                                response_text = parts[0]['text']
                                response_data = json.loads(response_text)
                                variations = response_data.get('variations', [])
                                
                                for variation in variations[:VARIATIONS_PER_SKILL]:
                                    samples.append({
                                        'type': 'positive',
                                        'text': variation,
                                        'skill_uri': metadata.get('skill_uri', ''),
                                        'skill_name': metadata.get('skill_name', ''),
                                        'skill_type': metadata.get('skill_type', ''),
                                        'alt_labels': metadata.get('alt_labels', ''),
                                        'original_description': metadata.get('original_description', ''),
                                    })
                    except Exception as e:
                        print(f"  ⚠️ Error parsing line {line_num}: {e}")
                        continue
            
            print(f"✅ Processed {len(samples)} positive samples")
            return samples
            
        except Exception as e:
            print(f"❌ Error processing results: {e}")
            return []
    
    def process_negative_results(self, job_id: str, batch_name: str) -> List[Dict]:
        """Process hard negative results"""
        print(f"📥 Processing negative results: {batch_name}")
        
        try:
            # Load metadata
            metadata_file = OUTPUT_DIR / "negative_batch_metadata.json"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            metadata_by_key = {m['key']: m for m in metadata_list}
            
            batch_job = client.batches.get(name=job_id)
            if not batch_job.output_uri:
                print(f"❌ No output URI available")
                return []
            
            output_file = OUTPUT_DIR / f"{batch_name}_results.jsonl"
            
            # Download results file
            result_file = client.files.get(name=batch_job.output_uri)
            # Note: Need to implement download - for now read directly
            
            samples = []
            with open(output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        result = json.loads(line)
                        metadata = result.get('metadata', {})
                        response = result.get('response', {})
                        candidates = response.get('candidates', [])
                        
                        if candidates:
                            content = candidates[0].get('content', {})
                            parts = content.get('parts', [])
                            
                            if parts and 'text' in parts[0]:
                                response_text = parts[0]['text']
                                response_data = json.loads(response_text)
                                hard_negatives = response_data.get('hard_negatives', [])
                                
                                samples.append({
                                    'type': 'negative',
                                    'hard_negatives': hard_negatives[:NUM_HARD_NEGATIVES],
                                    'skill_uri': metadata.get('skill_uri', ''),
                                    'skill_name': metadata.get('skill_name', ''),
                                })
                    except Exception as e:
                        print(f"  ⚠️ Error parsing line {line_num}: {e}")
                        continue
            
            print(f"✅ Processed {len(samples)} negative samples")
            return samples
            
        except Exception as e:
            print(f"❌ Error processing results: {e}")
            return []
    
    def combine_positive_negative(self, positive_results: List[Dict], negative_results: List[Dict]) -> List[Dict]:
        """Combine positive variations with hard negatives"""
        print(f"\n🔗 Combining positives and negatives...")
        
        # Group by skill_uri
        positives_by_skill = {}
        for pos in positive_results:
            skill_uri = pos['skill_uri']
            if skill_uri not in positives_by_skill:
                positives_by_skill[skill_uri] = []
            positives_by_skill[skill_uri].append(pos)
        
        negatives_by_skill = {}
        for neg in negative_results:
            skill_uri = neg['skill_uri']
            negatives_by_skill[skill_uri] = neg.get('hard_negatives', [])
        
        # Combine
        combined = []
        for skill_uri, pos_samples in positives_by_skill.items():
            negatives = negatives_by_skill.get(skill_uri, [])
            
            if not negatives:
                print(f"  ⚠️ No negatives for {skill_uri}, skipping...")
                continue
            
            for pos in pos_samples:
                combined.append({
                    'query': pos['text'],
                    'positive': f"{pos['skill_name']}. {pos['original_description']}",
                    'negatives': negatives,
                    'metadata': {
                        'skill_uri': skill_uri,
                        'skill_name': pos['skill_name'],
                        'skill_type': pos['skill_type'],
                        'alt_labels': pos['alt_labels'],
                        'generated_at': datetime.now().isoformat(),
                        'model': self.model_name
                    }
                })
        
        print(f"✅ Combined {len(combined)} samples from {len(positives_by_skill)} skills")
        return combined
    
    def generate_dataset(self, max_skills: Optional[int] = None):
        """Main generation function"""
        if not self.esco_skills:
            self.load_esco_skills()
        
        if max_skills is None:
            max_skills = TARGET_SAMPLES // VARIATIONS_PER_SKILL
        
        max_skills = min(max_skills, len(self.esco_skills))
        skills_to_process = self.esco_skills[:max_skills]
        
        print(f"\n💰 Batch API Processing Mode")
        print(f"  - Skills to process: {len(skills_to_process)}")
        print(f"  - Variations per skill: {VARIATIONS_PER_SKILL}")
        print(f"  - Hard negatives per skill: {NUM_HARD_NEGATIVES}")
        print(f"  - Batch size: {BATCH_SIZE}")
        
        positive_results = []
        negative_results = []
        
        num_batches = (len(skills_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # STEP 1: Generate POSITIVE variations
        print(f"\n{'='*60}")
        print("STEP 1: Generating POSITIVE Variations")
        print(f"{'='*60}")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(skills_to_process))
            batch_skills = skills_to_process[start_idx:end_idx]
            
            batch_name = f"positive_batch_{batch_idx + 1}"
            print(f"\n📦 Batch {batch_idx + 1}/{num_batches} (skills {start_idx + 1}-{end_idx})")
            
            batch_requests = self.create_positive_batch_requests(batch_skills)
            job_id = self.submit_batch_job(batch_requests, batch_name)
            
            if self.wait_for_batch_completion(job_id):
                batch_samples = self.process_positive_results(job_id, batch_name)
                positive_results.extend(batch_samples)
                print(f"  → Total positive samples: {len(positive_results)}")
        
        # STEP 2: Generate HARD NEGATIVES
        print(f"\n{'='*60}")
        print("STEP 2: Generating HARD NEGATIVES")
        print(f"{'='*60}")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(skills_to_process))
            batch_skills = skills_to_process[start_idx:end_idx]
            
            batch_name = f"negative_batch_{batch_idx + 1}"
            print(f"\n📦 Batch {batch_idx + 1}/{num_batches} (skills {start_idx + 1}-{end_idx})")
            
            batch_requests = self.create_negative_batch_requests(batch_skills)
            job_id = self.submit_batch_job(batch_requests, batch_name)
            
            if self.wait_for_batch_completion(job_id):
                batch_samples = self.process_negative_results(job_id, batch_name)
                negative_results.extend(batch_samples)
                print(f"  → Total negative samples: {len(negative_results)}")
        
        # STEP 3: Combine
        print(f"\n{'='*60}")
        print("STEP 3: Combining Results")
        print(f"{'='*60}")
        
        final_dataset = self.combine_positive_negative(positive_results, negative_results)
        
        # Save
        output_file = OUTPUT_DIR / "skill_finetuning_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        
        stats = {
            'total_samples': len(final_dataset),
            'unique_skills': len(set(s['metadata']['skill_uri'] for s in final_dataset)),
            'variations_per_skill': VARIATIONS_PER_SKILL,
            'hard_negatives_per_sample': NUM_HARD_NEGATIVES,
            'created_at': datetime.now().isoformat(),
            'model': self.model_name
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
        print(f"\n📁 Output:")
        print(f"  - Dataset: {output_file}")
        print(f"  - Statistics: {stats_file}")
        print(f"\n🎯 Ready for fine-tuning!")


def main():
    print("="*60)
    print("ESCO Skills Fine-tuning Dataset Generator")
    print("Using Gemini Batch API - Separate Positive/Negative Generation")
    print("="*60)
    
    if not GEMINI_API_KEY:
        print("\n⚠️ WARNING: GEMINI_API_KEY not set!")
        return
    
    generator = SkillDatasetGenerator()
    
    try:
        generator.generate_dataset()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
