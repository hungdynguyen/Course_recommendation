import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import asyncio
from google import genai
from google.genai import types
import random
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import time

# Configure Gemini API
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Paths configuration
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to /root/courses_rec/
JDS_PATH = BASE_DIR / "data" / "raw" / "jds" / "job_jds_1.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "jd_cv_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Generation parameters
TARGET_SAMPLES = 1000  # Số lượng JD-CV pairs muốn tạo
MIN_MATCHING = 60  # % matching tối thiểu
MAX_MATCHING = 80  # % matching tối đa
LLM_MODEL = "gemini-3-flash-preview"
MAX_CONCURRENT_REQUESTS = 10  # Giảm xuống vì prompt phức tạp
CHECKPOINT_INTERVAL = 10  # Save progress every N JDs
MAX_RETRIES = 3

# Define CV JSON Schema for structured output (simplified to match JD fields)
CV_SCHEMA = {
    "type": "object",
    "properties": {
        "experience": {
            "type": "string",
            "description": "Years of experience or experience description (e.g., '3 years', '2-3 years', 'Fresher')"
        },
        "degree": {
            "type": "string",
            "description": "Education level (e.g., 'Bachelor', 'Master', 'College', 'High School')"
        },
        "technical_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of technical skills the candidate has"
        },
        "soft_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of soft skills the candidate has"
        },
        "skill_gaps": {
            "type": "object",
            "description": "Skills and experience gaps compared to JD requirements",
            "properties": {
                "missing_technical_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Technical skills required by JD but missing in CV"
                },
                "missing_soft_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Soft skills required by JD but missing in CV"
                },
                "experience_gap": {
                    "type": "string",
                    "description": "Description of experience gap if any (e.g., 'Need 1 more year', 'No gap', 'Need Master degree')"
                }
            },
            "required": ["missing_technical_skills", "missing_soft_skills", "experience_gap"]
        }
    },
    "required": ["experience", "degree", "technical_skills", "soft_skills", "skill_gaps"]
}

class CVGenerator:
    """Generate CVs from JDs using Gemini API with specified matching percentage"""
    
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.jds = []
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
    def load_jds(self, max_jds: Optional[int] = None) -> List[Dict]:
        """Load JDs from Excel file"""
        print("Loading JDs from Excel...")
        
        if not JDS_PATH.exists():
            raise FileNotFoundError(f"JDs file not found: {JDS_PATH}")
        
        df = pd.read_excel(JDS_PATH, engine='openpyxl')
        df = df.fillna('')  # Replace NaN with empty strings
        
        # Filter active JDs with meaningful content
        df = df[df['IsActive'] == 1]
        df = df[df['MainDescription'].str.len() > 100]  # At least 100 chars
        
        if max_jds:
            df = df.head(max_jds)
        
        self.jds = df.to_dict('records')
        
        print(f"Loaded {len(self.jds)} valid JDs from Excel")
        return self.jds
    
    def extract_jd_info(self, jd: Dict) -> Dict:
        """Extract key information from JD"""
        return {
            'job_id': jd.get('JobID', ''),
            'title': jd.get('Title', 'Unknown Position'),
            'description': jd.get('MainDescription', ''),
            'technical_skills': jd.get('TechnicalSkill', ''),
            'soft_skills': jd.get('SoftSkill', ''),
            'experience': jd.get('Experience', ''),
            'degree': jd.get('Degree', ''),
            'experience_years': jd.get('ExperienceYear', 0),
            'keywords': jd.get('Keywords', ''),
            'benefits': jd.get('Benefits', '')
        }
    
    async def generate_cv_from_jd(self, jd: Dict, jd_idx: int, semaphore: asyncio.Semaphore) -> Dict:
        """Generate a CV that matches the JD at a specific percentage (60-80%)"""
        async with semaphore:
            jd_info = self.extract_jd_info(jd)
            
            # Random matching percentage between 60-80%
            matching_percent = random.randint(MIN_MATCHING, MAX_MATCHING)
            
            # Build detailed prompt
            prompt = f"""You are an expert HR professional. Your task is to generate a candidate profile that matches approximately {matching_percent}% with the following job description.

**Job Description:**
- **Position:** {jd_info['title']}
- **Description:** {jd_info['description'][:1000]}  
- **Required Technical Skills:** {jd_info['technical_skills']}
- **Required Soft Skills:** {jd_info['soft_skills']}
- **Experience Required:** {jd_info['experience']}
- **Degree Required:** {jd_info['degree']}
- **Keywords:** {jd_info['keywords']}

**Your Task:**
Generate a candidate profile that matches **{matching_percent}%** of the above job requirements.

**Matching Guidelines:**

1. **If matching is 60-65%:** The candidate should have:
   - About 60% of the required technical skills (missing 40%)
   - Slightly lower experience level (e.g., if JD requires 3 years, candidate has 1-2 years)
   - May have a lower degree level
   - About 60% of soft skills

2. **If matching is 66-74%:** The candidate should have:
   - About 70% of required technical skills (missing 30%)
   - Experience level close to requirements (e.g., if JD requires 3 years, candidate has 2-3 years)
   - Degree level matches or slightly lower
   - About 70% of soft skills

3. **If matching is 75-80%:** The candidate should have:
   - About 75-80% of required technical skills (missing 20-25%)
   - Experience level matches requirements
   - Degree level matches
   - Most soft skills present

**IMPORTANT Instructions:**
- Generate realistic candidate data
- Include deliberate gaps to achieve the target matching percentage
- DO NOT make a perfect 100% match
- For technical_skills and soft_skills: return arrays of skill names that the candidate HAS
- For experience: return a string like "2 years", "3-5 years", "Fresher", etc.
- For degree: return education level like "Bachelor", "Master", "College", etc.
- For skill_gaps: clearly identify what is MISSING:
  * missing_technical_skills: list technical skills required by JD but NOT in candidate's profile
  * missing_soft_skills: list soft skills required by JD but NOT in candidate's profile
  * experience_gap: describe experience/education gap (e.g., "Needs 1 more year of experience", "Needs Master degree", "No significant gap")
"""
            
            for attempt in range(MAX_RETRIES):
                try:
                    # Use structured output with schema validation
                    config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CV_SCHEMA
                    )
                    
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    )
                    
                    # Direct JSON parsing - no need to clean up
                    cv_data = json.loads(response.text)
                    
                    return {
                        'success': True,
                        'jd_idx': jd_idx,
                        'jd_info': jd_info,
                        'target_matching_percent': matching_percent,
                        'cv': cv_data,
                        'generated_at': datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        print(f"  ❌ Failed to generate CV for JD {jd_idx}: {jd_info['title']} - {e}")
                        return {
                            'success': False,
                            'jd_idx': jd_idx,
                            'jd_info': jd_info,
                            'error': str(e)
                        }
    
    async def generate_dataset_batch(self, jds: List[Dict], start_idx: int = 0) -> List[Dict]:
        """Generate CVs for all JDs concurrently"""
        print(f"\n{'='*60}")
        print("Generating JD-CV Pairs Dataset")
        print(f"{'='*60}")
        print(f"  - Total JDs: {len(jds)}")
        print(f"  - Matching range: {MIN_MATCHING}%-{MAX_MATCHING}%")
        print(f"  - Concurrent requests: {MAX_CONCURRENT_REQUESTS}")
        print(f"  - Model: {self.model_name}")
        
        checkpoint_file = OUTPUT_DIR / "jd_cv_checkpoint.json"
        final_dataset_file = OUTPUT_DIR / "jd_cv_pairs.json"
        
        # Load existing progress
        results = []
        processed_ids = set()
        
        if final_dataset_file.exists():
            try:
                with open(final_dataset_file, 'r', encoding='utf-8') as f:
                    existing_dataset = json.load(f)
                processed_ids = {item['jd_info']['job_id'] for item in existing_dataset if item.get('success', False)}
                print(f"  📂 Loaded {len(processed_ids)} successfully processed JDs from final dataset")
            except Exception as e:
                print(f"  ⚠️ Could not load final dataset: {e}")
        
        if not processed_ids and checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                processed_ids = {r['jd_info']['job_id'] for r in results if r.get('success', False)}
                print(f"  📂 Loaded checkpoint: {len(results)} JDs already processed")
            except Exception as e:
                print(f"  ⚠️ Could not load checkpoint: {e}")
                results = []
        
        # Filter out already processed JDs
        jds_to_process = [j for j in jds if j.get('JobID', '') not in processed_ids]
        
        if not jds_to_process:
            print(f"  ✅ All JDs already processed!")
            return results
        
        print(f"  - Remaining JDs: {len(jds_to_process)}")
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = []
        
        for idx, jd in enumerate(jds_to_process, start=start_idx + len(results)):
            task = self.generate_cv_from_jd(jd, idx, semaphore)
            tasks.append(task)
        
        # Process with progress bar
        checkpoint_counter = 0
        
        with tqdm(total=len(tasks), desc="Generating CVs", unit="cv") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                checkpoint_counter += 1
                
                pbar.update(1)
                success = result.get('success', False)
                pbar.set_postfix({
                    'total': len(results),
                    'current': 'OK' if success else 'FAIL'
                })
                
                # Save checkpoint
                if checkpoint_counter % CHECKPOINT_INTERVAL == 0:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Final checkpoint
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        successful_results = [r for r in results if r.get('success', False)]
        print(f"\n✅ Completed: {len(successful_results)}/{len(results)} successful")
        
        return successful_results
    
    async def generate_dataset(self, max_jds: Optional[int] = None):
        """Main generation function"""
        if not self.jds:
            self.load_jds(max_jds or TARGET_SAMPLES)
        
        print(f"\n💻 JD-CV Pair Generation Mode")
        print(f"  - JDs to process: {len(self.jds)}")
        print(f"  - Matching range: {MIN_MATCHING}%-{MAX_MATCHING}%")
        print(f"  - Concurrent requests: {MAX_CONCURRENT_REQUESTS}")
        print(f"  - Model: {self.model_name}")
        
        start_time = time.time()
        
        # Generate CVs
        results = await self.generate_dataset_batch(self.jds)
        
        # Save final dataset
        output_file = OUTPUT_DIR / "jd_cv_pairs.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save simplified version for easy viewing
        simplified_data = []
        for item in results:
            if not item.get('success', False):
                continue
            
            skill_gaps = item['cv'].get('skill_gaps', {})
            simplified_data.append({
                'jd_title': item['jd_info']['title'],
                'jd_experience': item['jd_info']['experience'],
                'jd_degree': item['jd_info']['degree'],
                'jd_technical_skills': item['jd_info']['technical_skills'],
                'jd_soft_skills': item['jd_info']['soft_skills'],
                'target_matching': f"{item['target_matching_percent']}%",
                'cv_experience': item['cv'].get('experience', ''),
                'cv_degree': item['cv'].get('degree', ''),
                'cv_technical_skills': '; '.join(item['cv'].get('technical_skills', [])),
                'cv_soft_skills': '; '.join(item['cv'].get('soft_skills', [])),
                'missing_technical_skills': '; '.join(skill_gaps.get('missing_technical_skills', [])),
                'missing_soft_skills': '; '.join(skill_gaps.get('missing_soft_skills', [])),
                'experience_gap': skill_gaps.get('experience_gap', '')
            })
        
        # Save as Excel
        df_simple = pd.DataFrame(simplified_data)
        excel_file = OUTPUT_DIR / "jd_cv_pairs_simple.xlsx"
        df_simple.to_excel(excel_file, index=False, engine='openpyxl')
        
        # Save statistics
        elapsed_time = time.time() - start_time
        matching_distribution = {}
        for item in results:
            if item.get('success', False):
                pct = item['target_matching_percent']
                matching_distribution[pct] = matching_distribution.get(pct, 0) + 1
        
        stats = {
            'total_pairs': len(results),
            'successful_pairs': len([r for r in results if r.get('success', False)]),
            'failed_pairs': len([r for r in results if not r.get('success', False)]),
            'matching_range': f"{MIN_MATCHING}%-{MAX_MATCHING}%",
            'matching_distribution': matching_distribution,
            'processing_time_seconds': elapsed_time,
            'created_at': datetime.now().isoformat(),
            'model': self.model_name,
            'concurrent_requests': MAX_CONCURRENT_REQUESTS
        }
        
        stats_file = OUTPUT_DIR / "jd_cv_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print("✅ COMPLETED!")
        print(f"{'='*60}")
        print(f"\n📊 Statistics:")
        print(f"  - Total pairs: {stats['total_pairs']}")
        print(f"  - Successful: {stats['successful_pairs']}")
        print(f"  - Failed: {stats['failed_pairs']}")
        print(f"  - Success rate: {stats['successful_pairs']/stats['total_pairs']*100:.1f}%")
        print(f"  - Matching range: {stats['matching_range']}")
        print(f"  - Processing time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
        print(f"\n📁 Output:")
        print(f"  - Dataset: {output_file}")
        print(f"  - Simplified Excel: {excel_file}")
        print(f"  - Statistics: {stats_file}")
        print(f"\n🎯 Ready for Step 2: Course Recommendation!")


def main():
    print("="*60)
    print("JD-CV Pairs Generator")
    print("Generate CVs with 60-80% matching to JDs")
    print("="*60)
    
    if not GEMINI_API_KEY:
        print("\n⚠️ WARNING: GEMINI_API_KEY not set!")
        return
    
    generator = CVGenerator()
    
    try:
        asyncio.run(generator.generate_dataset())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
