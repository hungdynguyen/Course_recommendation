import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Set
import pandas as pd
import asyncio
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import requests
import time

# Configure paths
BASE_DIR = Path(__file__).parent.parent.parent
JD_CV_PAIRS_PATH = BASE_DIR / "data" / "processed" / "jd_cv_pairs" / "jd_cv_pairs.json"
COURSES_DIR = BASE_DIR / "data" / "Data_Courses_Json"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "course_recommendations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TOP_K_PER_METHOD = 30  
BATCH_SIZE = 2


class CourseRecommender:
    """Generate course recommendations using 2 methods: API + Vector Search"""
    
    def __init__(self):
        self.courses = []
        self.course_embeddings = None
        self.embedding_model = None
        self.jd_cv_pairs = []
        
    def load_courses(self) -> List[Dict]:
        """Load all courses from Data_Courses_Json folder"""
        print("\n📚 Loading courses from Data_Courses_Json...")
        courses = []
        
        for faculty_folder in COURSES_DIR.iterdir():
            if not faculty_folder.is_dir():
                continue
                
            for course_file in faculty_folder.glob("*.json"):
                try:
                    with open(course_file, 'r', encoding='utf-8') as f:
                        course = json.load(f)
                        courses.append(course)
                except Exception as e:
                    print(f"  ⚠️ Error loading {course_file.name}: {e}")
        
        self.courses = courses
        print(f"✓ Loaded {len(self.courses)} courses")
        return courses
    
    def prepare_course_text_for_embedding(self, course: Dict) -> str:
        """Prepare course text for embedding (title + skill outcomes)"""
        title = course.get('title', '')
        
        # Extract skill names and descriptions from skill_outcomes
        skill_texts = []
        for skill in course.get('skill_outcomes', []):
            skill_name = skill.get('skill_name', '')
            outcome_desc = skill.get('outcome_description', '')
            if skill_name:
                skill_texts.append(skill_name)
            if outcome_desc:
                skill_texts.append(outcome_desc)
        
        # Combine title and skills
        text = f"{title}. " + " ".join(skill_texts)
        return text.strip()
    
    def embed_courses(self):
        """Embed all courses using sentence transformer"""
        print("\n🔢 Embedding courses...")
        print(f"  - Model: {EMBEDDING_MODEL}")
        
        # Load model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        
        # Prepare texts
        course_texts = [self.prepare_course_text_for_embedding(c) for c in self.courses]
        
        # Embed
        print(f"  - Encoding {len(course_texts)} courses...")
        self.course_embeddings = self.embedding_model.encode(
            course_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=BATCH_SIZE
        )
        
        print(f"✓ Course embeddings shape: {self.course_embeddings.shape}")
    
    def load_jd_cv_pairs(self) -> List[Dict]:
        """Load JD-CV pairs from previous step"""
        print(f"\n📂 Loading JD-CV pairs from {JD_CV_PAIRS_PATH}...")
        
        if not JD_CV_PAIRS_PATH.exists():
            raise FileNotFoundError(f"JD-CV pairs file not found: {JD_CV_PAIRS_PATH}")
        
        with open(JD_CV_PAIRS_PATH, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        
        # Filter successful pairs
        successful_pairs = [p for p in pairs if p.get('success', False)]
        self.jd_cv_pairs = successful_pairs
        
        print(f"✓ Loaded {len(successful_pairs)} successful JD-CV pairs")
        return successful_pairs
    
    def recommend_via_api(self, skill_gaps: Dict) -> List[Dict]:
        """Method 1: Recommend courses via API"""
        missing_technical = skill_gaps.get('missing_technical_skills', [])
        missing_soft = skill_gaps.get('missing_soft_skills', [])
        
        all_missing_skills = missing_technical + missing_soft
        
        if not all_missing_skills:
            return []
        
        try:
            # Call API
            response = requests.post(
                f"{API_BASE_URL}/recommendations/courses",
                json={"skill_names": all_missing_skills},
                params={"max_courses": TOP_K_PER_METHOD},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                courses = data.get('recommended_courses', [])
                
                # Extract course info
                result = []
                for course in courses[:TOP_K_PER_METHOD]:
                    result.append({
                        'course_id': course.get('course_id', ''),
                        'title': course.get('course_title', ''),  
                        'method': 'api',
                        'score': course.get('similarity_score', 0.0)  
                    })
                return result
            else:
                print(f"  ⚠️ API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"  ⚠️ API call failed: {e}")
            return []
    
    def recommend_via_vector_search(self, skill_gaps: Dict) -> List[Dict]:
        """Method 2: Recommend courses via vector search on skill gaps"""
        missing_technical = skill_gaps.get('missing_technical_skills', [])
        missing_soft = skill_gaps.get('missing_soft_skills', [])
        experience_gap = skill_gaps.get('experience_gap', '')
        
        # Build query text from gaps
        query_parts = []
        if missing_technical:
            query_parts.append("Technical skills: " + ", ".join(missing_technical))
        if missing_soft:
            query_parts.append("Soft skills: " + ", ".join(missing_soft))
        if experience_gap and experience_gap.lower() != 'no gap':
            query_parts.append(experience_gap)
        
        if not query_parts:
            return []
        
        query_text = ". ".join(query_parts)
        
        # Embed query
        query_embedding = self.embedding_model.encode(
            query_text,
            convert_to_tensor=True
        )
        
        # Compute similarity
        similarities = util.cos_sim(query_embedding, self.course_embeddings)[0]
        
        # Get top K
        top_k = torch.topk(similarities, k=min(TOP_K_PER_METHOD, len(self.courses)))
        
        result = []
        for idx, score in zip(top_k.indices, top_k.values):
            course = self.courses[idx.item()]
            result.append({
                'course_id': course.get('course_id', ''),
                'title': course.get('title', ''),
                'method': 'vector_search',
                'score': float(score.item())
            })
        
        return result
    
    def merge_recommendations(self, api_courses: List[Dict], vector_courses: List[Dict]) -> List[Dict]:
        """Merge and deduplicate courses from both methods"""
        # Use dict to deduplicate by course_id
        merged = {}
        
        # Add API courses first (higher priority in dedup)
        for course in api_courses:
            course_id = course['course_id']
            if course_id and course_id not in merged:
                merged[course_id] = {
                    **course,
                    'methods': ['api'],
                    'api_score': course.get('score', 0.0)
                }
        
        # Add vector search courses
        for course in vector_courses:
            course_id = course['course_id']
            if course_id:
                if course_id in merged:
                    # Update existing - add method
                    merged[course_id]['methods'].append('vector_search')
                    merged[course_id]['vector_score'] = course.get('score', 0.0)
                else:
                    # Add new
                    merged[course_id] = {
                        **course,
                        'methods': ['vector_search'],
                        'vector_score': course.get('score', 0.0)
                    }
        
        # Convert to list and sort by priority (courses from both methods first)
        result = list(merged.values())
        result.sort(key=lambda x: (len(x['methods']), x.get('api_score', 0) + x.get('vector_score', 0)), reverse=True)
        
        return result
    
    def generate_recommendations(self):
        """Main function to generate recommendations for all JD-CV pairs"""
        print("\n" + "="*60)
        print("STEP 2: COURSE RECOMMENDATION")
        print("="*60)
        
        # Load data
        self.load_courses()
        self.embed_courses()
        self.load_jd_cv_pairs()
        
        print(f"\n🎯 Generating recommendations for {len(self.jd_cv_pairs)} pairs...")
        print(f"  - API recommendations: {TOP_K_PER_METHOD} courses per pair")
        print(f"  - Vector search: {TOP_K_PER_METHOD} courses per pair")
        print(f"  - Merging and deduplicating...")
        
        results = []
        
        for idx, pair in enumerate(tqdm(self.jd_cv_pairs, desc="Recommending courses")):
            jd_info = pair.get('jd_info', {})
            cv = pair.get('cv', {})
            skill_gaps = cv.get('skill_gaps', {})
            
            # Method 1: API
            api_courses = self.recommend_via_api(skill_gaps)
            
            # Method 2: Vector search
            vector_courses = self.recommend_via_vector_search(skill_gaps)
            
            # Merge
            merged_courses = self.merge_recommendations(api_courses, vector_courses)
            
            # Store result
            result = {
                'pair_id': idx,
                'jd_title': jd_info.get('title', ''),
                'jd_id': jd_info.get('job_id', ''),
                'jd_info': {
                    'description': jd_info.get('description', ''),
                    'technical_skills': jd_info.get('technical_skills', ''),
                    'soft_skills': jd_info.get('soft_skills', ''),
                    'experience': jd_info.get('experience', ''),
                    'degree': jd_info.get('degree', ''),
                    'keywords': jd_info.get('keywords', ''),
                },
                'cv_experience': cv.get('experience', ''),
                'cv_degree': cv.get('degree', ''),
                'cv_info': {
                    'technical_skills': cv.get('technical_skills', []),
                    'soft_skills': cv.get('soft_skills', []),
                },
                'skill_gaps': skill_gaps,
                'recommendations': {
                    'api_count': len(api_courses),
                    'vector_count': len(vector_courses),
                    'merged_count': len(merged_courses),
                    'courses': merged_courses
                },
                'target_matching_percent': pair.get('target_matching_percent', 0)
            }
            results.append(result)
            
            # Small delay to avoid overwhelming API
            if api_courses:
                time.sleep(0.1)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save recommendation results"""
        print(f"\n💾 Saving results...")
        
        # Save full results as JSON
        output_file = OUTPUT_DIR / "course_recommendations.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Full results: {output_file}")
        
        # Save simplified version as Excel
        simplified_data = []
        for result in results:
            base_row = {
                'pair_id': result['pair_id'],
                'jd_title': result['jd_title'],
                'cv_experience': result['cv_experience'],
                'cv_degree': result['cv_degree'],
                'target_matching': f"{result['target_matching_percent']}%",
                'missing_technical_skills': '; '.join(result['skill_gaps'].get('missing_technical_skills', [])),
                'missing_soft_skills': '; '.join(result['skill_gaps'].get('missing_soft_skills', [])),
                'experience_gap': result['skill_gaps'].get('experience_gap', ''),
                'api_courses_count': result['recommendations']['api_count'],
                'vector_courses_count': result['recommendations']['vector_count'],
                'total_unique_courses': result['recommendations']['merged_count'],
                'top_5_courses': '; '.join([c['title'] for c in result['recommendations']['courses'][:5]])
            }
            simplified_data.append(base_row)
        
        df = pd.DataFrame(simplified_data)
        excel_file = OUTPUT_DIR / "course_recommendations_simple.xlsx"
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"  ✓ Simplified Excel: {excel_file}")
        
        # Save statistics
        total_pairs = len(results)
        avg_api_courses = sum(r['recommendations']['api_count'] for r in results) / total_pairs
        avg_vector_courses = sum(r['recommendations']['vector_count'] for r in results) / total_pairs
        avg_merged_courses = sum(r['recommendations']['merged_count'] for r in results) / total_pairs
        
        stats = {
            'total_pairs': total_pairs,
            'total_courses_available': len(self.courses),
            'average_api_courses_per_pair': round(avg_api_courses, 2),
            'average_vector_courses_per_pair': round(avg_vector_courses, 2),
            'average_merged_courses_per_pair': round(avg_merged_courses, 2),
            'top_k_per_method': TOP_K_PER_METHOD,
            'embedding_model': EMBEDDING_MODEL,
            'api_base_url': API_BASE_URL
        }
        
        stats_file = OUTPUT_DIR / "recommendation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Statistics: {stats_file}")
        
        print(f"\n📊 Statistics:")
        print(f"  - Total pairs processed: {total_pairs}")
        print(f"  - Avg API courses per pair: {avg_api_courses:.1f}")
        print(f"  - Avg Vector courses per pair: {avg_vector_courses:.1f}")
        print(f"  - Avg Merged courses per pair: {avg_merged_courses:.1f}")
        print(f"\n✅ STEP 2 COMPLETED!")


def main():
    print("="*60)
    print("STEP 2: Course Recommendation Generator")
    print("Using API + Vector Search Methods")
    print("="*60)
    
    recommender = CourseRecommender()
    
    try:
        recommender.generate_recommendations()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
