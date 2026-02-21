"""
Offline script to pre-compute all search results and save to cache file.
Run this once, then the web tool will load cached results instantly.

Usage:
    python precompute_search_cache.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directories to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "services" / "data_factory"))

from src.embeddings.embedding_service import EmbeddingService
from src.services.reranker_service import RerankerService
from src.utils.config_utils import load_config

# Paths
DATA_DIR = BASE_DIR / "data"
COURSES_DIR = DATA_DIR / "Data_Courses_Json"
ESCO_SKILLS_PATH = DATA_DIR / "raw" / "skill_taxonomy" / "skills_en.csv"
OUTPUT_DIR = DATA_DIR / "processed" / "training_dataset"
ESCO_EMBEDDINGS_CACHE = OUTPUT_DIR / "esco_embeddings_cache.npz"
SEARCH_RESULTS_CACHE = OUTPUT_DIR / "search_results_cache.json"

def load_course_skills():
    """Load all skills from course JSON files"""
    print("Loading course skills...")
    all_skills = []
    skill_metadata = {}
    
    for dept_folder in COURSES_DIR.iterdir():
        if not dept_folder.is_dir():
            continue
        
        for course_file in dept_folder.glob("*.json"):
            try:
                with open(course_file, 'r', encoding='utf-8') as f:
                    course_data = json.load(f)
                
                skills = course_data.get('skill_outcomes', [])
                course_name = course_data.get('title', course_file.stem)
                
                for skill in skills:
                    if isinstance(skill, str):
                        skill_text = skill
                        description = ''
                    else:
                        skill_text = skill.get('skill_name', '')
                        description = skill.get('outcome_description', '')
                    
                    if skill_text and skill_text not in all_skills:
                        all_skills.append(skill_text)
                        skill_metadata[skill_text] = {
                            'course_name': course_name,
                            'description': description
                        }
                        
            except Exception as e:
                print(f"Error loading {course_file}: {e}")
    
    print(f"Loaded {len(all_skills)} unique course skills")
    return all_skills, skill_metadata

def load_esco_skills():
    """Load ESCO skills taxonomy"""
    print("Loading ESCO skills...")
    df = pd.read_csv(ESCO_SKILLS_PATH, encoding='utf-8')
    df = df.fillna('')
    
    esco_skills = []
    for _, row in df.iterrows():
        skill_name = row.get('preferredLabel', '')
        description = row.get('description', '')
        definition = row.get('definition', '')
        scope_note = row.get('scopeNote', '')
        
        desc_parts = [p for p in [description, definition, scope_note] if p]
        full_description = ' '.join(desc_parts).strip()
        if not full_description:
            full_description = skill_name
        
        esco_skills.append({
            'uri': row.get('conceptUri', ''),
            'name': skill_name,
            'description': description,
            'full_text': f"{skill_name}. {full_description}",
            'skill_type': row.get('skillType', '')
        })
    
    print(f"Loaded {len(esco_skills)} ESCO skills")
    return esco_skills

def build_esco_embeddings(esco_skills, embedding_service):
    """Build embeddings for all ESCO skills"""
    # Try to load from cache first
    if ESCO_EMBEDDINGS_CACHE.exists():
        print("Loading ESCO embeddings from cache...")
        try:
            data = np.load(ESCO_EMBEDDINGS_CACHE)
            embeddings = data['embeddings']
            print(f"Loaded cached embeddings: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"Failed to load cache: {e}. Rebuilding...")
    
    # Build embeddings
    print("Building ESCO skill embeddings...")
    esco_texts = [s['full_text'] for s in esco_skills]
    
    embeddings = embedding_service.encode(
        esco_texts,
        show_progress=True
    )
    
    print(f"Built embeddings: {embeddings.shape}")
    
    # Save to cache
    print("Saving embeddings to cache...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ESCO_EMBEDDINGS_CACHE, embeddings=embeddings)
    print("Cache saved!")
    
    return embeddings

def batch_compute_all_matches(course_skills, esco_skills, esco_embeddings, 
                              embedding_service, reranker_service, top_k=50,
                              use_reranker=True, num_candidates=100):
    """Compute matches for all course skills in batches
    
    Args:
        use_reranker: If False, skip reranking and use embedding scores only (MUCH faster)
        num_candidates: Number of candidates to rerank per skill (default 100, was 200)
    """
    print(f"\n{'='*60}")
    print(f"Computing search results for {len(course_skills)} skills...")
    print(f"Reranking: {'Enabled' if use_reranker and reranker_service else 'Disabled (faster)'}")
    print(f"Candidates per skill: {num_candidates}")
    print(f"{'='*60}\n")
    
    # Step 1: Encode all course skills at once
    print("Step 1: Encoding all course skills...")
    course_embeddings = embedding_service.encode(
        course_skills,
        show_progress=True
    )
    print(f"Encoded {len(course_skills)} course skills")
    
    # Step 2: Compute similarities for all at once
    print("\nStep 2: Computing similarities for all skills...")
    # Normalize embeddings for faster cosine similarity
    esco_embeddings_norm = esco_embeddings / np.linalg.norm(esco_embeddings, axis=1, keepdims=True)
    course_embeddings_norm = course_embeddings / np.linalg.norm(course_embeddings, axis=1, keepdims=True)
    
    # Matrix multiplication: (num_courses, embedding_dim) x (embedding_dim, num_esco) = (num_courses, num_esco)
    all_similarities = np.dot(course_embeddings_norm, esco_embeddings_norm.T)
    print(f"Computed similarity matrix: {all_similarities.shape}")
    
    # Step 3: Get top candidates for each skill
    print(f"\nStep 3: Selecting top {num_candidates} candidates for each skill...")
    all_results = {}
    
    for idx, skill_text in enumerate(tqdm(course_skills, desc="Processing skills")):
        similarities = all_similarities[idx]
        
        # Get top candidates
        top_indices = np.argsort(similarities)[::-1][:num_candidates]
        
        candidates = []
        for esco_idx in top_indices:
            candidates.append({
                'esco_skill': esco_skills[esco_idx],
                'embedding_score': float(similarities[esco_idx]),
                'index': int(esco_idx)
            })
        
        # Rerank if reranker is available and enabled
        if use_reranker and reranker_service:
            # Prepare pairs for reranking
            pairs = [[skill_text, c['esco_skill']['full_text']] for c in candidates]
            
            # Get rerank scores
            rerank_scores = reranker_service.compute_scores(pairs)
            
            # Update scores
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(rerank_scores[i])
                candidate['final_score'] = float(rerank_scores[i])
                candidate['similarity'] = float(rerank_scores[i])
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        else:
            # Use embedding score only
            for candidate in candidates:
                candidate['final_score'] = candidate['embedding_score']
                candidate['similarity'] = candidate['embedding_score']
        
        # Store top K results
        cache_key = f"{skill_text}:{top_k}"
        all_results[cache_key] = candidates[:top_k]
    
    return all_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-compute search results cache')
    parser.add_argument('--use-rerank', action='store_true',
                       help='Enable reranking (slower but more accurate)')
    parser.add_argument('--candidates', type=int, default=100,
                       help='Number of candidates to rerank per skill (default: 100)')
    args = parser.parse_args()
    
    print("="*60)
    print("Pre-computing Search Results Cache")
    print("="*60)
    if not args.use_rerank:
        print("⚡ FAST MODE: Reranking disabled (default)")
    else:
        print("🔄 Using reranking (slower but more accurate)")
    print()
    
    # Load configuration
    print("Loading configuration...")
    settings = load_config()
    
    # Initialize services
    print("Initializing embedding service...")
    embedding_service = EmbeddingService(settings.embedding)
    
    reranker_service = None
    if settings.reranker.enabled and args.use_rerank:
        print("Initializing reranker service...")
        reranker_service = RerankerService(settings.reranker)
    else:
        print("⚡ Skipping reranker initialization (fast mode)")
    
    # Load data
    course_skills, _ = load_course_skills()
    esco_skills = load_esco_skills()
    
    # Build ESCO embeddings
    esco_embeddings = build_esco_embeddings(esco_skills, embedding_service)
    
    # Compute all matches
    all_results = batch_compute_all_matches(
        course_skills, 
        esco_skills, 
        esco_embeddings,
        embedding_service,
        reranker_service,
        top_k=50,
        use_reranker=args.use_rerank,
        num_candidates=args.candidates
    )
    
    # Save to cache file
    print(f"\n{'='*60}")
    print(f"Saving {len(all_results)} results to cache file...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEARCH_RESULTS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Cache saved to: {SEARCH_RESULTS_CACHE}")
    print("\nUsage options:")
    print("  Fast mode (default, ~2 min):       python precompute_search_cache.py")
    print("  With reranking (slower):           python precompute_search_cache.py --use-rerank")
    print("  Custom candidates:                 python precompute_search_cache.py --candidates 50")
    print("\n✓ Pre-computation complete!")
    print("You can now start the web tool - it will load cached results instantly.")

if __name__ == '__main__':
    main()
