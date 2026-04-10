"""
Map skills from JD-CV pairs to ESCO taxonomy
This script should be run AFTER generate_cv_by_jds.py and BEFORE generate_course_recommendations.py
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import numpy as np
from tqdm import tqdm
import re

# Add service paths
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "services" / "data_factory"))

from src.embeddings.embedding_service import EmbeddingService
from src.io.esco_embedding_loader import load_esco_embeddings
from src.utils.config_utils import load_config

# Paths
JD_CV_PAIRS_PATH = BASE_DIR / "data" / "processed" / "jd_cv_pairs" / "jd_cv_pairs.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "jd_cv_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
TOP_K_MATCHES = 3  # Number of ESCO skills to match per raw skill
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score


def extract_all_skills_from_pairs(pairs: List[Dict]) -> Set[str]:
    """Extract all unique skill names from JD-CV pairs"""
    skills = set()
    
    for pair in pairs:
        if not pair.get('success', False):
            continue
            
        jd_info = pair.get('jd_info', {})
        cv = pair.get('cv', {})
        skill_gaps = cv.get('skill_gaps', {})
        
        # From JD (strings that need splitting)
        jd_tech = jd_info.get('technical_skills', '')
        jd_soft = jd_info.get('soft_skills', '')
        
        # Split by common delimiters
        if jd_tech:
            skills.update(_split_skill_string(jd_tech))
        if jd_soft:
            skills.update(_split_skill_string(jd_soft))
        
        # From CV (arrays)
        cv_tech = cv.get('technical_skills', [])
        cv_soft = cv.get('soft_skills', [])
        skills.update(cv_tech)
        skills.update(cv_soft)
        
        # From skill gaps (arrays)
        missing_tech = skill_gaps.get('missing_technical_skills', [])
        missing_soft = skill_gaps.get('missing_soft_skills', [])
        skills.update(missing_tech)
        skills.update(missing_soft)
    
    # Clean and filter
    cleaned_skills = set()
    for skill in skills:
        cleaned = skill.strip()
        if cleaned and len(cleaned) > 2:  # Skip very short strings
            cleaned_skills.add(cleaned)
    
    return cleaned_skills


def _split_skill_string(skill_str: str) -> List[str]:
    """Split a skill string by common delimiters"""
    # Split by: comma, semicolon, pipe, newline, bullet points
    delimiters = r'[,;|\n•\-]'
    parts = re.split(delimiters, skill_str)
    
    # Clean each part
    cleaned = []
    for part in parts:
        part = part.strip()
        # Remove leading numbers and dots (1. 2. etc)
        part = re.sub(r'^\d+\.?\s*', '', part)
        # Remove parentheses content if it's just optional info
        part = re.sub(r'\([^)]*\)', '', part).strip()
        if part and len(part) > 2:
            cleaned.append(part)
    
    return cleaned


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return vectors / norms


def map_skills_to_esco(
    skill_names: List[str],
    esco_metadata: List[Dict],
    esco_embeddings: np.ndarray,
    embedding_service: EmbeddingService
) -> Dict[str, List[Dict]]:
    """Map raw skill names to ESCO skills using embeddings"""
    
    print(f"\n🔢 Encoding {len(skill_names)} raw skills...")
    
    # Encode raw skills
    skill_embeddings = embedding_service.encode(skill_names)
    
    # Normalize
    normalized_skills = normalize_vectors(skill_embeddings)
    normalized_esco = normalize_vectors(esco_embeddings)
    
    # Compute similarity matrix
    print(f"📊 Computing similarity matrix...")
    similarity_matrix = normalized_skills @ normalized_esco.T
    
    # Map each skill
    mapping = {}
    print(f"\n🔗 Mapping skills to ESCO...")
    
    for idx, skill_name in enumerate(tqdm(skill_names, desc="Mapping skills")):
        similarities = similarity_matrix[idx]
        
        # Get top K matches
        top_k_indices = np.argsort(similarities)[-TOP_K_MATCHES:][::-1]
        
        matches = []
        for esco_idx in top_k_indices:
            similarity = float(similarities[esco_idx])
            
            if similarity >= SIMILARITY_THRESHOLD:
                esco_skill = esco_metadata[esco_idx]
                matches.append({
                    'skill_id': esco_skill.get('skill_id', ''),
                    'label': esco_skill.get('preferred_label', ''),
                    'description': esco_skill.get('description', ''),
                    'skill_type': esco_skill.get('skill_type', ''),
                    'similarity': similarity
                })
        
        if matches:
            mapping[skill_name] = matches
    
    return mapping


def update_jdcv_pairs_with_esco_ids(
    pairs: List[Dict],
    skill_mapping: Dict[str, List[Dict]]
) -> List[Dict]:
    """Update JD-CV pairs with ESCO skill IDs"""
    
    print(f"\n✏️ Updating JD-CV pairs with ESCO IDs...")
    
    updated_pairs = []
    
    for pair in tqdm(pairs, desc="Updating pairs"):
        if not pair.get('success', False):
            updated_pairs.append(pair)
            continue
        
        jd_info = pair.get('jd_info', {})
        cv = pair.get('cv', {})
        skill_gaps = cv.get('skill_gaps', {})
        
        # Map JD skills
        jd_tech_raw = _split_skill_string(jd_info.get('technical_skills', ''))
        jd_soft_raw = _split_skill_string(jd_info.get('soft_skills', ''))
        
        jd_tech_esco_ids = []
        for skill in jd_tech_raw:
            if skill in skill_mapping and skill_mapping[skill]:
                # Take best match
                jd_tech_esco_ids.append(skill_mapping[skill][0]['skill_id'])
        
        jd_soft_esco_ids = []
        for skill in jd_soft_raw:
            if skill in skill_mapping and skill_mapping[skill]:
                jd_soft_esco_ids.append(skill_mapping[skill][0]['skill_id'])
        
        # Map CV skills
        cv_tech_raw = cv.get('technical_skills', [])
        cv_soft_raw = cv.get('soft_skills', [])
        
        cv_tech_esco_ids = []
        for skill in cv_tech_raw:
            if skill in skill_mapping and skill_mapping[skill]:
                cv_tech_esco_ids.append(skill_mapping[skill][0]['skill_id'])
        
        cv_soft_esco_ids = []
        for skill in cv_soft_raw:
            if skill in skill_mapping and skill_mapping[skill]:
                cv_soft_esco_ids.append(skill_mapping[skill][0]['skill_id'])
        
        # Map skill gaps
        missing_tech_raw = skill_gaps.get('missing_technical_skills', [])
        missing_soft_raw = skill_gaps.get('missing_soft_skills', [])
        
        missing_tech_esco_ids = []
        for skill in missing_tech_raw:
            if skill in skill_mapping and skill_mapping[skill]:
                missing_tech_esco_ids.append(skill_mapping[skill][0]['skill_id'])
        
        missing_soft_esco_ids = []
        for skill in missing_soft_raw:
            if skill in skill_mapping and skill_mapping[skill]:
                missing_soft_esco_ids.append(skill_mapping[skill][0]['skill_id'])
        
        # Update pair with ESCO IDs
        updated_pair = pair.copy()
        
        # Add esco_ids fields alongside raw text fields
        updated_pair['jd_info']['technical_skills_esco_ids'] = jd_tech_esco_ids
        updated_pair['jd_info']['soft_skills_esco_ids'] = jd_soft_esco_ids
        
        updated_pair['cv']['technical_skills_esco_ids'] = cv_tech_esco_ids
        updated_pair['cv']['soft_skills_esco_ids'] = cv_soft_esco_ids
        
        updated_pair['cv']['skill_gaps']['missing_technical_skills_esco_ids'] = missing_tech_esco_ids
        updated_pair['cv']['skill_gaps']['missing_soft_skills_esco_ids'] = missing_soft_esco_ids
        
        updated_pairs.append(updated_pair)
    
    return updated_pairs


def main():
    print("="*60)
    print("JD-CV Skills to ESCO Mapping")
    print("="*60)
    
    # Load JD-CV pairs
    print(f"\n📂 Loading JD-CV pairs from {JD_CV_PAIRS_PATH}...")
    if not JD_CV_PAIRS_PATH.exists():
        print(f"❌ Error: {JD_CV_PAIRS_PATH} not found!")
        print("   Please run generate_cv_by_jds.py first.")
        return
    
    with open(JD_CV_PAIRS_PATH, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    successful_pairs = [p for p in pairs if p.get('success', False)]
    print(f"✓ Loaded {len(successful_pairs)} successful pairs")
    
    # Extract all unique skills
    print(f"\n🔍 Extracting unique skills from pairs...")
    unique_skills = extract_all_skills_from_pairs(successful_pairs)
    unique_skills_list = sorted(list(unique_skills))
    print(f"✓ Found {len(unique_skills_list)} unique skills")
    
    # Save extracted skills for reference
    skills_file = OUTPUT_DIR / "extracted_skills.json"
    with open(skills_file, 'w', encoding='utf-8') as f:
        json.dump(unique_skills_list, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved to {skills_file}")
    
    # Load ESCO embeddings
    print(f"\n📚 Loading ESCO embeddings...")
    settings = load_config()
    esco_metadata, esco_embeddings = load_esco_embeddings(
        settings.paths.processed_embeddings_dir
    )
    print(f"✓ Loaded {len(esco_metadata)} ESCO skills")
    
    # Initialize embedding service
    print(f"\n🤖 Initializing embedding service...")
    embedding_service = EmbeddingService(settings.embedding)
    model_name = settings.embedding.model_path or settings.embedding.model_name
    print(f"✓ Model: {model_name}")
    
    # Map skills
    skill_mapping = map_skills_to_esco(
        unique_skills_list,
        esco_metadata,
        esco_embeddings,
        embedding_service
    )
    
    # Statistics
    mapped_count = len(skill_mapping)
    unmapped_count = len(unique_skills_list) - mapped_count
    
    print(f"\n📊 Mapping Results:")
    print(f"  - Total unique skills: {len(unique_skills_list)}")
    print(f"  - Mapped to ESCO: {mapped_count} ({mapped_count/len(unique_skills_list)*100:.1f}%)")
    print(f"  - Unmapped: {unmapped_count} ({unmapped_count/len(unique_skills_list)*100:.1f}%)")
    
    # Save skill mapping
    mapping_file = OUTPUT_DIR / "skill_to_esco_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(skill_mapping, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved skill mapping to {mapping_file}")
    
    # Update JD-CV pairs with ESCO IDs
    updated_pairs = update_jdcv_pairs_with_esco_ids(pairs, skill_mapping)
    
    # Save updated pairs
    updated_file = OUTPUT_DIR / "jd_cv_pairs_with_esco.json"
    with open(updated_file, 'w', encoding='utf-8') as f:
        json.dump(updated_pairs, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved updated pairs to {updated_file}")
    
    # Also update the original file (backup first)
    backup_file = OUTPUT_DIR / "jd_cv_pairs_backup.json"
    if not backup_file.exists():
        import shutil
        shutil.copy(JD_CV_PAIRS_PATH, backup_file)
        print(f"💾 Backup created at {backup_file}")
    
    with open(JD_CV_PAIRS_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_pairs, f, ensure_ascii=False, indent=2)
    print(f"💾 Updated original file {JD_CV_PAIRS_PATH}")
    
    # Show some examples
    print(f"\n📋 Sample Mappings (first 10):")
    for idx, (skill_name, matches) in enumerate(list(skill_mapping.items())[:10]):
        print(f"\n  {idx+1}. '{skill_name}'")
        print(f"     → {matches[0]['label']} (similarity: {matches[0]['similarity']:.3f})")
        if len(matches) > 1:
            print(f"     → {matches[1]['label']} (similarity: {matches[1]['similarity']:.3f})")
    
    print(f"\n✅ COMPLETED!")
    print(f"\n📌 Next Steps:")
    print(f"   1. Review skill mapping in: {mapping_file}")
    print(f"   2. Run: python data/data_generation/generate_course_recommendations.py")
    print(f"\n" + "="*60)


if __name__ == "__main__":
    main()
