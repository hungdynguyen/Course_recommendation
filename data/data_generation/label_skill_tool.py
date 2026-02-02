"""
Tool for manual labeling of course skills to ESCO skills
Creates a simple web interface for human labeling
Uses the same pipeline as production (embedding + reranker)
"""

import json
import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np

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
LABELED_DATA_FILE = OUTPUT_DIR / "human_labeled_skills.json"

app = Flask(__name__)

class SkillLabeler:
    def __init__(self):
        self.course_skills = []
        self.esco_skills = []
        self.esco_embeddings = None
        self.embedding_service = None
        self.reranker_service = None
        self.labeled_data = []
        self.settings = None
        
    def load_course_skills(self):
        """Load all skills from course JSON files"""
        print("Loading course skills...")
        all_skills = []
        skill_sources = {}  # Track which course each skill comes from
        
        for dept_folder in COURSES_DIR.iterdir():
            if not dept_folder.is_dir():
                continue
            
            for course_file in dept_folder.glob("*.json"):
                try:
                    with open(course_file, 'r', encoding='utf-8') as f:
                        course_data = json.load(f)
                    
                    # Extract skills from course
                    skills = course_data.get('skills', [])
                    course_name = course_data.get('course_name', course_file.stem)
                    
                    for skill in skills:
                        skill_text = skill if isinstance(skill, str) else skill.get('skill_name', '')
                        if skill_text and skill_text not in all_skills:
                            all_skills.append(skill_text)
                            skill_sources[skill_text] = course_name
                            
                except Exception as e:
                    print(f"Error loading {course_file}: {e}")
        
        # Create skill objects with metadata
        self.course_skills = [
            {
                'id': idx,
                'skill_text': skill,
                'source_course': skill_sources.get(skill, 'Unknown')
            }
            for idx, skill in enumerate(all_skills)
        ]
        
        print(f"Loaded {len(self.course_skills)} unique course skills")
        return self.course_skills
    
    def load_esco_skills(self):
        """Load ESCO skills taxonomy"""
        print("Loading ESCO skills...")
        df = pd.read_csv(ESCO_SKILLS_PATH, encoding='utf-8')
        df = df.fillna('')
        
        self.esco_skills = []
        for _, row in df.iterrows():
            skill_name = row.get('preferredLabel', '')
            description = row.get('description', '')
            definition = row.get('definition', '')
            scope_note = row.get('scopeNote', '')
            
            # Combine text for embedding
            desc_parts = [p for p in [description, definition, scope_note] if p]
            full_description = ' '.join(desc_parts).strip()
            if not full_description:
                full_description = skill_name
            
            self.esco_skills.append({
                'uri': row.get('conceptUri', ''),
                'name': skill_name,
                'description': description,
                'full_text': f"{skill_name}. {full_description}",
                'skill_type': row.get('skillType', '')
            })
        
        print(f"Loaded {len(self.esco_skills)} ESCO skills")
        return self.esco_skills
    
    def init_services(self):
        """Initialize embedding and reranker services"""
        print("Initializing services...")
        self.settings = load_config()
        
        # Initialize embedding service
        print("Loading embedding model...")
        self.embedding_service = EmbeddingService(self.settings.embedding)
        
        # Initialize reranker service
        if self.settings.reranker.enabled:
            print("Loading reranker model...")
            self.reranker_service = RerankerService(self.settings.reranker)
        
        print("Services initialized!")
    
    def build_esco_embeddings(self):
        """Build embeddings for all ESCO skills"""
        print("Building ESCO skill embeddings...")
        esco_texts = [s['full_text'] for s in self.esco_skills]
        
        # Encode in batches
        self.esco_embeddings = self.embedding_service.encode(
            esco_texts,
            batch_size=self.settings.embedding.batch_size,
            show_progress=True
        )
        
        print(f"Built embeddings: {self.esco_embeddings.shape}")
    
    def find_top_matches(self, query_skill: str, top_k: int = 50):
        """Find top K matching ESCO skills using embedding + reranker"""
        if self.embedding_service is None or self.esco_embeddings is None:
            return []
        
        # Step 1: Encode query skill
        query_embedding = self.embedding_service.encode([query_skill])[0]
        
        # Step 2: Compute cosine similarities
        similarities = np.dot(self.esco_embeddings, query_embedding) / (
            np.linalg.norm(self.esco_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Step 3: Get top candidates for reranking (more than final top_k)
        num_candidates = min(200, len(self.esco_skills))
        top_indices = np.argsort(similarities)[::-1][:num_candidates]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                'esco_skill': self.esco_skills[idx],
                'embedding_score': float(similarities[idx]),
                'index': int(idx)
            })
        
        # Step 4: Rerank if reranker is available
        if self.reranker_service:
            print(f"Reranking {len(candidates)} candidates...")
            
            # Prepare pairs for reranking
            pairs = [[query_skill, c['esco_skill']['full_text']] for c in candidates]
            
            # Get rerank scores
            rerank_scores = self.reranker_service.compute_scores(pairs)
            
            # Update scores
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(rerank_scores[i])
                candidate['final_score'] = float(rerank_scores[i])  # Use rerank as final
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        else:
            # Use embedding score only
            for candidate in candidates:
                candidate['final_score'] = candidate['embedding_score']
        
        # Return top K
        return candidates[:top_k]
    
    def load_labeled_data(self):
        """Load existing labeled data"""
        if LABELED_DATA_FILE.exists():
            with open(LABELED_DATA_FILE, 'r', encoding='utf-8') as f:
                self.labeled_data = json.load(f)
            print(f"Loaded {len(self.labeled_data)} existing labels")
        else:
            self.labeled_data = []
    
    def save_label(self, course_skill_id: int, course_skill_text: str, 
                   esco_uri: str, esco_name: str, esco_description: str):
        """Save a labeled pair"""
        # Check if already labeled
        for item in self.labeled_data:
            if item['course_skill_id'] == course_skill_id:
                # Update existing
                item['esco_uri'] = esco_uri
                item['esco_name'] = esco_name
                item['esco_description'] = esco_description
                self._save_to_file()
                return
        
        # Add new label
        self.labeled_data.append({
            'course_skill_id': course_skill_id,
            'course_skill_text': course_skill_text,
            'esco_uri': esco_uri,
            'esco_name': esco_name,
            'esco_description': esco_description
        })
        self._save_to_file()
    
    def _save_to_file(self):
        """Save labeled data to file"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(LABELED_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.labeled_data, f, ensure_ascii=False, indent=2)
    
    def get_labeled_skill_ids(self):
        """Get IDs of already labeled skills"""
        return set(item['course_skill_id'] for item in self.labeled_data)

# Initialize labeler
labeler = SkillLabeler()

# Routes
@app.route('/')
def index():
    """Main labeling interface"""
    return render_template('label_interface.html')

@app.route('/api/init', methods=['GET'])
def init_data():
    """Initialize and load all data"""
    try:
        labeler.load_esco_skills()
        labeler.init_services()
        labeler.build_esco_embeddings()
        labeler.load_course_skills()
        labeler.load_labeled_data()
        
        labeled_ids = labeler.get_labeled_skill_ids()
        
        return jsonify({
            'success': True,
            'total_skills': len(labeler.course_skills),
            'labeled_count': len(labeled_ids),
            'remaining_count': len(labeler.course_skills) - len(labeled_ids)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/skills', methods=['GET'])
def get_skills():
    """Get all course skills"""
    labeled_ids = labeler.get_labeled_skill_ids()
    
    # Mark which skills are already labeled
    skills_with_status = []
    for skill in labeler.course_skills:
        skills_with_status.append({
            **skill,
            'is_labeled': skill['id'] in labeled_ids
        })
    
    return jsonify({
        'success': True,
        'skills': skills_with_status
    })

@app.route('/api/search', methods=['POST'])
def search_matches():
    """Search for matching ESCO skills using production pipeline"""
    data = request.json
    skill_text = data.get('skill_text', '')
    top_k = data.get('top_k', 50)
    
    try:
        matches = labeler.find_top_matches(skill_text, top_k)
        return jsonify({
            'success': True,
            'matches': matches
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save_label', methods=['POST'])
def save_label():
    """Save a labeled skill pair"""
    data = request.json
    
    try:
        labeler.save_label(
            course_skill_id=data['course_skill_id'],
            course_skill_text=data['course_skill_text'],
            esco_uri=data['esco_uri'],
            esco_name=data['esco_name'],
            esco_description=data['esco_description']
        )
        
        return jsonify({
            'success': True,
            'message': 'Label saved successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get labeling statistics"""
    labeled_ids = labeler.get_labeled_skill_ids()
    total = len(labeler.course_skills)
    labeled = len(labeled_ids)
    
    return jsonify({
        'success': True,
        'total': total,
        'labeled': labeled,
        'remaining': total - labeled,
        'progress_percent': round((labeled / total * 100), 2) if total > 0 else 0
    })

def main():
    print("="*60)
    print("Skill Labeling Tool")
    print("="*60)
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")
    print("\nPress Ctrl+C to stop")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
