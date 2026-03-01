"""
Course Recommendation Labeling Tool
Manual labeling for validating and improving course recommendations
"""

import json
import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RECOMMENDATIONS_FILE = DATA_DIR / "processed" / "course_recommendations" / "course_recommendations.json"
COURSES_DIR = DATA_DIR / "Data_Courses_Json"
OUTPUT_DIR = DATA_DIR / "processed" / "training_dataset"
LABELED_FILE = OUTPUT_DIR / "human_labeled_recommendations.json"

app = Flask(__name__, template_folder=str(BASE_DIR / "data" / "data_generation" / "templates"))

class RecommendationLabeler:
    def __init__(self):
        self.recommendations = []
        self.courses_detail = {}
        self.labeled_data = []
        
    def load_recommendations(self):
        """Load recommendations from JSON"""
        print("Loading recommendations...")
        if not RECOMMENDATIONS_FILE.exists():
            raise FileNotFoundError(f"Recommendations file not found: {RECOMMENDATIONS_FILE}")
        
        with open(RECOMMENDATIONS_FILE, 'r', encoding='utf-8') as f:
            self.recommendations = json.load(f)
        
        print(f"Loaded {len(self.recommendations)} JD-CV pairs with recommendations")
        return self.recommendations
    
    def load_course_details(self):
        """Load detailed course information"""
        print("Loading course details...")
        course_map = {}
        
        for dept_folder in COURSES_DIR.iterdir():
            if not dept_folder.is_dir():
                continue
                
            for course_file in dept_folder.glob("*.json"):
                try:
                    with open(course_file, 'r', encoding='utf-8') as f:
                        course = json.load(f)
                        course_id = course.get('course_id', '')
                        if course_id:
                            course_map[course_id] = course
                except Exception as e:
                    print(f"Error loading {course_file}: {e}")
        
        self.courses_detail = course_map
        print(f"Loaded {len(self.courses_detail)} course details")
        return course_map
    
    def load_labeled_data(self):
        """Load existing labeled data"""
        if LABELED_FILE.exists():
            with open(LABELED_FILE, 'r', encoding='utf-8') as f:
                self.labeled_data = json.load(f)
            print(f"Loaded {len(self.labeled_data)} existing labels")
        else:
            self.labeled_data = []
    
    def save_label(self, pair_id: int, jd_id: str, selected_courses: list, notes: str = ""):
        """Save labeled recommendation"""
        # Check if already labeled
        for item in self.labeled_data:
            if item['pair_id'] == pair_id:
                # Update existing
                item['selected_courses'] = selected_courses
                item['notes'] = notes
                item['labeled_at'] = datetime.now().isoformat()
                self._save_to_file()
                return
        
        # Add new label
        self.labeled_data.append({
            'pair_id': pair_id,
            'jd_id': jd_id,
            'selected_courses': selected_courses,
            'notes': notes,
            'labeled_at': datetime.now().isoformat()
        })
        self._save_to_file()
    
    def _save_to_file(self):
        """Save labeled data to file"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(LABELED_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.labeled_data, f, ensure_ascii=False, indent=2)
    
    def get_labeled_pair_ids(self):
        """Get IDs of already labeled pairs"""
        return set(item['pair_id'] for item in self.labeled_data)
    
    def get_label_for_pair(self, pair_id: int):
        """Get existing label for a pair"""
        for item in self.labeled_data:
            if item['pair_id'] == pair_id:
                return item
        return None

# Initialize labeler
labeler = RecommendationLabeler()

# Routes
@app.route('/')
def index():
    """Main labeling interface"""
    return render_template('label_course_interface.html')

@app.route('/api/init', methods=['GET'])
def init_data():
    """Initialize and load all data"""
    try:
        labeler.load_recommendations()
        labeler.load_course_details()
        labeler.load_labeled_data()
        
        labeled_ids = labeler.get_labeled_pair_ids()
        
        return jsonify({
            'success': True,
            'total_pairs': len(labeler.recommendations),
            'labeled_count': len(labeled_ids),
            'remaining_count': len(labeler.recommendations) - len(labeled_ids)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pairs', methods=['GET'])
def get_pairs():
    """Get all JD-CV pairs with recommendation status"""
    labeled_ids = labeler.get_labeled_pair_ids()
    
    pairs_summary = []
    for rec in labeler.recommendations:
        pair_id = rec['pair_id']
        is_labeled = pair_id in labeled_ids
        
        pairs_summary.append({
            'pair_id': pair_id,
            'jd_title': rec['jd_title'],
            'jd_id': rec['jd_id'],
            'cv_experience': rec['cv_experience'],
            'cv_degree': rec['cv_degree'],
            'target_matching': rec.get('target_matching_percent', 0),
            'api_count': rec['recommendations']['api_count'],
            'vector_count': rec['recommendations']['vector_count'],
            'total_count': rec['recommendations']['merged_count'],
            'is_labeled': is_labeled
        })
    
    return jsonify({
        'success': True,
        'pairs': pairs_summary
    })

@app.route('/api/pair/<int:pair_id>', methods=['GET'])
def get_pair_detail(pair_id):
    """Get detailed information for a specific pair"""
    try:
        # Find the pair
        pair_data = None
        for rec in labeler.recommendations:
            if rec['pair_id'] == pair_id:
                pair_data = rec
                break
        
        if not pair_data:
            return jsonify({'success': False, 'error': 'Pair not found'}), 404
        
        # Enrich courses with full details
        courses = pair_data['recommendations']['courses']
        api_courses = []
        vector_courses = []
        
        for course in courses:
            course_id = course['course_id']
            course_detail = labeler.courses_detail.get(course_id, {})
            
            enriched = {
                'course_id': course_id,
                'title': course.get('title', course_detail.get('title', 'Unknown')),
                'score': course.get('score', 0),
                'methods': course.get('methods', []),
                'api_score': course.get('api_score'),
                'vector_score': course.get('vector_score'),
                'category': course_detail.get('category', ''),
                'credit': course_detail.get('credit', ''),
                'description': course_detail.get('description', ''),
                'faculty': course_detail.get('faculty', ''),
                'skill_outcomes': course_detail.get('skill_outcomes', [])  # All skills
            }
            
            # Split by method
            if 'api' in course.get('methods', []):
                api_courses.append(enriched)
            if 'vector_search' in course.get('methods', []):
                vector_courses.append(enriched)
        
        # Get existing label if any
        existing_label = labeler.get_label_for_pair(pair_id)
        
        # Get JD/CV details directly from recommendation record
        jd_id = pair_data.get('jd_id', '')
        jd_info = pair_data.get('jd_info', {})
        cv_info = pair_data.get('cv_info', {})
        
        return jsonify({
            'success': True,
            'pair': {
                'pair_id': pair_id,
                'jd_title': pair_data['jd_title'],
                'jd_id': jd_id,
                'jd_description': jd_info.get('description', ''),
                'jd_technical_skills': [s.strip() for s in jd_info.get('technical_skills', '').split(';')] if jd_info.get('technical_skills') else [],
                'jd_soft_skills': [s.strip() for s in jd_info.get('soft_skills', '').split(';')] if jd_info.get('soft_skills') else [],
                'jd_experience': jd_info.get('experience', ''),
                'jd_degree': jd_info.get('degree', ''),
                'cv_experience': pair_data['cv_experience'],
                'cv_degree': pair_data['cv_degree'],
                'cv_technical_skills': cv_info.get('technical_skills', []),
                'cv_soft_skills': cv_info.get('soft_skills', []),
                'target_matching': pair_data.get('target_matching_percent', 0),
                'skill_gaps': pair_data['skill_gaps'],
                'api_courses': api_courses,
                'vector_courses': vector_courses
            },
            'existing_label': existing_label
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save_label', methods=['POST'])
def save_label():
    """Save a labeled recommendation"""
    try:
        data = request.get_json()
        pair_id = data.get('pair_id')
        jd_id = data.get('jd_id', '')
        selected_courses = data.get('selected_courses', [])
        notes = data.get('notes', '')

        if pair_id is None:
            return jsonify({'success': False, 'error': 'pair_id is required'}), 400

        labeler.save_label(pair_id, jd_id, selected_courses, notes)
        return jsonify({'success': True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get labeling statistics"""
    labeled_ids = labeler.get_labeled_pair_ids()
    total = len(labeler.recommendations)
    labeled = len(labeled_ids)
    
    return jsonify({
        'success': True,
        'total_pairs': total,
        'labeled_count': labeled,
        'remaining_count': total - labeled,
        'progress_percent': round((labeled / total * 100), 2) if total > 0 else 0
    })

def main():
    print("="*60)
    print("Course Recommendation Labeling Tool")
    print("="*60)
    print("\nStarting Flask server...")
    print("Open http://localhost:5001 in your browser")
    print("\nPress Ctrl+C to stop")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == '__main__':
    main()
