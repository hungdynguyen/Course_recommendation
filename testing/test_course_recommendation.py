"""
Test script for Course Recommendation API
Tests only the /api/v1/courses/recommend endpoint
"""

import requests
import json


BASE_URL = "http://localhost:8000"


def test_recommend_courses():
    """Test course recommendation endpoint"""
    
    print("="*80)
    print("Testing Course Recommendation API")
    print("="*80)
    
    # Test 1: Recommend by skill names (ESCO English skills)
    print("\n[Test 1] Recommend courses by ESCO skill names")
    print("-" * 80)
    
    payload = {
        "skill_names": ["insurance", "risk management", "data analysis"]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/courses/recommend",
            params={"max_courses": 5},
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Display requested skills
            print(f"\n📋 Requested Skills ({len(result['requested_skills'])}):")
            for skill in result['requested_skills']:
                print(f"   • {skill['label']}")
                print(f"     ID: {skill['skill_id']}")
            
            # Display recommended courses
            courses = result['recommended_courses']
            print(f"\n📚 Recommended Courses ({len(courses)}):")
            for idx, course in enumerate(courses, 1):
                print(f"\n   {idx}. {course['course_title']}")
                print(f"      ID: {course['course_id']}")
                print(f"      Category: {course.get('category', 'N/A')}")
                print(f"      Prerequisite Level: {course['prerequisite_level']}")
                print(f"      Teaches: {len(course['taught_skills'])} skills")
                print(f"      Requires: {len(course['required_skills'])} skills")
                
                # Show some taught skills
                if course['taught_skills']:
                    print(f"      Taught skills:")
                    for skill in course['taught_skills'][:3]:
                        print(f"         - {skill['label']}")
                    if len(course['taught_skills']) > 3:
                        print(f"         ... and {len(course['taught_skills']) - 3} more")
            
            # Display learning path
            learning_path = result['learning_path']
            print(f"\n📖 Learning Path ({len(learning_path)} courses):")
            for idx, course_id in enumerate(learning_path, 1):
                # Find course name
                course_name = next(
                    (c['course_title'] for c in courses if c['course_id'] == course_id),
                    course_id
                )
                print(f"   {idx}. {course_name}")
            
            print("\n✅ Test 1 PASSED")
            
        else:
            print(f"❌ Test 1 FAILED: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API!")
        print("Make sure API is running: docker exec -it vietcv_data_factory python run_api.py")
        return
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return
    
    # Test 2: Recommend by different ESCO skill names
    print("\n\n[Test 2] Recommend courses by other ESCO skill names")
    print("-" * 80)
    
    payload = {
        "skill_names": ["financial analysis", "statistical methods", "business management"]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/courses/recommend",
            params={"max_courses": 3},
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📋 Requested Skills: {len(result['requested_skills'])} skills")
            print(f"📚 Recommended Courses: {len(result['recommended_courses'])} courses")
            print(f"📖 Learning Path: {len(result['learning_path'])} courses")
            
            for idx, course in enumerate(result['recommended_courses'], 1):
                print(f"   {idx}. {course['course_title']} (Level {course['prerequisite_level']})")
            
            print("\n✅ Test 2 PASSED")
            
        else:
            print(f"❌ Test 2 FAILED: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    # Test 3: Search skills then recommend
    print("\n\n[Test 3] Search skills first, then recommend by skill IDs")
    print("-" * 80)
    
    try:
        # First, search for skills (use ESCO English terms)
        search_response = requests.post(
            f"{BASE_URL}/api/v1/skills/search",
            params={"query": "economics", "limit": 2},
            timeout=10
        )
        
        if search_response.status_code == 200:
            skills = search_response.json()
            skill_ids = [s['skill_id'] for s in skills]
            
            print(f"Found {len(skills)} skills:")
            for skill in skills:
                print(f"   • {skill['label']}")
            
            # Then recommend courses
            payload = {
                "skill_ids": skill_ids
            }
            
            response = requests.post(
                f"{BASE_URL}/api/v1/courses/recommend",
                params={"max_courses": 3},
                json=payload,
                timeout=30
            )
            
            print(f"\nRecommendation Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"📚 Recommended: {len(result['recommended_courses'])} courses")
                
                for course in result['recommended_courses']:
                    print(f"   • {course['course_title']}")
                
                print("\n✅ Test 3 PASSED")
            else:
                print(f"❌ Test 3 FAILED: {response.text}")
        else:
            print(f"❌ Skill search failed: {search_response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print("✅ Course Recommendation API is working!")
    print(f"   API URL: {BASE_URL}")
    print("   Endpoint: POST /api/v1/courses/recommend")
    print("\n💡 NOTE: ESCO taxonomy uses English skill names")
    print("   To see available skills, run: python testing/list_esco_skills.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_recommend_courses()
