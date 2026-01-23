"""Course recommendation endpoints"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List
import logging

from src.models.request import SkillInput
from src.models.response import (
    RecommendationResponse,
    CoursesForSkillResponse,
    CourseResponse,
    SkillResponse,
    CourseSkillDetail,
)
from src.services.course_recommendation import CourseRecommendationService
from src.services.skill_search import SkillSearchService
from src.dependencies import get_recommendation_service, get_skill_search_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/courses", response_model=RecommendationResponse)
def recommend_courses(
    input_data: SkillInput,
    max_courses: int = Query(10, ge=1, le=50, description="Maximum number of courses"),
    recommendation_service: CourseRecommendationService = Depends(get_recommendation_service),
    skill_service: SkillSearchService = Depends(get_skill_search_service),
):
    """
    Recommend courses based on target skills.
    Returns ordered learning path considering prerequisites.
    
    - **skill_ids**: List of ESCO skill IDs (optional)
    - **skill_names**: List of skill names to search (optional)
    - **max_courses**: Maximum number of courses to recommend
    """
    try:
        # Resolve skill names to IDs if needed
        skill_ids = input_data.skill_ids or []
        
        if input_data.skill_names:
            for name in input_data.skill_names:
                matches = skill_service.search_by_name(name, limit=1)
                if matches:
                    skill_ids.append(matches[0]["skill_id"])
        
        if not skill_ids:
            raise HTTPException(
                status_code=400,
                detail="No valid skills provided. Specify skill_ids or skill_names.",
            )
        
        # Get recommendations
        result = recommendation_service.recommend_courses(
            skill_ids=skill_ids, max_courses=max_courses
        )
        
        # Convert to response models
        requested_skills = [
            SkillResponse(
                skill_id=s["skill_id"],
                label=s["label"],
                description=s.get("description"),
            )
            for s in result["requested_skills"]
        ]
        
        recommended_courses = [
            CourseResponse(
                course_id=c["course_id"],
                course_title=c["course_title"],
                category=c.get("category"),
                taught_skills=[
                    CourseSkillDetail(
                        skill_id=s["skill_id"],
                        label=s["label"],
                        similarity_score=s.get("similarity"),
                    )
                    for s in c.get("taught_skills", [])
                ],
                required_skills=[
                    CourseSkillDetail(
                        skill_id=s["skill_id"],
                        label=s["label"],
                    )
                    for s in c.get("required_skills", [])
                ],
                similarity_score=c.get("similarity_score"),
                prerequisite_level=c.get("prerequisite_level", 0),
            )
            for c in result["recommended_courses"]
        ]
        
        return RecommendationResponse(
            requested_skills=requested_skills,
            recommended_courses=recommended_courses,
            learning_path=result["learning_path"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@router.get("/courses/by-skill/{skill_id}", response_model=CoursesForSkillResponse)
def get_courses_by_skill(
    skill_id: str,
    limit: int = Query(10, ge=1, le=100, description="Maximum number of courses"),
    service: CourseRecommendationService = Depends(get_recommendation_service),
):
    """
    Get all courses that teach a specific skill.
    
    - **skill_id**: ESCO skill ID
    - **limit**: Maximum number of courses to return
    """
    try:
        courses_data = service.get_courses_teaching_skill(skill_id=skill_id, limit=limit)
        
        courses = [
            CourseResponse(
                course_id=c["course_id"],
                course_title=c["course_title"],
                category=c.get("category"),
                taught_skills=[],
                required_skills=[],
                similarity_score=c.get("similarity_score"),
            )
            for c in courses_data
        ]
        
        return CoursesForSkillResponse(skill_id=skill_id, courses=courses)
        
    except Exception as e:
        logger.error(f"Get courses by skill error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get courses: {str(e)}")


@router.get("/courses/{course_id}", response_model=CourseResponse)
def get_course_details(
    course_id: str,
    service: CourseRecommendationService = Depends(get_recommendation_service),
):
    """
    Get detailed information about a course including skills taught and required.
    
    - **course_id**: Course ID
    """
    try:
        course = service.get_course_details(course_id)
        
        if not course:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
        
        return CourseResponse(
            course_id=course["course_id"],
            course_title=course["course_title"],
            category=course.get("category"),
            taught_skills=[
                CourseSkillDetail(
                    skill_id=s["skill_id"],
                    label=s["label"],
                    similarity_score=s.get("similarity"),
                )
                for s in course.get("taught_skills", [])
            ],
            required_skills=[
                CourseSkillDetail(
                    skill_id=s["skill_id"],
                    label=s["label"],
                )
                for s in course.get("required_skills", [])
            ],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get course details error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get course: {str(e)}")
