from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from src.service_api.services.course_recommendation import CourseRecommendationService
from src.service_api.services.skill_search import SkillSearchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VietCV Course Recommendation API",
    description="GraphRAG-based course recommendation system for skill gap analysis",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
recommendation_service = CourseRecommendationService()
skill_search_service = SkillSearchService()


class SkillInput(BaseModel):
    skill_ids: Optional[List[str]] = Field(None, description="List of ESCO skill IDs")
    skill_names: Optional[List[str]] = Field(None, description="List of skill names to search")


class CourseRecommendation(BaseModel):
    course_id: str
    course_title: str
    category: Optional[str]
    taught_skills: List[dict]
    required_skills: List[dict]
    similarity_score: Optional[float]
    prerequisite_level: int = Field(description="0=foundational, higher=advanced")


class RecommendationResponse(BaseModel):
    requested_skills: List[dict]
    recommended_courses: List[CourseRecommendation]
    learning_path: List[str] = Field(description="Ordered course IDs forming a learning path")


@app.get("/")
def root():
    return {
        "message": "VietCV Course Recommendation API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/api/v1/skills/search",
            "/api/v1/courses/recommend",
            "/api/v1/courses/by-skill",
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "course-recommendation"}


@app.post("/api/v1/skills/search", response_model=List[dict])
def search_skills(query: str, limit: int = 10):
    """
    Search ESCO skills by name (fuzzy matching via Elasticsearch).
    Returns list of matching skills with IDs.
    """
    try:
        results = skill_search_service.search_by_name(query, limit=limit)
        return results
    except Exception as e:
        logger.error(f"Skill search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/courses/recommend", response_model=RecommendationResponse)
def recommend_courses(input_data: SkillInput, max_courses: int = 10):
    """
    Recommend courses based on target skills.
    Returns ordered learning path considering prerequisites.
    """
    try:
        # Resolve skill names to IDs if needed
        skill_ids = input_data.skill_ids or []
        if input_data.skill_names:
            for name in input_data.skill_names:
                matches = skill_search_service.search_by_name(name, limit=1)
                if matches:
                    skill_ids.append(matches[0]["skill_id"])

        if not skill_ids:
            raise HTTPException(status_code=400, detail="No valid skills provided")

        # Get recommendations
        result = recommendation_service.recommend_courses(
            skill_ids=skill_ids, max_courses=max_courses
        )

        return result
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/courses/by-skill/{skill_id}")
def get_courses_by_skill(skill_id: str, limit: int = 10):
    """
    Get all courses that teach a specific skill.
    """
    try:
        courses = recommendation_service.get_courses_teaching_skill(
            skill_id=skill_id, limit=limit
        )
        return {"skill_id": skill_id, "courses": courses}
    except Exception as e:
        logger.error(f"Get courses by skill error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/courses/{course_id}")
def get_course_details(course_id: str):
    """
    Get detailed information about a course including skills taught and required.
    """
    try:
        course = recommendation_service.get_course_details(course_id)
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        return course
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get course details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
def startup_event():
    logger.info("Starting VietCV Course Recommendation API")


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down VietCV Course Recommendation API")
    recommendation_service.close()
    skill_search_service.close()