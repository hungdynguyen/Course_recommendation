"""Response schemas"""
from pydantic import BaseModel, Field
from typing import List, Optional


class SkillResponse(BaseModel):
    """Skill information response"""
    skill_id: str
    label: str
    description: Optional[str] = None
    skill_type: Optional[str] = None
    score: Optional[float] = None


class CourseSkillDetail(BaseModel):
    """Skill detail in course context"""
    skill_id: str
    label: str
    similarity_score: Optional[float] = None


class CourseResponse(BaseModel):
    """Course information response"""
    course_id: str
    course_title: str
    category: Optional[str] = None
    taught_skills: List[CourseSkillDetail] = []
    required_skills: List[CourseSkillDetail] = []
    similarity_score: Optional[float] = None
    prerequisite_level: int = Field(0, description="0=foundational, higher=advanced")


class RecommendationResponse(BaseModel):
    """Course recommendation response"""
    requested_skills: List[SkillResponse]
    recommended_courses: List[CourseResponse]
    learning_path: List[str] = Field(description="Ordered course IDs forming a learning path")


class CoursesForSkillResponse(BaseModel):
    """Courses teaching a specific skill"""
    skill_id: str
    courses: List[CourseResponse]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    databases: dict = Field(default_factory=dict)
