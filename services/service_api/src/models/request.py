"""Request schemas"""
from pydantic import BaseModel, Field
from typing import List, Optional


class SkillSearchRequest(BaseModel):
    """Skill search request"""
    query: str = Field(..., description="Search query for skills")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")


class SkillInput(BaseModel):
    """Input for course recommendation"""
    skill_ids: Optional[List[str]] = Field(None, description="List of ESCO skill IDs")
    skill_names: Optional[List[str]] = Field(None, description="List of skill names to search")


class CourseRecommendationRequest(BaseModel):
    """Course recommendation request"""
    skills: SkillInput = Field(..., description="Target skills for recommendation")
    max_courses: int = Field(10, ge=1, le=50, description="Maximum number of courses to recommend")


class CVAnalysisRequest(BaseModel):
    """CV analysis request for skill extraction"""
    cv_text: str = Field(..., description="CV text content")
    extract_skills: bool = Field(True, description="Whether to extract skills from CV")


class JDAnalysisRequest(BaseModel):
    """Job description analysis request"""
    jd_text: str = Field(..., description="Job description text content")
    extract_skills: bool = Field(True, description="Whether to extract required skills")
