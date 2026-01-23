"""Domain models"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Skill:
    """Skill domain model"""
    skill_id: str
    preferred_label: str
    description: Optional[str] = None
    skill_type: Optional[str] = None
    alternative_labels: Optional[str] = None


@dataclass
class Course:
    """Course domain model"""
    course_id: str
    course_title: str
    category: Optional[str] = None
    description: Optional[str] = None


@dataclass
class CourseSkillMapping:
    """Course-Skill relationship"""
    course_id: str
    skill_id: str
    relationship_type: str  # 'teaches' or 'requires'
    similarity_score: Optional[float] = None
