from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class SkillNode:
    skill_id: str
    preferred_label: str
    description: Optional[str]
    skill_type: Optional[str]
    alternative_labels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "preferred_label": self.preferred_label,
            "description": self.description,
            "skill_type": self.skill_type,
            "alternative_labels": self.alternative_labels,
        }


@dataclass(slots=True)
class CourseNode:
    course_id: str
    course_title: str
    category: Optional[str]
    source_file: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "course_id": self.course_id,
            "course_title": self.course_title,
            "category": self.category,
            "source_file": self.source_file,
        }


@dataclass(slots=True)
class BroaderEdge:
    child_id: str
    parent_id: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
        }


@dataclass(slots=True)
class TeachesEdge:
    course_id: str
    skill_id: str
    similarity_score: Optional[float]
    skill_type: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "course_id": self.course_id,
            "skill_id": self.skill_id,
            "similarity_score": self.similarity_score,
            "skill_type": self.skill_type,
            "source": self.source,
        }


@dataclass(slots=True)
class RequiresEdge:
    course_id: str
    skill_id: str
    skill_type: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "course_id": self.course_id,
            "skill_id": self.skill_id,
            "skill_type": self.skill_type,
            "source": self.source,
        }
