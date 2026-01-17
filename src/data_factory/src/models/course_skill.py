from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(slots=True)
class CourseSkill:
    course_id: str
    course_title: str
    skill_name: str
    skill_type: str  # "outcome" or "entry"
    description: Optional[str]
    category: Optional[str]
    proficiency_level: Optional[int]
    bloom_taxonomy_level: Optional[str]
    source_file: Path

    def to_embedding_payload(self) -> str:
        fragments = [self.skill_name]
        if self.description:
            fragments.append(self.description)
        if self.category:
            fragments.append(f"Category: {self.category}")
        if self.bloom_taxonomy_level:
            fragments.append(f"Bloom level: {self.bloom_taxonomy_level}")
        if self.proficiency_level is not None:
            fragments.append(f"Proficiency: {self.proficiency_level}")
        fragments.append(f"Course: {self.course_title}")
        return ". ".join(fragment for fragment in fragments if fragment)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "course_id": self.course_id,
            "course_title": self.course_title,
            "skill_name": self.skill_name,
            "skill_type": self.skill_type,
            "description": self.description,
            "category": self.category,
            "proficiency_level": self.proficiency_level,
            "bloom_taxonomy_level": self.bloom_taxonomy_level,
            "source_file": str(self.source_file),
        }


@dataclass(slots=True)
class MappedCourseSkill:
    original: CourseSkill
    esco_skill_id: str | None
    esco_preferred_label: str | None
    esco_description: str | None
    similarity_score: float | None

    def as_record(self) -> Dict[str, Any]:
        record = self.original.as_dict()
        record.update(
            {
                "esco_skill_id": self.esco_skill_id,
                "esco_preferred_label": self.esco_preferred_label,
                "esco_description": self.esco_description,
                "similarity_score": self.similarity_score,
            }
        )
        return record
