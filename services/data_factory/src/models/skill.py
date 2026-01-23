from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(slots=True)
class Skill:
    skill_id: str
    preferred_label: str
    description: Optional[str]
    skill_type: Optional[str]
    broader_skill_ids: List[str]
    alternative_labels: List[str]

    def as_index_document(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "preferred_label": self.preferred_label,
            "description": self.description,
            "skill_type": self.skill_type,
            "broader_skill_ids": self.broader_skill_ids,
            "alternative_labels": self.alternative_labels,
        }
