from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from ..models.skill import Skill


def load_skills(skills_path: Path, relations_path: Path) -> List[Skill]:
    skills_df = pd.read_csv(skills_path)
    relations_df = pd.read_csv(relations_path)

    relation_map: Dict[str, List[str]] = defaultdict(list)
    for row in relations_df.itertuples(index=False):
        relation_map[row.originalSkillUri].append(row.relatedSkillUri)

    skills: List[Skill] = []
    for row in skills_df.itertuples(index=False):
        concept_uri = row.conceptUri
        broader = relation_map.get(concept_uri, [])
        alternatives = _split_labels(getattr(row, "altLabels", None))

        raw_description = getattr(row, "description", None) or getattr(row, "definition", None)
        description = None if (isinstance(raw_description, float) and pd.isna(raw_description)) else raw_description

        raw_skill_type = getattr(row, "skillType", None)
        skill_type = None if (isinstance(raw_skill_type, float) and pd.isna(raw_skill_type)) else raw_skill_type

        skills.append(
            Skill(
                skill_id=concept_uri,
                preferred_label=row.preferredLabel,
                description=description,
                skill_type=skill_type,
                broader_skill_ids=broader,
                alternative_labels=alternatives,
            )
        )
    return skills


def iter_skill_documents(skills: Iterable[Skill]) -> Iterable[dict]:
    for skill in skills:
        yield skill.as_index_document()


def _split_labels(raw_value: str | float | None) -> List[str]:
    if not raw_value or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []
    return [label.strip() for label in str(raw_value).split("|") if label.strip()]
