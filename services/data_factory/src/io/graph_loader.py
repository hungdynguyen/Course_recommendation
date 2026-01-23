from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List

from ..models.graph import BroaderEdge, CourseNode, RequiresEdge, SkillNode, TeachesEdge
from ..models.skill import Skill

LOGGER = logging.getLogger("data_factory.graph_loader")


def load_skill_nodes(skills: Iterable[Skill]) -> List[SkillNode]:
    """Convert Skill objects to SkillNode for graph."""
    nodes: List[SkillNode] = []
    for skill in skills:
        nodes.append(
            SkillNode(
                skill_id=skill.skill_id,
                preferred_label=skill.preferred_label,
                description=skill.description,
                skill_type=skill.skill_type,
                alternative_labels=skill.alternative_labels or [],
            )
        )
    return nodes


def load_broader_edges(skills: Iterable[Skill]) -> List[BroaderEdge]:
    """Extract BROADER edges from skill broader_skill_ids."""
    edges: List[BroaderEdge] = []
    for skill in skills:
        for parent_id in skill.broader_skill_ids:
            edges.append(BroaderEdge(child_id=skill.skill_id, parent_id=parent_id))
    return edges


def load_course_mappings(mappings_path: Path) -> tuple[List[CourseNode], List[TeachesEdge], List[RequiresEdge]]:
    """
    Load course_skill_mappings.jsonl and extract:
    - CourseNode: unique courses
    - TeachesEdge: when skill_type=outcome and esco_skill_id exists
    - RequiresEdge: when skill_type=entry and esco_skill_id exists
    """
    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")

    courses_dict = {}
    teaches_edges: List[TeachesEdge] = []
    requires_edges: List[RequiresEdge] = []

    with mappings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)

            course_id = record.get("course_id")
            course_title = record.get("course_title", "Unknown")
            category = record.get("category")
            source_file = record.get("source_file", "")
            skill_type = record.get("skill_type")
            esco_skill_id = record.get("esco_skill_id")
            similarity_score = record.get("similarity_score")

            if course_id and course_id not in courses_dict:
                courses_dict[course_id] = CourseNode(
                    course_id=course_id,
                    course_title=course_title,
                    category=category,
                    source_file=source_file,
                )

            if not esco_skill_id or not course_id:
                continue

            if skill_type == "outcome":
                teaches_edges.append(
                    TeachesEdge(
                        course_id=course_id,
                        skill_id=esco_skill_id,
                        similarity_score=similarity_score,
                        skill_type=skill_type,
                        source="embedding+rerank",
                    )
                )
            elif skill_type == "entry":
                requires_edges.append(
                    RequiresEdge(
                        course_id=course_id,
                        skill_id=esco_skill_id,
                        skill_type=skill_type,
                        source="embedding+rerank",
                    )
                )

    courses = list(courses_dict.values())
    LOGGER.info("Loaded %s courses, %s TEACHES, %s REQUIRES edges", len(courses), len(teaches_edges), len(requires_edges))
    return courses, teaches_edges, requires_edges
