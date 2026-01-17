from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List

from ..models.course_skill import CourseSkill

LOGGER = logging.getLogger("data_factory.course_skill_loader")


def load_course_skills(root_dir: Path) -> List[CourseSkill]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Course directory does not exist: {root_dir}")

    skills: List[CourseSkill] = []
    for json_path in sorted(_iter_course_files(root_dir)):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse %s: %s", json_path, exc)
            continue

        course_id = payload.get("course_id") or payload.get("courseCode")
        course_title = payload.get("title") or payload.get("courseTitle") or "Unknown title"

        skills.extend(_extract_outcomes(course_id, course_title, json_path, payload))
        skills.extend(_extract_entry_skills(course_id, course_title, json_path, payload))

    LOGGER.info("Loaded %s course skill records", len(skills))
    return skills


def _iter_course_files(root_dir: Path) -> Iterable[Path]:
    for path in root_dir.rglob("*.json"):
        if path.is_file():
            yield path


def _extract_outcomes(
    course_id: str | None,
    course_title: str,
    source_file: Path,
    payload: dict,
) -> List[CourseSkill]:
    outcomes = payload.get("skill_outcomes", [])
    results: List[CourseSkill] = []
    for item in outcomes:
        results.append(
            CourseSkill(
                course_id=course_id or source_file.stem,
                course_title=course_title,
                skill_name=item.get("skill_name", ""),
                skill_type="outcome",
                description=item.get("outcome_description"),
                category=item.get("category"),
                proficiency_level=item.get("target_proficiency_level"),
                bloom_taxonomy_level=item.get("bloom_taxonomy_level"),
                source_file=source_file,
            )
        )
    return results


def _extract_entry_skills(
    course_id: str | None,
    course_title: str,
    source_file: Path,
    payload: dict,
) -> List[CourseSkill]:
    entry_section = payload.get("entry_requirements", {})
    skills = entry_section.get("minimum_entry_skills", [])
    results: List[CourseSkill] = []
    for item in skills:
        results.append(
            CourseSkill(
                course_id=course_id or source_file.stem,
                course_title=course_title,
                skill_name=item.get("skill_name", ""),
                skill_type="entry",
                description=None,
                category=None,
                proficiency_level=item.get("minimum_proficiency_level"),
                bloom_taxonomy_level=None,
                source_file=source_file,
            )
        )
    return results
