"""Load course skills từ các file JSON trích xuất bằng LLM."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List

from shared.models.course import CourseSkill

logger = logging.getLogger("data_factory.io.course_skill_loader")


def load_course_skills(root_dir: Path) -> List[CourseSkill]:
    """Đọc toàn bộ file JSON trong *root_dir* và trả về list CourseSkill."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Course directory not found: {root_dir}")

    skills: List[CourseSkill] = []
    for json_path in sorted(_iter_json_files(root_dir)):
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse %s: %s", json_path, exc)
            continue

        course_id = payload.get("course_id") or payload.get("courseCode") or json_path.stem
        course_title = payload.get("title") or payload.get("courseTitle") or "Unknown"

        skills.extend(_extract_outcomes(course_id, course_title, json_path, payload))
        skills.extend(_extract_entry_skills(course_id, course_title, json_path, payload))

    logger.info("Loaded %d course skill records from %s", len(skills), root_dir)
    return skills


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _iter_json_files(root_dir: Path) -> Iterable[Path]:
    for p in root_dir.rglob("*.json"):
        if p.is_file():
            yield p


def _extract_outcomes(course_id: str, course_title: str, src: Path, payload: dict) -> List[CourseSkill]:
    results = []
    for item in payload.get("skill_outcomes", []):
        skill_name = item.get("skill_name", "").strip()
        if not skill_name:
            continue
        results.append(CourseSkill(
            course_id=course_id,
            course_title=course_title,
            skill_name=skill_name,
            skill_type="outcome",
            description=item.get("outcome_description"),
            category=item.get("category"),
            proficiency_level=item.get("target_proficiency_level"),
            bloom_taxonomy_level=item.get("bloom_taxonomy_level"),
            source_file=src,
        ))
    return results


def _extract_entry_skills(course_id: str, course_title: str, src: Path, payload: dict) -> List[CourseSkill]:
    results = []
    entry = payload.get("entry_requirements", {})
    for item in entry.get("minimum_entry_skills", []):
        skill_name = item.get("skill_name", "").strip()
        if not skill_name:
            continue
        results.append(CourseSkill(
            course_id=course_id,
            course_title=course_title,
            skill_name=skill_name,
            skill_type="entry",
            description=item.get("description"),
            category=item.get("category"),
            proficiency_level=item.get("proficiency_level"),
            bloom_taxonomy_level=None,
            source_file=src,
        ))
    return results
