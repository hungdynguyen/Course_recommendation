"""CourseRecommendationService – query Neo4j để tìm khoá học dạy các skill gaps.

Luồng:
  canonical skill_ids → MATCH (:Course)-[:TEACHES]->(:Skill) → ranked courses
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

from shared.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class CourseRecommendationService:

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def recommend_for_gaps(
        self,
        gap_skill_ids: List[str],
        max_courses: int = 10,
    ) -> Dict:
        """
        Gợi ý khoá học dạy các skill trong *gap_skill_ids*.

        Parameters
        ----------
        gap_skill_ids : canonical skill_ids (từ SkillSearchService)
        max_courses   : số khoá học tối đa trả về

        Returns
        -------
        {
            "gap_skills": [...],          # skill details
            "recommended_courses": [...], # course details + skills covered
        }
        """
        if not gap_skill_ids:
            return {"gap_skills": [], "recommended_courses": []}

        skill_details = self._get_skill_details(gap_skill_ids)
        courses = self._get_courses_for_skills(gap_skill_ids, limit=max_courses * 3)

        if not courses:
            return {"gap_skills": skill_details, "recommended_courses": []}

        # Gộp, enriched, sắp xếp theo số skills covered
        enriched = self._enrich_and_rank(courses, gap_skill_ids, max_courses)

        return {
            "gap_skills":          skill_details,
            "recommended_courses": enriched,
        }

    def get_course_details(self, course_id: str) -> Optional[Dict]:
        rows = self._neo4j.query(
            """
            MATCH (c:Course {course_id: $cid})
            OPTIONAL MATCH (c)-[:TEACHES]->(ts:Skill)
            OPTIONAL MATCH (c)-[:REQUIRES]->(rs:Skill)
            RETURN c.course_id    AS course_id,
                   c.course_title AS course_title,
                   c.category     AS category,
                   collect(DISTINCT {skill_id: ts.skill_id, label: ts.canonical_label}) AS taught,
                   collect(DISTINCT {skill_id: rs.skill_id, label: rs.canonical_label}) AS required
            """,
            {"cid": course_id},
        )
        if not rows:
            return None
        r = rows[0]
        r["taught"]    = [s for s in r["taught"]    if s.get("skill_id")]
        r["required"]  = [s for s in r["required"]  if s.get("skill_id")]
        return r

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_skill_details(self, skill_ids: List[str]) -> List[Dict]:
        rows = self._neo4j.query(
            """
            MATCH (s:Skill) WHERE s.skill_id IN $ids
            RETURN s.skill_id AS skill_id, s.canonical_label AS label,
                   s.description AS description, s.category AS category
            """,
            {"ids": skill_ids},
        )
        return rows

    def _get_courses_for_skills(self, skill_ids: List[str], limit: int = 50) -> List[Dict]:
        rows = self._neo4j.query(
            """
            MATCH (c:Course)-[:TEACHES]->(s:Skill)
            WHERE s.skill_id IN $ids
            RETURN c.course_id    AS course_id,
                   c.course_title AS course_title,
                   c.category     AS category,
                   collect(DISTINCT s.skill_id) AS covered_skill_ids
            ORDER BY size(collect(DISTINCT s.skill_id)) DESC
            LIMIT $limit
            """,
            {"ids": skill_ids, "limit": limit},
        )
        return rows

    def _enrich_and_rank(
        self,
        courses: List[Dict],
        gap_skill_ids: List[str],
        max_courses: int,
    ) -> List[Dict]:
        gap_set = set(gap_skill_ids)
        result = []
        for c in courses:
            covered = [sid for sid in c.get("covered_skill_ids", []) if sid in gap_set]
            result.append({
                "course_id":       c["course_id"],
                "course_title":    c["course_title"],
                "category":        c.get("category", ""),
                "covered_gaps":    covered,
                "coverage_count":  len(covered),
            })
        # Sắp xếp: nhiều gap coverage nhất lên trước
        result.sort(key=lambda x: x["coverage_count"], reverse=True)
        return result[:max_courses]
