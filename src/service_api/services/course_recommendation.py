"""CourseRecommendationService – query Neo4j để tìm khoá học dạy các skill gaps.

Luồng:
  canonical skill_ids → MATCH (:Course)-[:TEACHES]->(:Skill) → ranked courses
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional

from shared.db.neo4j_client import Neo4jClient
from service_api.config import settings

logger = logging.getLogger(__name__)

NOISY_COURSE_ANCHOR_KEYWORDS = {
    "CNTT1198": [
        "cloud security",
        "security architecture",
        "identity management",
        "access control",
        "cloud computing",
        "security",
    ],
    "CNTT1186": [
        "artificial intelligence",
        "machine learning",
        "cloud computing",
        "blockchain",
        "cybersecurity",
        "data analytics",
        "iot",
    ],
    "TIKT1129": [
        "e-commerce",
        "ecommerce",
        "payment",
        "checkout",
        "order management",
        "online business",
        "web service",
    ],
    "TIKT1135": [
        "open source",
        "software development",
        "web development",
        "php",
        "wordpress",
        "python",
        "docker",
    ],
    "CNTT1153": [
        "java programming",
        "java",
        "spring",
        "oop",
        "multithreading",
        "design patterns",
    ],
    "CNTT1188": [
        "web programming",
        "web application",
        "restful api",
        "mvc",
        "database integration",
        "web testing",
    ],
    "KTKE1103": [
        "accounting",
        "tax",
        "financial",
        "audit",
    ],
    "TMKT1141": [
        "sales",
        "commerce",
        "enterprise management",
        "customer relationship",
        "marketing",
    ],
    "CNTT1157": [
        "mobile",
        "android",
        "ios",
        "react native",
        "flutter",
        "app store",
        "google play",
    ],
    "TIHT1104": [
        "software engineering",
        "design patterns",
        "testing",
        "requirements",
        "ci/cd",
        "uml",
    ],
}


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
        skill_match_scores: Optional[Dict[str, float]] = None,
        use_weighted_ranker: Optional[bool] = None,
        debug_rank_explain: bool = False,
        full_rank_limit: Optional[int] = None,
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
            result = {"gap_skills": [], "recommended_courses": []}
            if debug_rank_explain:
                result["all_ranked_courses"] = []
            return result

        skill_details = self._get_skill_details(gap_skill_ids)
        skill_label_by_id = {
            str(item.get("skill_id")): str(item.get("label") or item.get("canonical_label") or item.get("skill_id") or "")
            for item in skill_details
            if item.get("skill_id")
        }
        candidate_limit = max_courses * 5
        if full_rank_limit is not None:
            candidate_limit = max(candidate_limit, int(full_rank_limit))
        courses = self._get_courses_for_skills(gap_skill_ids, limit=candidate_limit)

        if not courses:
            result = {"gap_skills": skill_details, "recommended_courses": []}
            if debug_rank_explain:
                result["all_ranked_courses"] = []
            return result

        # Gộp, enriched, sắp xếp theo weighted score (fallback: coverage_count).
        enabled_weighted = settings.COURSE_WEIGHTED_RANKER_ENABLED if use_weighted_ranker is None else bool(use_weighted_ranker)
        skill_stats = self._compute_skill_stats(gap_skill_ids) if enabled_weighted else {}
        enriched_all = self._enrich_and_rank(
            courses,
            gap_skill_ids,
            skill_match_scores=skill_match_scores or {},
            skill_stats=skill_stats,
            skill_label_by_id=skill_label_by_id,
            use_weighted_ranker=enabled_weighted,
        )
        top_courses = enriched_all[:max_courses]

        result = {
            "gap_skills":          skill_details,
            "recommended_courses": top_courses,
        }
        if debug_rank_explain:
            result["all_ranked_courses"] = enriched_all
        return result

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

    def _compute_skill_stats(self, gap_skill_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute per-skill stats so generic skills contribute less."""
        if not gap_skill_ids:
            return {}

        total_rows = self._neo4j.query("MATCH (c:Course) RETURN count(c) AS total_courses")
        total_courses = int(total_rows[0].get("total_courses", 0) or 0) if total_rows else 0
        total_courses = max(total_courses, 1)

        rows = self._neo4j.query(
            """
            MATCH (c:Course)-[:TEACHES]->(s:Skill)
            WHERE s.skill_id IN $ids
            RETURN s.skill_id AS skill_id, count(DISTINCT c) AS course_count
            """,
            {"ids": gap_skill_ids},
        )

        stats_by_skill: Dict[str, Dict[str, float]] = {}
        for row in rows:
            sid = str(row.get("skill_id") or "")
            if not sid:
                continue
            df = int(row.get("course_count") or 0)
            df_ratio = min(max(df / float(total_courses), 0.0), 1.0)
            # Smooth IDF in [~1, +inf): higher for rarer skills.
            idf = 1.0 + math.log((total_courses + 1.0) / (df + 1.0))
            generic_multiplier = self._generic_skill_multiplier(df_ratio)
            stats_by_skill[sid] = {
                "idf": idf,
                "df_ratio": df_ratio,
                "generic_multiplier": generic_multiplier,
            }

        # Fallback for missing skills in stats.
        for sid in gap_skill_ids:
            stats_by_skill.setdefault(
                sid,
                {
                    "idf": 1.0,
                    "df_ratio": 0.0,
                    "generic_multiplier": 1.0,
                },
            )
        return stats_by_skill

    def _generic_skill_multiplier(self, df_ratio: float) -> float:
        if not settings.COURSE_RANKER_GENERIC_PENALTY_ENABLED:
            return 1.0

        alpha = max(float(settings.COURSE_RANKER_GENERIC_PENALTY_ALPHA), 0.0)
        beta = max(float(settings.COURSE_RANKER_GENERIC_PENALTY_BETA), 0.0)
        min_mult = min(max(float(settings.COURSE_RANKER_GENERIC_PENALTY_MIN_MULT), 0.0), 1.0)
        penalty_scale = 1.0 - alpha * (df_ratio ** beta)
        return max(min_mult, min(1.0, penalty_scale))

    def _normalize_text(self, value: str) -> str:
        return " ".join(str(value or "").lower().split())

    def _course_has_anchor_hit(self, course_id: str, covered_skill_labels: List[str]) -> bool:
        anchors = NOISY_COURSE_ANCHOR_KEYWORDS.get(str(course_id).strip().upper())
        if not anchors:
            return False

        normalized_labels = [self._normalize_text(label) for label in covered_skill_labels if str(label).strip()]
        normalized_anchors = [self._normalize_text(anchor) for anchor in anchors]
        for label in normalized_labels:
            for anchor in normalized_anchors:
                if anchor and anchor in label:
                    return True
        return False

    def _enrich_and_rank(
        self,
        courses: List[Dict],
        gap_skill_ids: List[str],
        skill_match_scores: Dict[str, float],
        skill_stats: Dict[str, Dict[str, float]],
        skill_label_by_id: Dict[str, str],
        use_weighted_ranker: bool,
    ) -> List[Dict]:
        gap_set = set(gap_skill_ids)
        result = []
        for c in courses:
            covered = [sid for sid in c.get("covered_skill_ids", []) if sid in gap_set]
            coverage_count = len(covered)
            weighted_core = float(coverage_count)
            coverage_bonus = 0.0
            contributions: List[Dict[str, float]] = []
            weighted_score = float(coverage_count)
            covered_labels = [skill_label_by_id.get(sid, sid) for sid in covered]
            anchor_hit = self._course_has_anchor_hit(c["course_id"], covered_labels)

            if use_weighted_ranker and covered:
                weighted_core = 0.0
                for sid in covered:
                    match_score = float(skill_match_scores.get(sid, 1.0))
                    skill_stat = skill_stats.get(sid, {})
                    idf = float(skill_stat.get("idf", 1.0))
                    generic_multiplier = float(skill_stat.get("generic_multiplier", 1.0))
                    contribution = match_score * idf * generic_multiplier
                    weighted_core += contribution
                    contributions.append({
                        "skill_id": sid,
                        "match_score": round(match_score, 6),
                        "idf": round(idf, 6),
                        "generic_multiplier": round(generic_multiplier, 6),
                        "contribution": round(contribution, 6),
                    })
                # Bonus for covering multiple skills, slightly favoring broad-but-relevant courses.
                coverage_bonus = settings.COURSE_RANKER_COVERAGE_BONUS * max(0, coverage_count - 1)
                weighted_score = weighted_core + coverage_bonus
                if (
                    settings.COURSE_RANKER_ANCHOR_PENALTY_ENABLED
                    and str(c["course_id"]).strip().upper() in NOISY_COURSE_ANCHOR_KEYWORDS
                    and not anchor_hit
                    and coverage_count <= max(1, int(settings.COURSE_RANKER_ANCHOR_MAX_COVERAGE))
                ):
                    anchor_penalty = min(max(float(settings.COURSE_RANKER_ANCHOR_PENALTY), 0.0), 0.95)
                    weighted_score = weighted_score * (1.0 - anchor_penalty)
            elif covered:
                for sid in covered:
                    contributions.append({
                        "skill_id": sid,
                        "match_score": round(float(skill_match_scores.get(sid, 1.0)), 6),
                        "idf": 1.0,
                        "generic_multiplier": 1.0,
                        "contribution": 1.0,
                    })

            result.append({
                "course_id":       c["course_id"],
                "course_title":    c["course_title"],
                "category":        c.get("category", ""),
                "covered_gaps":    covered,
                "coverage_count":  coverage_count,
                "weighted_core":   round(weighted_core, 6),
                "coverage_bonus":  round(coverage_bonus, 6),
                "weighted_score":  round(weighted_score, 6),
                "anchor_hit": anchor_hit,
                "skill_contributions": contributions,
            })

        if use_weighted_ranker:
            # Ưu tiên weighted_score, tie-break bằng coverage_count.
            result.sort(key=lambda x: (x["weighted_score"], x["coverage_count"]), reverse=True)
        else:
            result.sort(key=lambda x: x["coverage_count"], reverse=True)

        for rank, row in enumerate(result, start=1):
            row["rank"] = rank
        return result
