"""Recommendations endpoint – luồng: gap detection → skill search → course recommendation."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException

from service_api.config import settings
from service_api.dependencies import (
    get_gap_detection_service,
    get_skill_search_service,
    get_recommendation_service,
)
from service_api.models.request import GapRecommendationRequest
from service_api.models.response import (
    CourseResponse,
    GapRecommendationResponse,
    SkillResponse,
)
from service_api.services.course_recommendation import CourseRecommendationService
from service_api.services.gap_detection import GapDetectionService
from service_api.services.skill_search import SkillSearchService

logger = logging.getLogger(__name__)
router = APIRouter()


def _normalize_keywords(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).replace("\n", ";").replace(",", ";").split(";")
    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return list(dict.fromkeys(cleaned))


def _build_enriched_gap_query(gap: str, jd_title: str, jd_keywords: List[str]) -> str:
    parts = [str(gap).strip()]
    title = str(jd_title or "").strip()
    if title:
        parts.append(f"role {title}")
    if jd_keywords:
        parts.append(" ".join(jd_keywords[:8]))
    return " ; ".join([p for p in parts if p])


def _dynamic_top1_threshold(gap_name: str, base_threshold: float, short_gap_bonus: float) -> float:
    words = [w for w in str(gap_name).strip().split() if w]
    if len(words) <= 2:
        return min(0.99, base_threshold + short_gap_bonus)
    return base_threshold


def _select_canonical_candidates(
    search_results: Dict[str, List[Dict]],
    min_total: int,
    max_total: int,
    min_score: float,
    score_margin: float,
    top1_gate_enabled: bool,
    top1_min_score: float,
    top1_short_gap_bonus: float,
) -> Tuple[List[Dict], List[str]]:
    """Select canonical skill candidates with bounded length.

    Strategy (practical, no extra JD/CV fields needed):
    - Keep top-1 per gap as mandatory.
    - Keep extra hits only if score is high enough and close to top-1.
    - Backfill from deferred hits if total is too short.
    - Cap total count to avoid long noisy candidate lists.
    """
    mandatory: List[Dict] = []
    optional: List[Dict] = []
    deferred: List[Dict] = []

    for _gap_name, hits in search_results.items():
        if not hits:
            continue

        top = hits[0]
        top_score = float(top.get("score") or 0.0)

        if top1_gate_enabled:
            gap_threshold = _dynamic_top1_threshold(
                gap_name=_gap_name,
                base_threshold=top1_min_score,
                short_gap_bonus=top1_short_gap_bonus,
            )
            if top_score < gap_threshold:
                # Không đủ tin cậy: bỏ qua gap để tránh bơm nhiễu.
                continue

        mandatory.append(top)

        for h in hits[1:]:
            score = float(h.get("score") or 0.0)
            if score >= min_score and (top_score - score) <= score_margin:
                optional.append(h)
            else:
                deferred.append(h)

    # Sort by score so higher-confidence candidates are used first.
    optional.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    deferred.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    selected: List[Dict] = []
    seen_ids: set = set()

    def add_hit(hit: Dict) -> bool:
        sid = hit.get("skill_id")
        if not sid or sid in seen_ids:
            return False
        seen_ids.add(sid)
        selected.append(hit)
        return True

    for hit in mandatory:
        add_hit(hit)
    for hit in optional:
        if len(selected) >= max_total:
            break
        add_hit(hit)

    if len(selected) < min_total:
        for hit in deferred:
            if len(selected) >= min_total or len(selected) >= max_total:
                break
            add_hit(hit)

    canonical_ids = [h["skill_id"] for h in selected if h.get("skill_id")]
    return selected, canonical_ids


@router.post("/gap", response_model=GapRecommendationResponse)
def recommend_for_skill_gap(
    request: GapRecommendationRequest,
    gap_svc: GapDetectionService = Depends(get_gap_detection_service),
    skill_svc: SkillSearchService = Depends(get_skill_search_service),
    rec_svc: CourseRecommendationService = Depends(get_recommendation_service),
):
    """
    Gợi ý khoá học dựa trên skill gap giữa JD và CV.

    Luồng:
    1. Tìm gaps = skills trong JD mà CV không có (cosine similarity < threshold).
    2. Với mỗi gap, tìm canonical skills tương ứng trong Knowledge Graph (ES kNN).
    3. Query Neo4j để tìm khoá học dạy các canonical skills đó.
    4. Trả về khoá học sắp xếp theo số gap được cover.
    """
    try:
        # Bước 1: Gap detection
        gaps: List[str] = gap_svc.find_gaps(
            jd_skills=request.jd_skills,
            cv_skills=request.cv_skills,
            threshold=request.gap_threshold,
        )

        if not gaps:
            return GapRecommendationResponse(
                jd_skill_count=len(request.jd_skills),
                cv_skill_count=len(request.cv_skills),
                gap_skills=[],
                raw_gaps=[],
                recommended_courses=[],
            )

        # Bước 2: Tìm canonical skills trong KG cho mỗi gap (enrich bằng JD metadata nếu có)
        jd_keywords = _normalize_keywords(request.jd_keywords)
        jd_title = str(request.jd_title or "")

        search_results: Dict[str, List[Dict]] = {}
        for gap in gaps:
            query_text = _build_enriched_gap_query(
                gap=gap,
                jd_title=jd_title,
                jd_keywords=jd_keywords,
            )
            search_results[gap] = skill_svc.search_by_text(
                skill_name=query_text,
                limit=settings.SKILL_SEARCH_RAW_LIMIT,
            )

        # Chọn canonical candidates theo ngưỡng động để tránh quá dài / quá ngắn.
        bounded_min = min(settings.SKILL_CANDIDATE_MIN_TOTAL, settings.SKILL_CANDIDATE_MAX_TOTAL)
        bounded_max = max(settings.SKILL_CANDIDATE_MIN_TOTAL, settings.SKILL_CANDIDATE_MAX_TOTAL)
        target_min = min(bounded_max, max(len(gaps), min(request.max_courses, bounded_min)))

        selected_hits, canonical_ids = _select_canonical_candidates(
            search_results=search_results,
            min_total=target_min,
            max_total=bounded_max,
            min_score=settings.SKILL_SEARCH_MIN_SCORE,
            score_margin=settings.SKILL_SEARCH_SCORE_MARGIN,
            top1_gate_enabled=settings.SKILL_TOP1_CONFIDENCE_GATE_ENABLED,
            top1_min_score=settings.SKILL_TOP1_MIN_SCORE,
            top1_short_gap_bonus=settings.SKILL_TOP1_SHORT_GAP_BONUS,
        )

        gap_skill_details: List[SkillResponse] = [
            SkillResponse(
                skill_id=h["skill_id"],
                label=h.get("canonical_label"),
                aliases=h.get("aliases", []),
                description=h.get("description"),
                category=h.get("category"),
                score=h.get("score"),
            )
            for h in selected_hits
            if h.get("skill_id")
        ]

        if not canonical_ids:
            return GapRecommendationResponse(
                jd_skill_count=len(request.jd_skills),
                cv_skill_count=len(request.cv_skills),
                gap_skills=[],
                raw_gaps=gaps,
                recommended_courses=[],
            )

        # Bước 3: Tìm khoá học trong Neo4j
        skill_match_scores = {
            str(h.get("skill_id")): float(h.get("score") or 1.0)
            for h in selected_hits
            if h.get("skill_id")
        }
        rec_result = rec_svc.recommend_for_gaps(
            gap_skill_ids=canonical_ids,
            max_courses=request.max_courses,
            skill_match_scores=skill_match_scores,
            use_weighted_ranker=settings.COURSE_WEIGHTED_RANKER_ENABLED,
        )

        courses = [
            CourseResponse(
                course_id=c["course_id"],
                course_title=c["course_title"],
                category=c.get("category"),
                covered_gaps=c.get("covered_gaps", []),
                coverage_count=c.get("coverage_count", 0),
            )
            for c in rec_result["recommended_courses"]
        ]

        return GapRecommendationResponse(
            jd_skill_count=len(request.jd_skills),
            cv_skill_count=len(request.cv_skills),
            gap_skills=gap_skill_details,
            raw_gaps=gaps,
            recommended_courses=courses,
        )

    except Exception as exc:
        logger.exception("Recommendation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
