"""Recommendations endpoint – luồng: gap detection → skill search → course recommendation."""
from __future__ import annotations

import logging
from typing import List

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

        # Bước 2: Tìm canonical skills trong KG cho mỗi gap
        search_results = skill_svc.search_batch(
            skill_names=gaps,
            limit_per_skill=settings.SKILL_SEARCH_LIMIT,
        )

        # Gộp unique canonical skill_ids
        seen_ids: set = set()
        gap_skill_details: List[SkillResponse] = []
        canonical_ids: List[str] = []
        for gap_name, hits in search_results.items():
            for h in hits:
                sid = h["skill_id"]
                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    canonical_ids.append(sid)
                    gap_skill_details.append(SkillResponse(
                        skill_id=sid,
                        label=h["canonical_label"],
                        aliases=h.get("aliases", []),
                        description=h.get("description"),
                        category=h.get("category"),
                        score=h.get("score"),
                    ))

        if not canonical_ids:
            return GapRecommendationResponse(
                jd_skill_count=len(request.jd_skills),
                cv_skill_count=len(request.cv_skills),
                gap_skills=[],
                raw_gaps=gaps,
                recommended_courses=[],
            )

        # Bước 3: Tìm khoá học trong Neo4j
        rec_result = rec_svc.recommend_for_gaps(
            gap_skill_ids=canonical_ids,
            max_courses=request.max_courses,
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
