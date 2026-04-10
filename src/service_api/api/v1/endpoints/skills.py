"""Skills endpoint – tìm kiếm canonical skills trong KG."""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from service_api.dependencies import get_skill_search_service
from service_api.models.response import SkillResponse, SkillSearchResponse
from service_api.services.skill_search import SkillSearchService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search", response_model=SkillSearchResponse)
def search_skills(
    query: str = Query(..., description="Tên skill cần tìm"),
    limit: int = Query(10, ge=1, le=100),
    svc: SkillSearchService = Depends(get_skill_search_service),
):
    """Tìm kiếm canonical skills bằng vector similarity (dùng embedding model)."""
    try:
        hits = svc.search_by_text(query, limit=limit)
        results = [
            SkillResponse(
                skill_id=h["skill_id"],
                label=h["canonical_label"],
                aliases=h.get("aliases", []),
                description=h.get("description"),
                category=h.get("category"),
                score=h.get("score"),
            )
            for h in hits
        ]
        return SkillSearchResponse(query=query, results=results)
    except Exception as exc:
        logger.exception("Skill search error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{skill_id}", response_model=SkillResponse)
def get_skill(
    skill_id: str,
    svc: SkillSearchService = Depends(get_skill_search_service),
):
    """Lấy thông tin chi tiết của một canonical skill theo ID."""
    skill = svc.get_skill_by_id(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill {skill_id!r} not found")
    return SkillResponse(
        skill_id=skill["skill_id"],
        label=skill["canonical_label"],
        aliases=skill.get("aliases", []),
        description=skill.get("description"),
        category=skill.get("category"),
    )
