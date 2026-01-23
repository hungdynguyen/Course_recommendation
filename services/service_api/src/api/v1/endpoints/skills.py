"""Skill search endpoints"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List
import logging

from src.models.response import SkillResponse
from src.services.skill_search import SkillSearchService
from src.dependencies import get_skill_search_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/search", response_model=List[SkillResponse])
def search_skills(
    query: str = Query(..., description="Search query for skills"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    service: SkillSearchService = Depends(get_skill_search_service),
):
    """
    Search ESCO skills by name (fuzzy matching via MySQL).
    Returns list of matching skills with IDs.
    
    - **query**: Search text for skill names
    - **limit**: Maximum number of results to return
    """
    try:
        results = service.search_by_name(query, limit=limit)
        
        # Convert to response model
        return [
            SkillResponse(
                skill_id=r["skill_id"],
                label=r["label"],
                description=r.get("description"),
                skill_type=r.get("skill_type"),
                score=r.get("score"),
            )
            for r in results
        ]
    except Exception as e:
        logger.error(f"Skill search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Skill search failed: {str(e)}")


@router.get("/{skill_id}", response_model=SkillResponse)
def get_skill_details(
    skill_id: str,
    service: SkillSearchService = Depends(get_skill_search_service),
):
    """
    Get detailed information about a specific skill.
    
    - **skill_id**: ESCO skill ID
    """
    try:
        skill = service.get_skill_by_id(skill_id)
        
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill {skill_id} not found")
        
        return SkillResponse(
            skill_id=skill["skill_id"],
            label=skill["label"],
            description=skill.get("description"),
            skill_type=skill.get("skill_type"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get skill error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get skill: {str(e)}")
