"""API v1 router."""
from fastapi import APIRouter

from service_api.api.v1.endpoints import health, recommendations, skills

api_router = APIRouter()
api_router.include_router(health.router,           tags=["health"])
api_router.include_router(skills.router,           prefix="/skills",          tags=["skills"])
api_router.include_router(recommendations.router,  prefix="/recommendations", tags=["recommendations"])
