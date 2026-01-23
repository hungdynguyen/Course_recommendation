"""API v1 router aggregator"""
from fastapi import APIRouter

from src.api.v1.endpoints import health, skills, recommendations

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(skills.router, prefix="/skills", tags=["skills"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
