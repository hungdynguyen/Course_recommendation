"""API v1 router."""
from fastapi import APIRouter

from service_api.api.v1.endpoints import admin_demo, health, jds, recommendations, skills

api_router = APIRouter()
api_router.include_router(health.router,           tags=["health"])
api_router.include_router(skills.router,           prefix="/skills",          tags=["skills"])
api_router.include_router(recommendations.router,  prefix="/recommendations", tags=["recommendations"])
api_router.include_router(jds.router,              prefix="/jds",             tags=["jds"])
api_router.include_router(admin_demo.router,       prefix="/admin",           tags=["admin"])
