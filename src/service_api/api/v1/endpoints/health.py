"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter
import logging

from service_api.config import settings
from service_api.dependencies import get_neo4j, get_es
from service_api.models.response import HealthCheckResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    """Kiểm tra trạng thái kết nối tất cả databases."""
    databases = {}
    try:
        databases["neo4j"] = get_neo4j().verify_connection()
    except Exception:
        databases["neo4j"] = False
    try:
        databases["elasticsearch"] = get_es().verify_connection()
    except Exception:
        databases["elasticsearch"] = False

    status = "healthy" if all(databases.values()) else "degraded"
    return HealthCheckResponse(
        status=status,
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        databases=databases,
    )
