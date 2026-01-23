"""Health check endpoints"""
from fastapi import APIRouter, Depends
import logging

from src.models.response import HealthCheckResponse
from src.core.database import (
    get_neo4j_client,
    get_elasticsearch_client,
    get_mysql_client,
    Neo4jClient,
    ElasticsearchClient,
    MySQLClient,
)
from src.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
def health_check(
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    es: ElasticsearchClient = Depends(get_elasticsearch_client),
    mysql: MySQLClient = Depends(get_mysql_client),
):
    """
    Health check endpoint - verify all services are operational
    """
    databases = {
        "neo4j": neo4j.verify_connection(),
        "elasticsearch": es.verify_connection(),
        "mysql": mysql.verify_connection(),
    }
    
    all_healthy = all(databases.values())
    status = "healthy" if all_healthy else "degraded"
    
    return HealthCheckResponse(
        status=status,
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        databases=databases,
    )
