from __future__ import annotations

import logging
from typing import Optional

from service_api.config import settings
from shared.db.es_client import ElasticsearchClient
from shared.db.neo4j_client import Neo4jClient
from shared.embeddings.embedding_service import EmbeddingService
from service_api.services.skill_search import SkillSearchService
from service_api.services.gap_detection import GapDetectionService
from service_api.services.course_recommendation import CourseRecommendationService
from service_api.services.admin_ingest_demo import AdminIngestDemoService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton holders
# ---------------------------------------------------------------------------
_neo4j: Optional[Neo4jClient] = None
_es: Optional[ElasticsearchClient] = None
_embedding: Optional[EmbeddingService] = None
_skill_search: Optional[SkillSearchService] = None
_gap_detection: Optional[GapDetectionService] = None
_recommendation: Optional[CourseRecommendationService] = None
_admin_ingest_demo: Optional[AdminIngestDemoService] = None


# ---------------------------------------------------------------------------
# Client factories (dùng trong lifespan / Depends)
# ---------------------------------------------------------------------------

def get_neo4j() -> Neo4jClient:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
        )
    return _neo4j


def get_es() -> ElasticsearchClient:
    global _es
    if _es is None:
        hosts = [h.strip() for h in settings.ELASTICSEARCH_HOSTS.split(",")]
        _es = ElasticsearchClient(
            hosts=hosts,
            username=settings.ELASTICSEARCH_USERNAME,
            password=settings.ELASTICSEARCH_PASSWORD,
            timeout=settings.ELASTICSEARCH_TIMEOUT,
            index=settings.ELASTICSEARCH_INDEX,
        )
    return _es


def get_embedding() -> EmbeddingService:
    global _embedding
    if _embedding is None:
        _embedding = EmbeddingService(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_path=settings.EMBEDDING_MODEL_PATH,
            device=settings.EMBEDDING_DEVICE,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    return _embedding


# ---------------------------------------------------------------------------
# Service factories (FastAPI Depends)
# ---------------------------------------------------------------------------

def get_skill_search_service() -> SkillSearchService:
    global _skill_search
    if _skill_search is None:
        _skill_search = SkillSearchService(
            es=get_es(),
            embedding=get_embedding(),
        )
    return _skill_search


def get_gap_detection_service() -> GapDetectionService:
    global _gap_detection
    if _gap_detection is None:
        _gap_detection = GapDetectionService(
            embedding=get_embedding(),
            threshold=settings.GAP_SIMILARITY_THRESHOLD,
        )
    return _gap_detection


def get_recommendation_service() -> CourseRecommendationService:
    global _recommendation
    if _recommendation is None:
        _recommendation = CourseRecommendationService(neo4j=get_neo4j())
    return _recommendation


def get_admin_ingest_demo_service() -> AdminIngestDemoService:
    global _admin_ingest_demo
    if _admin_ingest_demo is None:
        _admin_ingest_demo = AdminIngestDemoService()
    return _admin_ingest_demo


def close_all() -> None:
    """Đóng tất cả DB connections khi shutdown."""
    global _neo4j, _es
    if _neo4j:
        _neo4j.close()
        _neo4j = None
    logger.info("All connections closed")
