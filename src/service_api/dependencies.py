"""Dependency injection – singleton holders and factories for FastAPI Depends().

Post-refactor: removed Neo4j, added MySQL, S3, CourseES clients.
"""
from __future__ import annotations

import logging
from typing import Optional

from service_api.config import settings
from shared.storage.mysql_client import MySQLClient
from shared.storage.s3_client import S3Client
from shared.storage.es_client import CourseElasticsearchClient
from shared.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton holders
# ---------------------------------------------------------------------------
_mysql: Optional[MySQLClient] = None
_s3: Optional[S3Client] = None
_course_es: Optional[CourseElasticsearchClient] = None
_embedding: Optional[EmbeddingService] = None


# ---------------------------------------------------------------------------
# Client factories (dùng trong lifespan / Depends)
# ---------------------------------------------------------------------------

def get_mysql() -> MySQLClient:
    global _mysql
    if _mysql is None:
        _mysql = MySQLClient(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            username=settings.DB_USER,
            password=settings.DB_PASSWORD,
        )
    return _mysql


def get_s3() -> S3Client:
    global _s3
    if _s3 is None:
        _s3 = S3Client(
            endpoint=settings.S3_ENDPOINT,
            bucket=settings.S3_BUCKET,
            access_key=settings.S3_ACCESS_KEY,
            secret_key=settings.S3_SECRET_KEY,
        )
    return _s3


def get_course_es() -> CourseElasticsearchClient:
    global _course_es
    if _course_es is None:
        hosts = [h.strip() for h in settings.ELASTICSEARCH_HOSTS.split(",")]
        _course_es = CourseElasticsearchClient(
            hosts=hosts,
            username=settings.ELASTICSEARCH_USERNAME,
            password=settings.ELASTICSEARCH_PASSWORD,
            timeout=settings.ELASTICSEARCH_TIMEOUT,
            vector_dim=settings.ELASTICSEARCH_VECTOR_DIM,
        )
    return _course_es


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
# Cleanup
# ---------------------------------------------------------------------------

def close_all() -> None:
    """Đóng tất cả connections khi shutdown."""
    global _mysql
    if _mysql:
        _mysql.close()
        _mysql = None
    logger.info("All connections closed")
