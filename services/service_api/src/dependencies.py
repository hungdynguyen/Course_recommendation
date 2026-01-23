"""Dependency injection for FastAPI"""
from functools import lru_cache

from src.core.database import (
    get_neo4j_client,
    get_elasticsearch_client,
    get_mysql_client,
    Neo4jClient,
    ElasticsearchClient,
    MySQLClient,
)
from src.services.course_recommendation import CourseRecommendationService
from src.services.skill_search import SkillSearchService


# Service instances cache
_recommendation_service = None
_skill_search_service = None


def get_recommendation_service() -> CourseRecommendationService:
    """
    Get or create CourseRecommendationService instance.
    This is a dependency for FastAPI endpoints.
    """
    global _recommendation_service
    if _recommendation_service is None:
        neo4j_client = get_neo4j_client()
        _recommendation_service = CourseRecommendationService(neo4j_client)
    return _recommendation_service


def get_skill_search_service() -> SkillSearchService:
    """
    Get or create SkillSearchService instance.
    This is a dependency for FastAPI endpoints.
    """
    global _skill_search_service
    if _skill_search_service is None:
        es_client = get_elasticsearch_client()
        mysql_client = get_mysql_client()
        _skill_search_service = SkillSearchService(es_client, mysql_client)
    return _skill_search_service


def reset_services():
    """Reset service instances (useful for testing)"""
    global _recommendation_service, _skill_search_service
    _recommendation_service = None
    _skill_search_service = None
