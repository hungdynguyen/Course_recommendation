"""Settings của service_api – load từ environment variables."""
from __future__ import annotations

from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    APP_NAME: str = "VietCV Course Recommendation API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Neo4j
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"
    NEO4J_DATABASE: str = "neo4j"

    # Elasticsearch
    ELASTICSEARCH_HOSTS: str = "http://elasticsearch:9200"  # comma-separated
    ELASTICSEARCH_USERNAME: str = ""
    ELASTICSEARCH_PASSWORD: str = ""
    ELASTICSEARCH_INDEX: str = "course_skills"
    ELASTICSEARCH_TIMEOUT: int = 30
    ELASTICSEARCH_VECTOR_DIM: int = 1024

    # Embedding model (lazy-loaded khi serving)
    EMBEDDING_MODEL_NAME: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_MODEL_PATH: Optional[str] = None   # local path override
    EMBEDDING_DEVICE: str = "cpu"

    # Gap detection
    GAP_SIMILARITY_THRESHOLD: float = 0.80  # nếu max sim(JD_skill, CV_skills) < threshold → gap
    SKILL_SEARCH_LIMIT: int = 3             # số canonical skills tìm cho mỗi gap

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
