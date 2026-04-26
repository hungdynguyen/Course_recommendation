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
    EMBEDDING_BATCH_SIZE: int = 2

    # Gap detection
    GAP_SIMILARITY_THRESHOLD: float = 0.80  # nếu max sim(JD_skill, CV_skills) < threshold → gap
    SKILL_SEARCH_LIMIT: int = 3             # số canonical skills tìm cho mỗi gap
    SKILL_SEARCH_RAW_LIMIT: int = 5         # số hits thô lấy từ ES trước khi lọc
    SKILL_SEARCH_MIN_SCORE: float = 0.80    # ngưỡng điểm tối thiểu cho candidate phụ
    SKILL_SEARCH_SCORE_MARGIN: float = 0.08 # chỉ giữ candidate phụ nếu gần top-1 theo margin
    SKILL_TOP1_CONFIDENCE_GATE_ENABLED: bool = True
    SKILL_TOP1_MIN_SCORE: float = 0.87
    SKILL_TOP1_SHORT_GAP_BONUS: float = 0.00
    SKILL_CANDIDATE_MIN_TOTAL: int = 10     # số canonical candidates tối thiểu sau lọc
    SKILL_CANDIDATE_MAX_TOTAL: int = 24     # số canonical candidates tối đa sau lọc

    # Course ranking
    COURSE_WEIGHTED_RANKER_ENABLED: bool = True
    COURSE_RANKER_COVERAGE_BONUS: float = 0.15
    COURSE_RANKER_GENERIC_PENALTY_ENABLED: bool = False
    COURSE_RANKER_GENERIC_PENALTY_ALPHA: float = 0.80
    COURSE_RANKER_GENERIC_PENALTY_BETA: float = 1.20
    COURSE_RANKER_GENERIC_PENALTY_MIN_MULT: float = 0.35
    COURSE_RANKER_ANCHOR_PENALTY_ENABLED: bool = False
    COURSE_RANKER_ANCHOR_PENALTY: float = 0.25
    COURSE_RANKER_ANCHOR_MAX_COVERAGE: int = 2

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
