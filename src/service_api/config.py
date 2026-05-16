"""Settings của service_api – load từ environment variables.

Post-refactor: removed Neo4j, added MySQL, S3/Minio, Airflow configs.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    APP_NAME: str = "VietCV Course Recommendation API"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False

    # MySQL
    DB_HOST: str = "mysql"
    DB_PORT: int = 3306
    DB_NAME: str = "vietcv"
    DB_USER: str = "vietcv_user"
    DB_PASSWORD: str = "secure_password"

    # Elasticsearch
    ELASTICSEARCH_HOSTS: str = "http://elasticsearch:9200"  # comma-separated
    ELASTICSEARCH_USERNAME: str = ""
    ELASTICSEARCH_PASSWORD: str = ""
    ELASTICSEARCH_TIMEOUT: int = 30
    ELASTICSEARCH_VECTOR_DIM: int = 1024

    # S3/Minio
    S3_ENDPOINT: str = "http://minio:9000"
    S3_BUCKET: str = "vietcv"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"

    # Embedding model (lazy-loaded khi serving)
    EMBEDDING_MODEL_NAME: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_MODEL_PATH: Optional[str] = None   # local path override
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 8

    # Airflow (to trigger DAGs)
    AIRFLOW_URL: str = "http://airflow-webserver:8080/api/v1"
    AIRFLOW_USERNAME: str = "airflow"
    AIRFLOW_PASSWORD: str = "airflow"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
