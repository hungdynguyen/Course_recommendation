"""Application configuration settings"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    APP_NAME: str = "VietCV Course Recommendation API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Neo4j Settings
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"
    
    # Elasticsearch Settings
    ELASTICSEARCH_HOST: str = "http://elasticsearch:9200"
    ELASTICSEARCH_TIMEOUT: int = 30
    ESCO_SKILLS_INDEX: str = "esco_skills"
    
    # MySQL Settings
    MYSQL_HOST: str = "mysql"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "vietcv"
    MYSQL_PASSWORD: str = "yourpassword"
    MYSQL_DATABASE: str = "vietcv"
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()
