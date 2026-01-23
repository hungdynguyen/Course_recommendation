from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

from src.embeddings.embedding_service import EmbeddingService
from src.pipelines.course_skill_mapping_pipeline import CourseSkillMappingPipeline
from src.services.mysql_course_skill_service import MySQLCourseSkillService
from src.services.reranker_service import RerankerService
from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger("data_factory.map_course_skills")


def main() -> None:
    default_logging = Path(__file__).resolve().parent.parent / "config" / "logging.yaml"
    if default_logging.exists():
        setup_logging(default_logging)

    settings = load_config()
    LOGGER.info("Running course skill mapping with environment=%s", settings.environment)

    # Phase 1: Load embedding model, encode, then cleanup
    LOGGER.info("[Phase 1/2] Loading embedding model...")
    embedding_service = EmbeddingService(settings.embedding)
    mysql_service = MySQLCourseSkillService(settings.mysql)
    
    # Create pipeline without reranker first
    pipeline = CourseSkillMappingPipeline(
        settings,
        embedding_service,
        mysql_service,
        reranker_service=None,
    )
    
    # Run embedding phase and get intermediate results
    course_embeddings, course_skills, esco_metadata, esco_embeddings = pipeline.run_embedding_phase()
    
    # Cleanup embedding model
    LOGGER.info("[Phase 1/2] Cleaning up embedding model from memory...")
    del embedding_service._model
    del embedding_service
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        LOGGER.info("CUDA memory cleared")
    
    # Phase 2: Load reranker model and complete mapping
    if settings.reranker.enabled:
        LOGGER.info("[Phase 2/2] Loading reranker model...")
        reranker_service = RerankerService(settings.reranker)
    else:
        reranker_service = None
    
    pipeline._reranker = reranker_service
    pipeline.run_mapping_phase(course_skills, course_embeddings, esco_metadata, esco_embeddings)


if __name__ == "__main__":
    main()
