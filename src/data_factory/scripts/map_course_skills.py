from __future__ import annotations

import logging
from pathlib import Path

from src.data_factory.src.embeddings.embedding_service import EmbeddingService
from src.data_factory.src.pipelines.course_skill_mapping_pipeline import CourseSkillMappingPipeline
from src.data_factory.src.services.mysql_course_skill_service import MySQLCourseSkillService
from src.data_factory.src.services.reranker_service import RerankerService
from src.data_factory.src.utils.config_utils import load_config
from src.data_factory.src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger("data_factory.map_course_skills")


def main() -> None:
    default_logging = Path(__file__).resolve().parent.parent / "config" / "logging.yaml"
    if default_logging.exists():
        setup_logging(default_logging)

    settings = load_config()
    LOGGER.info("Running course skill mapping with environment=%s", settings.environment)

    embedding_service = EmbeddingService(settings.embedding)
    mysql_service = MySQLCourseSkillService(settings.mysql)
    reranker_service = RerankerService(settings.reranker) if settings.reranker.enabled else None
    pipeline = CourseSkillMappingPipeline(
        settings,
        embedding_service,
        mysql_service,
        reranker_service,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
