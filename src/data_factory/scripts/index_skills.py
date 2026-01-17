from __future__ import annotations

import logging
from pathlib import Path

from src.data_factory.src.embeddings.embedding_service import EmbeddingService
from src.data_factory.src.pipelines.skill_embedding_pipeline import SkillEmbeddingPipeline
from src.data_factory.src.services.elasticsearch_service import ElasticsearchService
from src.data_factory.src.services.mysql_service import MySQLService
from src.data_factory.src.utils.config_utils import load_config
from src.data_factory.src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger("data_factory.cli")


def main() -> None:
    default_logging = Path(__file__).resolve().parent.parent / "config" / "logging.yaml"
    if default_logging.exists():
        setup_logging(default_logging)

    settings = load_config()
    LOGGER.info("Running skill indexing with environment=%s", settings.environment)

    embedding_service = EmbeddingService(settings.embedding)
    es_service = ElasticsearchService(settings.elasticsearch)
    mysql_service = MySQLService(settings.mysql)
    pipeline = SkillEmbeddingPipeline(settings, embedding_service, es_service, mysql_service)
    pipeline.run()


if __name__ == "__main__":
    main()
