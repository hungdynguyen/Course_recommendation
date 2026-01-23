from __future__ import annotations

import logging
from pathlib import Path

from src.pipelines.graph_build_pipeline import GraphBuildPipeline
from src.services.neo4j_service import Neo4jService
from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger("data_factory.build_graph")


def main() -> None:
    default_logging = Path(__file__).resolve().parent.parent / "config" / "logging.yaml"
    if default_logging.exists():
        setup_logging(default_logging)

    settings = load_config()
    LOGGER.info("Running graph build with environment=%s", settings.environment)

    neo4j_service = Neo4jService(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password,
        database=settings.neo4j.database,
    )

    pipeline = GraphBuildPipeline(settings, neo4j_service)
    pipeline.run(clear_existing=True)  # Clear existing on first run


if __name__ == "__main__":
    main()
