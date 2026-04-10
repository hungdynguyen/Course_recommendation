#!/usr/bin/env python
"""Entry point: xây dựng Knowledge Graph từ course skills được trích xuất bằng LLM.

Chạy từ thư mục gốc dự án:
    PYTHONPATH=src python src/data_factory/scripts/build_graph.py [--config path/to/settings.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the Python path khi chạy trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.db.es_client import ElasticsearchIndexClient
from shared.db.neo4j_client import Neo4jBatchClient
from shared.embeddings.embedding_service import EmbeddingService
from shared.utils.logging_utils import setup_logging
from data_factory.settings import Settings
from data_factory.pipelines.graph_build_pipeline import GraphBuildPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Build course skill Knowledge Graph")
    p.add_argument("--config", default=None, help="Path to settings.yaml (default: data_factory/config/settings.yaml)")
    p.add_argument("--no-clear", action="store_true", help="Không xóa graph trước khi build")
    p.add_argument("--no-metrics-report", action="store_true", help="Không output metrics report JSON")
    p.add_argument(
        "--metrics-output",
        default="build_graph_metrics_report.json",
        help="Đường dẫn file JSON output cho metrics/report",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log_cfg = Path(__file__).resolve().parents[2] / "config" / "logging.yaml"
    setup_logging(log_cfg if log_cfg.exists() else None)

    logger = logging.getLogger("data_factory.build_graph")

    settings = Settings.load(args.config)
    logger.info("Environment: %s", settings.environment)
    logger.info("Course catalog: %s", settings.paths.course_catalog_dir)
    logger.info(
        "Dedup thresholds: fuzzy=%.3f embedding=%.3f",
        settings.deduplication.fuzzy_threshold,
        settings.deduplication.embedding_threshold,
    )

    embedding = EmbeddingService(
        model_name=settings.embedding.model_name,
        model_path=settings.embedding.model_path,
        device=settings.embedding.device,
        batch_size=settings.embedding.batch_size,
        normalize=settings.embedding.normalize,
    )

    es = ElasticsearchIndexClient(
        hosts=settings.elasticsearch.hosts,
        username=settings.elasticsearch.username,
        password=settings.elasticsearch.password,
        vector_dim=settings.elasticsearch.vector_dim,
        batch_size=settings.elasticsearch.batch_size,
        recreate_index=settings.elasticsearch.recreate_index,
        index=settings.elasticsearch.index,
    )

    neo4j = Neo4jBatchClient(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password,
        database=settings.neo4j.database,
        batch_size=settings.neo4j.batch_size,
    )

    pipeline = GraphBuildPipeline(
        settings=settings,
        neo4j=neo4j,
        es=es,
        embedding=embedding,
    )
    pipeline.run(
        clear_existing=not args.no_clear,
        output_metrics_report=not args.no_metrics_report,
        metrics_output_path=args.metrics_output,
    )
    neo4j.close()


if __name__ == "__main__":
    main()
