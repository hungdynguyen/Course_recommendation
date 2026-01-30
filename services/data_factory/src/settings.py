from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class ElasticsearchConfig:
    hosts: List[str]
    username: str
    password: str
    index: str
    vector_dim: int
    batch_size: int
    recreate_index: bool


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model_name: str
    model_path: Optional[str]  # Local path to trained model (overrides model_name if provided)
    batch_size: int
    device: str
    normalize: bool


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    model_name: str = ""
    device: str = "cpu"
    batch_size: int = 4
    max_length: int = 256
    torch_dtype: Optional[str] = None


@dataclass(frozen=True)
class PipelineConfig:
    min_description_length: int
    max_records: Optional[int]
    concurrency: int
    flush_every: int


@dataclass(frozen=True)
class MappingConfig:
    min_similarity: float
    rerank_top_k: int = 50


@dataclass(frozen=True)
class MySQLConfig:
    enabled: bool
    host: str
    port: int
    username: str
    password: str
    database: str
    esco_table: str
    mapping_table: str
    charset: str
    connect_timeout: int
    batch_size: int


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str
    batch_size: int


@dataclass(frozen=True)
class PathConfig:
    esco_skills: Path
    esco_skill_relations: Path
    course_catalog_dir: Path
    processed_embeddings_dir: Path
    course_skill_mappings_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class Settings:
    environment: str
    paths: PathConfig
    elasticsearch: ElasticsearchConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    pipeline: PipelineConfig
    mapping: MappingConfig
    mysql: MySQLConfig
    neo4j: Neo4jConfig

    @staticmethod
    def load(config_path: Path | str) -> "Settings":
        with Path(config_path).open("r", encoding="utf-8") as fh:
            raw_config = yaml.safe_load(fh)
        paths = raw_config.get("paths", {})
        reranker_cfg = raw_config.get("reranker") or {}
        default_reranker = RerankerConfig()
        merged_reranker = {**default_reranker.__dict__, **reranker_cfg}
        return Settings(
            environment=raw_config.get("environment", "dev"),
            paths=PathConfig(
                esco_skills=Path(paths["esco_skills"]),
                esco_skill_relations=Path(paths["esco_skill_relations"]),
                course_catalog_dir=Path(paths["course_catalog_dir"]),
                processed_embeddings_dir=Path(paths["processed_embeddings_dir"]),
                course_skill_mappings_dir=Path(paths["course_skill_mappings_dir"]),
                cache_dir=Path(paths["cache_dir"]),
            ),
            elasticsearch=ElasticsearchConfig(**raw_config["elasticsearch"]),
            embedding=EmbeddingConfig(**raw_config["embedding"]),
            reranker=RerankerConfig(**merged_reranker),
            pipeline=PipelineConfig(**raw_config["pipeline"]),
            mapping=MappingConfig(**raw_config["mapping"]),
            mysql=MySQLConfig(**raw_config["mysql"]),
            neo4j=Neo4jConfig(**raw_config["neo4j"]),
        )
