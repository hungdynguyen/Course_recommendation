"""Settings dataclasses cho data_factory.

Load từ YAML; mọi biến ${ENV_VAR} được expand từ environment.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


def _expand_env(s: str) -> str:
    """Replace ${VAR} hoặc $VAR bằng giá trị environment variable."""
    return re.sub(r"\$\{?(\w+)\}?", lambda m: os.environ.get(m.group(1), m.group(0)), str(s))


def _expand_dict(obj):
    """Đệ quy expand env vars trong dict/list/str."""
    if isinstance(obj, dict):
        return {k: _expand_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_dict(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj


# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathConfig:
    course_catalog_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    batch_size: int
    device: str
    normalize: bool
    model_path: Optional[str] = None


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
class DeduplicationConfig:
    """Tham số detect + merge skill trùng lặp."""
    fuzzy_threshold: float = 0.90        # Jaro-Winkler threshold cho fuzzy matching
    embedding_threshold: float = 0.92    # cosine similarity ngưỡng merge embedding
    batch_size: int = 512                # batch size khi tính cosine matrix
    canonical_strategy: str = "first"    # first | longest | most_common


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str
    batch_size: int


@dataclass(frozen=True)
class SoftSkillFilterConfig:
    enabled: bool = True
    exclude_terms: List[str] = field(default_factory=lambda: [
        "classroom discipline",
        "group activity organization",
        "group organization",
        "collaborate with team members",
        "team collaboration",
        "teamwork",
        "communication skills",
        "critical thinking",
        "self-directed learning",
        "self directed learning",
        "ethical responsibility",
        "social responsibility",
        "professional ethics",
        "working independently",
    ])


@dataclass(frozen=True)
class Settings:
    environment: str
    paths: PathConfig
    elasticsearch: ElasticsearchConfig
    embedding: EmbeddingConfig
    deduplication: DeduplicationConfig
    neo4j: Neo4jConfig
    soft_skill_filter: SoftSkillFilterConfig

    # ------------------------------------------------------------------

    @staticmethod
    def load(config_path: Path | str = None) -> "Settings":
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "settings.yaml"
        with Path(config_path).open("r", encoding="utf-8") as fh:
            raw = _expand_dict(yaml.safe_load(fh))

        p = raw.get("paths", {})
        em = raw.get("embedding", {})
        es = raw.get("elasticsearch", {})
        dd = raw.get("deduplication", {})
        n4 = raw.get("neo4j", {})
        ss = raw.get("soft_skill_filter", {})
        default_ss = SoftSkillFilterConfig()

        default_dd = DeduplicationConfig()
        return Settings(
            environment=raw.get("environment", "dev"),
            paths=PathConfig(
                course_catalog_dir=Path(p["course_catalog_dir"]),
                cache_dir=Path(p.get("cache_dir", ".cache")),
            ),
            elasticsearch=ElasticsearchConfig(
                hosts=es["hosts"],
                username=es.get("username", ""),
                password=es.get("password", ""),
                index=es.get("index", "course_skills"),
                vector_dim=int(es.get("vector_dim", 1024)),
                batch_size=int(es.get("batch_size", 128)),
                recreate_index=bool(es.get("recreate_index", True)),
            ),
            embedding=EmbeddingConfig(
                model_name=em.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
                model_path=em.get("model_path") or None,
                batch_size=int(em.get("batch_size", 16)),
                device=em.get("device", "cpu"),
                normalize=bool(em.get("normalize", True)),
            ),
            deduplication=DeduplicationConfig(
                fuzzy_threshold=float(dd.get("fuzzy_threshold", default_dd.fuzzy_threshold)),
                embedding_threshold=float(dd.get("embedding_threshold", default_dd.embedding_threshold)),
                batch_size=int(dd.get("batch_size", default_dd.batch_size)),
                canonical_strategy=dd.get("canonical_strategy", default_dd.canonical_strategy),
            ),
            neo4j=Neo4jConfig(
                uri=n4["uri"],
                username=n4["username"],
                password=n4["password"],
                database=n4.get("database", "neo4j"),
                batch_size=int(n4.get("batch_size", 5000)),
            ),
            soft_skill_filter=SoftSkillFilterConfig(
                enabled=bool(ss.get("enabled", default_ss.enabled)),
                exclude_terms=[
                    str(x).strip().lower()
                    for x in ss.get("exclude_terms", default_ss.exclude_terms)
                    if str(x).strip()
                ],
            ),
        )
