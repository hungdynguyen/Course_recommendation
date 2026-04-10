"""Unified Elasticsearch client – dùng chung cho data_factory (index) lẫn service_api (search).

- `ElasticsearchClient`: thin wrapper dùng trong service_api (vector search).
- `ElasticsearchIndexClient`: mở rộng thêm index management cho data_factory.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger(__name__)

# Tên index mặc định cho canonical skills
SKILLS_INDEX = "course_skills"


class ElasticsearchClient:
    """Read-oriented ES client dùng trong service_api."""

    def __init__(
        self,
        hosts: List[str],
        username: str = "",
        password: str = "",
        timeout: int = 30,
        index: str = SKILLS_INDEX,
    ) -> None:
        basic_auth = (username, password) if username else None
        self.client = Elasticsearch(hosts=hosts, basic_auth=basic_auth, request_timeout=timeout)
        self.index = index
        logger.info("ElasticsearchClient connected to %s (index=%s)", hosts, index)

    def verify_connection(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False

    def vector_search(
        self,
        vector: List[float],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[dict]:
        """KNN / cosine similarity search trả về list dicts với skill metadata + score."""
        query = {
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "num_candidates": max(limit * 5, 50),
                "k": limit,
            },
            "_source": ["skill_id", "canonical_label", "aliases", "description", "category"],
        }
        response = self.client.search(index=self.index, body=query)
        results = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
            if score < min_score:
                continue
            src = hit["_source"]
            results.append({
                "skill_id":        src.get("skill_id"),
                "canonical_label": src.get("canonical_label"),
                "aliases":         src.get("aliases", []),
                "description":     src.get("description", ""),
                "category":        src.get("category", ""),
                "score":           score,
            })
        return results

    def text_search(self, query: str, limit: int = 10) -> List[dict]:
        """Full-text search trên canonical_label + aliases."""
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["canonical_label^3", "aliases^2", "description"],
                    "type": "best_fields",
                }
            },
            "size": limit,
            "_source": ["skill_id", "canonical_label", "aliases", "description", "category"],
        }
        response = self.client.search(index=self.index, body=body)
        return [
            {
                "skill_id":        h["_source"].get("skill_id"),
                "canonical_label": h["_source"].get("canonical_label"),
                "score":           h["_score"],
            }
            for h in response["hits"]["hits"]
        ]


class ElasticsearchIndexClient(ElasticsearchClient):
    """Write-oriented ES client dùng trong data_factory."""

    def __init__(
        self,
        hosts: List[str],
        username: str = "",
        password: str = "",
        vector_dim: int = 1024,
        batch_size: int = 128,
        recreate_index: bool = True,
        index: str = SKILLS_INDEX,
    ) -> None:
        super().__init__(hosts=hosts, username=username, password=password, index=index)
        self._vector_dim = vector_dim
        self._batch_size = batch_size
        self._recreate_index = recreate_index

    def ensure_index(self) -> None:
        exists = self.client.indices.exists(index=self.index)
        if exists and self._recreate_index:
            logger.info("Recreating ES index %s", self.index)
            self.client.indices.delete(index=self.index)
            exists = False
        if not exists:
            self.client.indices.create(
                index=self.index,
                mappings=self._build_mappings(),
                settings={"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            )
            logger.info("Created ES index %s", self.index)

    def bulk_index(self, documents: Iterable[dict]) -> Dict[str, Any]:
        """Index documents; each doc must have skill_id + vector + metadata fields."""
        def _actions():
            for doc in documents:
                doc = dict(doc)
                skill_id = doc["skill_id"]
                yield {
                    "_op_type": "index",
                    "_index":   self.index,
                    "_id":      skill_id,
                    "_source":  doc,
                }

        success, errors = helpers.bulk(
            client=self.client,
            actions=_actions(),
            stats_only=False,
            raise_on_error=False,
        )
        error_count = len(errors) if errors else 0
        logger.info("ES bulk indexed %d documents", success)
        if errors:
            logger.error("%d errors during bulk index (first 3: %s)", error_count, errors[:3])
        return {
            "success_count": int(success),
            "error_count": int(error_count),
            "error_samples": errors[:3] if errors else [],
        }

    def _build_mappings(self) -> dict:
        return {
            "dynamic": "strict",
            "properties": {
                "skill_id":        {"type": "keyword"},
                "canonical_label": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "aliases":         {"type": "text"},
                "description":     {"type": "text"},
                "category":        {"type": "keyword"},
                "vector": {
                    "type":       "dense_vector",
                    "dims":       self._vector_dim,
                    "index":      True,
                    "similarity": "cosine",
                },
            },
        }
