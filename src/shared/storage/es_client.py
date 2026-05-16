"""Elasticsearch client for course embeddings – extends shared.db.es_client.

Adds:
- Course embeddings index mapping (dense_vector 1024-dim)
- Alias management for zero-downtime version swaps
- KNN search for course recommendations
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger(__name__)

# Default index/alias names
COURSE_EMBEDDINGS_ALIAS = "course_embeddings"


class CourseElasticsearchClient:
    """Elasticsearch client specialised for course embedding operations."""

    def __init__(
        self,
        hosts: List[str],
        username: str = "",
        password: str = "",
        timeout: int = 30,
        vector_dim: int = 1024,
    ) -> None:
        basic_auth = (username, password) if username else None
        self.client = Elasticsearch(
            hosts=hosts, basic_auth=basic_auth, request_timeout=timeout
        )
        self._vector_dim = vector_dim
        logger.info(
            "CourseElasticsearchClient connected to %s (vector_dim=%d)",
            hosts,
            vector_dim,
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def verify_connection(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_versioned_index(self, version_id: str) -> str:
        """Create a new index for a specific cache version.

        Returns the index name (e.g., ``course_embeddings_v_20260430_001``).
        """
        index_name = f"{COURSE_EMBEDDINGS_ALIAS}_{version_id}"
        if self.client.indices.exists(index=index_name):
            logger.warning("Index %s already exists, deleting first", index_name)
            self.client.indices.delete(index=index_name)

        self.client.indices.create(
            index=index_name,
            mappings=self._build_mappings(),
            settings={
                "index": {
                    "number_of_shards": 2,
                    "number_of_replicas": 1,
                    "knn": True,
                }
            },
        )
        logger.info("Created index %s", index_name)
        return index_name

    def swap_alias(self, new_index: str, old_index: Optional[str] = None) -> None:
        """Atomically swap the read alias to point to a new index.

        If *old_index* is provided, the alias is removed from it.
        """
        actions: List[Dict[str, Any]] = [
            {"add": {"index": new_index, "alias": COURSE_EMBEDDINGS_ALIAS}}
        ]
        if old_index:
            actions.insert(
                0,
                {"remove": {"index": old_index, "alias": COURSE_EMBEDDINGS_ALIAS}},
            )
        self.client.indices.update_aliases(actions=actions)
        logger.info(
            "Alias '%s' swapped: %s -> %s",
            COURSE_EMBEDDINGS_ALIAS,
            old_index or "(none)",
            new_index,
        )

    def get_current_alias_index(self) -> Optional[str]:
        """Return the index currently pointed to by the read alias."""
        try:
            result = self.client.indices.get_alias(name=COURSE_EMBEDDINGS_ALIAS)
            indices = list(result.keys())
            return indices[0] if indices else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def bulk_index_courses(
        self, index_name: str, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Bulk index course documents into a specific index.

        Each document must contain: course_id, title, description, skills,
        embedding (List[float]), version_id.
        """

        def _actions():
            for doc in documents:
                yield {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": doc["course_id"],
                    "_source": doc,
                }

        success, errors = helpers.bulk(
            client=self.client,
            actions=_actions(),
            stats_only=False,
            raise_on_error=False,
        )
        error_count = len(errors) if errors else 0
        logger.info("Bulk indexed %d courses into %s", success, index_name)
        if errors:
            logger.error(
                "%d errors during bulk index (first 3: %s)", error_count, errors[:3]
            )
        return {
            "success_count": int(success),
            "error_count": error_count,
            "error_samples": errors[:3] if errors else [],
        }

    # ------------------------------------------------------------------
    # KNN Search (for recommendations)
    # ------------------------------------------------------------------

    def knn_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        num_candidates: int = 100,
        version_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """KNN cosine similarity search on the course_embeddings alias.

        Returns list of dicts with course metadata + score.
        """
        knn_body: Dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": max(num_candidates, top_k * 5),
        }
        if version_filter:
            knn_body["filter"] = {"term": {"version_id": version_filter}}

        body: Dict[str, Any] = {
            "knn": knn_body,
            "_source": [
                "course_id",
                "title",
                "description",
                "skills",
                "version_id",
            ],
        }
        try:
            response = self.client.search(
                index=COURSE_EMBEDDINGS_ALIAS, body=body
            )
        except Exception as exc:
            logger.error("KNN search failed: %s", exc)
            return []

        results: List[Dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append(
                {
                    "course_id": src.get("course_id"),
                    "title": src.get("title"),
                    "description": src.get("description"),
                    "skills": src.get("skills", []),
                    "version_id": src.get("version_id"),
                    "score": hit["_score"],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_mappings(self) -> Dict[str, Any]:
        return {
            "properties": {
                "course_id": {"type": "keyword"},
                "title": {"type": "text"},
                "description": {"type": "text"},
                "skills": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": self._vector_dim,
                    "index": True,
                    "similarity": "cosine",
                },
                "version_id": {"type": "keyword"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        }
