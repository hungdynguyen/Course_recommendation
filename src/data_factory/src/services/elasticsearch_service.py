from __future__ import annotations

import logging
from typing import Iterable, Optional

from elasticsearch import Elasticsearch, helpers

from ..settings import ElasticsearchConfig

LOGGER = logging.getLogger("data_factory.elasticsearch")


class ElasticsearchService:
    def __init__(self, config: ElasticsearchConfig) -> None:
        self._config = config
        basic_auth: Optional[tuple[str, str]] = None
        if config.username:
            basic_auth = (config.username, config.password)
        self._client = Elasticsearch(hosts=config.hosts, basic_auth=basic_auth)

    def ensure_index(self) -> None:
        index = self._config.index
        exists = self._client.indices.exists(index=index)
        if exists and self._config.recreate_index:
            LOGGER.info("Recreating index %s", index)
            self._client.indices.delete(index=index)
            exists = False

        if not exists:
            LOGGER.info("Creating index %s", index)
            self._client.indices.create(
                index=index,
                mappings=self._build_mappings(),
                settings={
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    }
                },
            )
        else:
            LOGGER.info("Index %s already exists", index)

    def bulk_index(self, documents: Iterable[dict]) -> None:
        LOGGER.info("Indexing documents into %s", self._config.index)
        success, errors = helpers.bulk(
            client=self._client,
            actions=self._prepare_actions(documents),
            stats_only=False,
            raise_on_error=False,
        )
        LOGGER.info("Indexed %s documents", success)
        if errors:
            LOGGER.error("Encountered %s errors during bulk indexing", len(errors))
            for error in errors[:5]:
                LOGGER.error("Error detail: %s", error)
            raise RuntimeError("Bulk indexing completed with errors")

    def _prepare_actions(self, documents: Iterable[dict]) -> Iterable[dict]:
        for doc in documents:
            skill_id = doc["skill_id"]
            vector = doc.pop("vector")
            yield {
                "_op_type": "index",
                "_index": self._config.index,
                "_id": skill_id,
                "_source": {**doc, "vector": vector},
            }

    def _build_mappings(self) -> dict:
        return {
            "dynamic": "strict",
            "properties": {
                "skill_id": {"type": "keyword"},
                "vector": {
                    "type": "dense_vector",
                    "dims": self._config.vector_dim,
                    "index": True,
                    "similarity": "cosine",
                },
            },
        }
