"""SkillSearchService – tìm canonical skills trong KG bằng vector similarity.

Luồng:
  raw skill name (text) → embed → kNN search ES → canonical skill_ids
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from shared.db.es_client import ElasticsearchClient
from shared.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SkillSearchService:

    def __init__(self, es: ElasticsearchClient, embedding: EmbeddingService) -> None:
        self._es        = es
        self._embedding = embedding

    def search_by_text(self, skill_name: str, limit: int = 5) -> List[Dict]:
        """
        Tìm canonical skills tương đồng với *skill_name*.

        Returns list of dicts:
            skill_id, canonical_label, aliases, description, category, score
        """
        vec = self._embedding.encode_single(skill_name).tolist()
        results = self._es.hybrid_search(
            query_text=skill_name,
            vector=vec,
            limit=limit,
        )
        logger.debug("Search '%s' → %d results", skill_name, len(results))
        return results

    def search_batch(
        self,
        skill_names: List[str],
        limit_per_skill: int = 3,
        min_score: float = 0.0,
    ) -> Dict[str, List[Dict]]:
        """
        Batch tìm canonical skills cho nhiều skill names.

        Returns {skill_name: [canonical_skill_dicts]}
        """
        if not skill_names:
            return {}

        # Encode tất cả một lần để tận dụng batch
        vecs = self._embedding.encode(skill_names)  # (N, D)
        result: Dict[str, List[Dict]] = {}
        for name, vec in zip(skill_names, vecs):
            hits = self._es.hybrid_search(
                query_text=name,
                vector=vec.tolist(),
                limit=limit_per_skill,
                min_score=min_score,
            )
            result[name] = hits
        return result

    def get_skill_by_id(self, skill_id: str) -> Optional[Dict]:
        """Fetch một canonical skill theo ID (exact match)."""
        try:
            resp = self._es.client.get(index=self._es.index, id=skill_id)
            src = resp["_source"]
            return {
                "skill_id":        src.get("skill_id"),
                "canonical_label": src.get("canonical_label"),
                "aliases":         src.get("aliases", []),
                "description":     src.get("description", ""),
                "category":        src.get("category", ""),
            }
        except Exception:
            return None
