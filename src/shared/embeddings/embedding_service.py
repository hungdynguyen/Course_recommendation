"""Unified EmbeddingService – dùng chung cho cả data_factory lẫn service_api.

data_factory: encode batch lớn khi build KG.
service_api:  encode query nhỏ khi serving (gap detection, skill search).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wrapper cho SentenceTransformer, lazy-load model một lần."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        model_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        source = model_path if model_path else model_name
        logger.info("Loading embedding model: %s  device=%s", source, device)
        self._model = SentenceTransformer(model_name_or_path=source, device=device)
        self._batch_size = batch_size
        self._normalize = normalize
        logger.info("Embedding model loaded  dim=%d", self.dim)

    # ---------- Public API ----------

    def encode(self, texts: Iterable[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Encode danh sách text → ma trận (N, D) float32."""
        sentences = list(texts)
        if not sentences:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            sentences,
            batch_size=batch_size or self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=len(sentences) > 200,
        )
        return np.array(vecs, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Tiện ích encode một chuỗi đơn → vector 1-D."""
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())
