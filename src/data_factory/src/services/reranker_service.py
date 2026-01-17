from __future__ import annotations

import logging
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..settings import RerankerConfig

LOGGER = logging.getLogger("data_factory.reranker")


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device.startswith("cuda:") and torch.cuda.is_available():
        return torch.device(device)
    if device.startswith("mps") and torch.backends.mps.is_available():
        return torch.device(device)
    if device != "cpu":
        LOGGER.warning("Requested device %s unavailable; falling back to CPU", device)
    return torch.device("cpu")


def _resolve_dtype(dtype_name: str | None, device: torch.device) -> torch.dtype:
    if dtype_name:
        torch_dtype = getattr(torch, dtype_name, None)
        if torch_dtype is not None:
            return torch_dtype
        LOGGER.warning("Unknown dtype %s; defaulting based on device", dtype_name)
    if device.type == "cuda":
        return torch.float16
    return torch.float32


class RerankerService:
    def __init__(self, config: RerankerConfig) -> None:
        if not config.enabled:
            raise ValueError("RerankerService initialized while reranker is disabled")
        self._config = config
        self._device = _resolve_device(config.device)
        dtype = _resolve_dtype(config.torch_dtype, self._device)

        LOGGER.info(
            "Loading reranker model %s on %s (dtype=%s)",
            config.model_name,
            self._device,
            dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
        ).to(self._device)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []

        pairs = [(query, candidate.get("text", "")) for candidate in candidates]
        scores: List[float] = []
        batch_size = max(1, self._config.batch_size)

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            query_texts = [pair[0] for pair in batch_pairs]
            doc_texts = [pair[1] for pair in batch_pairs]
            encoded = self._tokenizer(
                query_texts,
                doc_texts,
                padding=True,
                truncation=True,
                max_length=self._config.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**encoded).logits.squeeze(-1)
            scores.extend(logits.detach().cpu().tolist())

        reranked: List[Dict] = []
        for candidate, score in zip(candidates, scores):
            enriched = dict(candidate)
            enriched["rerank_score"] = float(score)
            reranked.append(enriched)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked
