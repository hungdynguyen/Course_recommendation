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

        # Log GPU information
        if torch.cuda.is_available():
            LOGGER.info("CUDA available: %d GPU(s)", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                LOGGER.info("  GPU %d: %s (%.2f GB)", i, gpu_name, gpu_memory)
            if self._device.type == "cuda":
                current_device = self._device.index if self._device.index is not None else 0
                LOGGER.info("Using GPU %d: %s", current_device, torch.cuda.get_device_name(current_device))
        else:
            LOGGER.warning("CUDA not available, using CPU")

        LOGGER.info(
            "Loading reranker model %s on %s (dtype=%s)",
            config.model_name,
            self._device,
            dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Set padding token if not defined
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            LOGGER.info("Set pad_token to eos_token for tokenizer")
        
        self._model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
        ).to(self._device)
        
        # Also set pad_token_id for model config
        if self._model.config.pad_token_id is None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id
            LOGGER.info("Set pad_token_id=%s for model config", self._tokenizer.pad_token_id)
        
        # Log GPU memory after loading model
        if torch.cuda.is_available() and self._device.type == "cuda":
            current_device = self._device.index if self._device.index is not None else 0
            allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            LOGGER.info("GPU memory: %.2f GB allocated, %.2f GB reserved", allocated, reserved)

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
            
            # Log GPU utilization for first batch
            if start == 0 and torch.cuda.is_available() and self._device.type == "cuda":
                current_device = self._device.index if self._device.index is not None else 0
                allocated = torch.cuda.memory_allocated(current_device) / 1024**3
                LOGGER.debug("GPU memory before inference: %.2f GB allocated", allocated)
            
            with torch.no_grad():
                logits = self._model(**encoded).logits.squeeze(-1)
            
            # Handle both single and batch predictions
            batch_scores = logits.detach().cpu()
            if batch_scores.dim() == 0:  # Single prediction
                scores.append(float(batch_scores.item()))
            else:  # Batch predictions
                scores.extend(batch_scores.tolist())

        reranked: List[Dict] = []
        for candidate, score in zip(candidates, scores):
            enriched = dict(candidate)
            enriched["rerank_score"] = float(score) if isinstance(score, (int, float)) else float(score[0])
            reranked.append(enriched)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked
