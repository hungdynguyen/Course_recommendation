"""GapDetectionService – tìm skill gaps giữa JD và CV.

Chiến thuật:
  Với mỗi skill trong JD, tính max cosine similarity với tất cả skills trong CV.
  Nếu max similarity < threshold → đây là một skill gap.
  
Vì cả JD skills và CV skills đều là text tự do (không chuẩn hóa), việc so sánh
dựa trên embedding thay vì exact match để bắt được các biến thể khác nhau.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from shared.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class GapDetectionService:

    def __init__(self, embedding: EmbeddingService, threshold: float = 0.80) -> None:
        """
        Parameters
        ----------
        embedding  : EmbeddingService dùng chung (đã lazy-load)
        threshold  : nếu max(sim(jd_skill, cv_skills)) < threshold → gap
                     Giá trị hợp lý: 0.75 – 0.85
        """
        self._embedding = embedding
        self._threshold = threshold
        logger.info("GapDetectionService  threshold=%.2f", threshold)

    def find_gaps_with_details(
        self,
        jd_skills: List[str],
        cv_skills: List[str],
        threshold: float | None = None,
    ) -> Tuple[List[str], List[Dict[str, object]]]:
        """Find gaps and return per-JD-skill matching trace details.

        Returns
        -------
        (gaps, details)
          gaps: List of missing JD skills.
          details: Per JD skill detail dict with keys:
            - jd_skill: JD skill text
            - best_cv_skill: best matched CV skill text (empty if CV is empty)
            - best_similarity: max cosine similarity in [0, 1]
            - is_gap: whether the skill is classified as a gap
        """
        effective_threshold = self._threshold if threshold is None else threshold

        if not jd_skills:
            return [], []

        # Nếu CV trống → mọi JD skill đều là gap
        if not cv_skills:
            logger.info("CV skills trống → %d gaps", len(jd_skills))
            details = [
                {
                    "jd_skill": skill,
                    "best_cv_skill": "",
                    "best_similarity": 0.0,
                    "is_gap": True,
                }
                for skill in jd_skills
            ]
            return list(jd_skills), details

        jd_vecs = self._embedding.encode(jd_skills)   # (M, D)
        cv_vecs = self._embedding.encode(cv_skills)   # (K, D)

        # cosine sim matrix (M, K) – vectors đã normalize từ EmbeddingService
        sim_matrix = jd_vecs @ cv_vecs.T              # (M, K)
        max_indices = sim_matrix.argmax(axis=1)       # (M,)
        max_sims = sim_matrix.max(axis=1)             # (M,)

        details: List[Dict[str, object]] = []
        gaps: List[str] = []
        for i, (skill, sim) in enumerate(zip(jd_skills, max_sims)):
            best_idx = int(max_indices[i])
            best_cv_skill = cv_skills[best_idx] if 0 <= best_idx < len(cv_skills) else ""
            is_gap = bool(sim < effective_threshold)
            if is_gap:
                gaps.append(skill)
            details.append(
                {
                    "jd_skill": skill,
                    "best_cv_skill": best_cv_skill,
                    "best_similarity": float(sim),
                    "is_gap": is_gap,
                }
            )

        logger.info(
            "Gap detection: %d JD skills, %d CV skills → %d gaps (threshold=%.2f)",
            len(jd_skills), len(cv_skills), len(gaps), effective_threshold,
        )
        return gaps, details

    def find_gaps(
        self,
        jd_skills: List[str],
        cv_skills: List[str],
        threshold: float | None = None,
    ) -> List[str]:
        """
        Tìm danh sách skill trong JD mà CV không có.

        Parameters
        ----------
        jd_skills : skills yêu cầu từ JD (đã trích xuất bằng LLM)
        cv_skills : skills ứng viên có từ CV (đã trích xuất bằng LLM)

        Returns
        -------
        Danh sách skill gap (subset của jd_skills)
        """
        gaps, _details = self.find_gaps_with_details(
            jd_skills=jd_skills,
            cv_skills=cv_skills,
            threshold=threshold,
        )
        return gaps
