"""SkillDedupService – phát hiện và gộp các skill trùng lặp dựa trên fuzzy matching + embedding.

Thuật toán 2-pass:
    1. Fuzzy Match Pass (token_set_ratio >= fuzzy_threshold): bắt typos/format variants
    2. Embedding Pass (cosine similarity >= embedding_threshold): bắt semantic variants

Service giữ lại metrics + merge details để pipeline xuất JSON report sau build.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz

from shared.models.skill import CanonicalSkill
from data_factory.settings import DeduplicationConfig

logger = logging.getLogger("data_factory.skill_dedup")


@dataclass
class MergeRecord:
    """Record của một merge operation."""
    canonical_label: str
    merged_from: List[str]
    method: str  # "fuzzy_matching" | "embedding_similarity"
    threshold: float
    reason: Optional[str] = None


class SkillDedupService:

    def __init__(self, config: DeduplicationConfig) -> None:
        self._fuzzy_threshold = config.fuzzy_threshold
        self._embedding_threshold = config.embedding_threshold
        self._bs = config.batch_size
        self._strategy = config.canonical_strategy
        self._merge_records: List[MergeRecord] = []
        self._last_metrics: Dict[str, Any] = {}
        logger.info(
            "SkillDedupService – fuzzy=%.3f  embedding=%.3f  strategy=%s",
            self._fuzzy_threshold, self._embedding_threshold, self._strategy,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        labels: List[str],
        embeddings: np.ndarray,
        descriptions: Optional[List[Optional[str]]] = None,
        categories: Optional[List[Optional[str]]] = None,
    ) -> Tuple[List[CanonicalSkill], Dict[str, str]]:
        """
        Gộp các skill trùng lặp qua 2 pass.

        Parameters
        ----------
        labels       : tên raw của mỗi skill (từ LLM)
        embeddings   : ma trận (N, D) tương ứng
        descriptions : mô tả (optional, cùng thứ tự)
        categories   : danh mục (optional, cùng thứ tự)

        Returns
        -------
        canonical_skills : các CanonicalSkill sau dedup
        merge_map        : {raw_label -> canonical_skill_id}
        """
        n = len(labels)
        if n == 0:
            return [], {}
        if embeddings.shape[0] != n:
            raise ValueError(f"labels ({n}) và embeddings ({embeddings.shape[0]}) không khớp")

        descs = descriptions or [None] * n
        cats  = categories or [None] * n

        # Reset merge history
        self._merge_records = []

        # Pass 1: Fuzzy matching
        logger.info("=== Pass 1: Fuzzy Matching ===")
        fuzzy_clusters = self._fuzzy_match_pass(labels)
        logger.info("Fuzzy: %d raw → %d clusters (threshold=%.2f)", n, len(fuzzy_clusters), self._fuzzy_threshold)

        # Pass 2: Embedding similarity trên những skill còn lại
        logger.info("=== Pass 2: Embedding Similarity ===")
        embedding_clusters = self._embedding_match_pass(
            fuzzy_clusters, labels, embeddings
        )
        logger.info(
            "Embedding: %d clusters → %d final (threshold=%.3f)",
            len(fuzzy_clusters), len(embedding_clusters), self._embedding_threshold,
        )

        # Build canonical skills từ final clusters
        canonical_skills = self._build_canonical_skills(
            embedding_clusters, labels, descs, cats
        )
        
        # Build merge map
        merge_map = {}
        for cs in canonical_skills:
            # Aliases là các label khác trong cluster
            for alias in cs.aliases:
                merge_map[alias] = cs.skill_id
            # Canonical label cũng vào map
            merge_map[cs.canonical_label] = cs.skill_id

        # Effective removed count should match raw - canonical.
        # alias_entries_count keeps the old alias-based view for diagnostics.
        effective_removed_count = n - len(canonical_skills)
        alias_entries_count = sum(len(cs.aliases) for cs in canonical_skills)
        cluster_sizes = [1 + len(cs.aliases) for cs in canonical_skills]
        size_distribution = {
            "1": sum(1 for s in cluster_sizes if s == 1),
            "2_3": sum(1 for s in cluster_sizes if 2 <= s <= 3),
            "4_10": sum(1 for s in cluster_sizes if 4 <= s <= 10),
            "gt_10": sum(1 for s in cluster_sizes if s > 10),
        }
        self._last_metrics = {
            "raw_count": n,
            "canonical_count": len(canonical_skills),
            "removed_count": effective_removed_count,
            "alias_entries_count": alias_entries_count,
            "reduction_percent": round(100.0 * (1 - len(canonical_skills) / n), 4),
            "fuzzy_merge_groups": sum(1 for r in self._merge_records if r.method == "fuzzy_matching"),
            "embedding_merge_groups": sum(1 for r in self._merge_records if r.method == "embedding_similarity"),
            "fuzzy_removed_count": sum(len(r.merged_from) for r in self._merge_records if r.method == "fuzzy_matching"),
            "embedding_removed_count": sum(len(r.merged_from) for r in self._merge_records if r.method == "embedding_similarity"),
            "cluster_size_distribution": size_distribution,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "avg_aliases_per_canonical": round(alias_entries_count / len(canonical_skills), 4) if canonical_skills else 0.0,
        }
        logger.info(
            "=== Dedup hoàn tất: %d raw → %d canonical (giảm %.1f%%) ===",
            n, len(canonical_skills), 100.0 * (1 - len(canonical_skills) / n),
        )
        
        return canonical_skills, merge_map

    def get_merge_report(self) -> Dict:
        """Trả về report chi tiết về các merge được thực hiện."""
        fuzzy_merges = [r for r in self._merge_records if r.method == "fuzzy_matching"]
        embedding_merges = [r for r in self._merge_records if r.method == "embedding_similarity"]

        return {
            "fuzzy_matched": [
                {
                    "canonical": r.canonical_label,
                    "merged_from": r.merged_from,
                    "method": r.method,
                    "threshold": r.threshold,
                    "reason": r.reason,
                }
                for r in fuzzy_merges
            ],
            "embedding_matched": [
                {
                    "canonical": r.canonical_label,
                    "merged_from": r.merged_from,
                    "method": r.method,
                    "threshold": r.threshold,
                    "reason": r.reason,
                }
                for r in embedding_merges
            ],
            "summary": {
                "total_fuzzy_clusters": len(fuzzy_merges),
                "total_embedding_clusters": len(embedding_merges),
                "total_merge_groups": len(self._merge_records),
                "total_removed_skills": sum(len(r.merged_from) for r in self._merge_records),
            },
            "metrics": self._last_metrics,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Trả về metrics tổng quan của lần dedup gần nhất."""
        return dict(self._last_metrics)

    # ------------------------------------------------------------------
    # Private: Fuzzy Matching Pass
    # ------------------------------------------------------------------

    def _fuzzy_match_pass(self, labels: List[str]) -> List[List[int]]:
        """
        Pass 1: Fuzzy string matching dùng token_set_ratio.
        Trả về list of clusters (mỗi cluster là list of indices).
        """
        n = len(labels)
        assigned: List[int] = [-1] * n
        clusters: List[List[int]] = []

        for i in range(n):
            if assigned[i] != -1:
                continue
            
            cid = len(clusters)
            clusters.append([i])
            assigned[i] = cid

            label_i = labels[i]

            # So sánh với tất cả labels chưa được gán
            for j in range(i + 1, n):
                if assigned[j] != -1:
                    continue
                
                label_j = labels[j]
                similarity = fuzz.token_set_ratio(label_i, label_j) / 100.0
                
                if similarity >= self._fuzzy_threshold:
                    assigned[j] = cid
                    clusters[cid].append(j)
                    
                    # Log merge
                    if len(clusters[cid]) == 2:  # Lần đầu tiên merge trong cluster này
                        self._merge_records.append(MergeRecord(
                            canonical_label=label_i,
                            merged_from=[label_j],
                            method="fuzzy_matching",
                            threshold=self._fuzzy_threshold,
                            reason=f"token_set_ratio: {similarity:.3f}",
                        ))
                    else:
                        # Cập nhật existing merge record
                        for mr in self._merge_records:
                            if mr.canonical_label == label_i and mr.method == "fuzzy_matching":
                                mr.merged_from.append(label_j)
                                break

        logger.debug("Fuzzy match: %d clusters tạo ra", len(clusters))
        return clusters

    # ------------------------------------------------------------------
    # Private: Embedding Similarity Pass
    # ------------------------------------------------------------------

    def _embedding_match_pass(
        self,
        fuzzy_clusters: List[List[int]],
        labels: List[str],
        embeddings: np.ndarray,
    ) -> List[List[int]]:
        """
        Pass 2: Embedding similarity trên những skill chưa được merge hoàn toàn.
        
        Đưa vào: fuzzy clusters (từ pass 1)
        Đưa ra: final clusters sau embedding matching
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.where(norms == 0, 1.0, norms)

        # Pass embedding matching trong mỗi fuzzy cluster
        final_clusters: List[List[int]] = []

        for fuzzy_cluster in fuzzy_clusters:
            if not fuzzy_cluster:
                continue

            # Embedding matching chỉ trong fuzzy cluster này
            local_assigned: List[int] = [-1] * len(fuzzy_cluster)
            local_clusters: List[List[int]] = []

            for local_i, global_i in enumerate(fuzzy_cluster):
                if local_assigned[local_i] != -1:
                    continue

                local_cid = len(local_clusters)
                local_clusters.append([global_i])
                local_assigned[local_i] = local_cid

                # So sánh embedding với các index còn lại trong fuzzy cluster
                for local_j in range(local_i + 1, len(fuzzy_cluster)):
                    if local_assigned[local_j] != -1:
                        continue

                    global_j = fuzzy_cluster[local_j]
                    similarity = normed[global_i] @ normed[global_j]

                    if similarity >= self._embedding_threshold:
                        local_assigned[local_j] = local_cid
                        local_clusters[local_cid].append(global_j)

                        # Log merge
                        if len(local_clusters[local_cid]) == 2:
                            self._merge_records.append(MergeRecord(
                                canonical_label=labels[global_i],
                                merged_from=[labels[global_j]],
                                method="embedding_similarity",
                                threshold=self._embedding_threshold,
                                reason=f"Cosine similarity: {similarity:.3f}",
                            ))
                        else:
                            for mr in self._merge_records:
                                if mr.canonical_label == labels[global_i] and mr.method == "embedding_similarity":
                                    mr.merged_from.append(labels[global_j])
                                    break

            # Thêm local clusters vào final clusters
            for cluster in local_clusters:
                final_clusters.append(cluster)

        return final_clusters

    # ------------------------------------------------------------------
    # Private: Build Canonical Skills
    # ------------------------------------------------------------------

    def _build_canonical_skills(
        self,
        clusters: List[List[int]],
        labels: List[str],
        descs: List[Optional[str]],
        cats: List[Optional[str]],
    ) -> List[CanonicalSkill]:
        """Tạo CanonicalSkill từ final clusters."""
        canonical_skills: List[CanonicalSkill] = []

        for cluster in clusters:
            cluster_labels = [labels[i] for i in cluster]
            canonical = self._pick_canonical(cluster_labels)
            sid = _label_to_id(canonical)

            rep = cluster[0]
            cs = CanonicalSkill(
                skill_id=sid,
                canonical_label=canonical,
                aliases=[lb for lb in cluster_labels if lb != canonical],
                description=descs[rep],
                category=cats[rep],
            )
            canonical_skills.append(cs)

        return canonical_skills

    # ------------------------------------------------------------------
    # Private: Helpers
    # ------------------------------------------------------------------

    def _pick_canonical(self, labels: List[str]) -> str:
        if self._strategy == "longest":
            return max(labels, key=len)
        if self._strategy == "most_common":
            from collections import Counter
            return Counter(labels).most_common(1)[0][0]
        return labels[0]  # "first"


def _label_to_id(label: str) -> str:
    return hashlib.sha256(label.lower().strip().encode()).hexdigest()[:16]

