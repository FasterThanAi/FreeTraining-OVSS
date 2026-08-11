"""
Step 4: Contextual Clustering — Label Unidentified Patches
-----------------------------------------------------------
Labels unidentified (preprocessed) patches by combining:
  1. Visual similarity to identified patches (embedding cosine distance)
  2. Spatial context from the Co-occurrence Matrix M (Step 2)
  3. Neighbor label voting from adjacent identified patches

This module assigns a class label to every previously unlabeled patch.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ClusteringResult:
    """Output of the contextual clustering step."""
    labeled_patches: list         # List[Patch] with .label set
    label_confidence: Dict[int, float]   # patch_id → confidence score
    fallback_label: str = "background"


class ContextualClusterer:
    """
    Labels unidentified patches using:
      - Embedding similarity to identified patches
      - Co-occurrence matrix context (what classes co-occur nearby?)
      - Spatial neighbor voting
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dict containing:
                - embedding_weight (float): Weight for visual similarity score
                - cooccurrence_weight (float): Weight for co-occurrence score
                - neighbor_weight (float): Weight for spatial neighbor vote
                - fallback_label (str): Label if all scores are below threshold
                - min_label_confidence (float): Min confidence to accept a label
                - k_neighbors (int): Number of nearest identified patches to consider
        """
        self.config = config

    def label(
        self,
        unidentified_patches,      # List[Patch] from Step 3
        identified_patches,        # List[Patch] from Step 1/2
        cooccurrence_result,       # CooccurrenceResult from Step 2
    ) -> ClusteringResult:
        """
        Assign labels to all unidentified patches.

        Args:
            unidentified_patches: Preprocessed unlabeled patches
            identified_patches: Labeled patches with embeddings
            cooccurrence_result: M matrix and INFO from Step 2

        Returns:
            ClusteringResult with all patches labeled
        """
        labeled = []
        label_confidence = {}
        fallback = self.config.get("fallback_label", "background")

        for uid, patch in enumerate(unidentified_patches):
            label, confidence = self._score_patch(
                patch, identified_patches, cooccurrence_result
            )
            patch.label = label
            patch.confidence = confidence
            labeled.append(patch)
            label_confidence[uid] = confidence

        return ClusteringResult(
            labeled_patches=labeled,
            label_confidence=label_confidence,
            fallback_label=fallback,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _score_patch(self, patch, identified_patches, cooc_result):
        """
        Compute a combined score for each candidate class and pick best.

        Returns:
            (best_label: str, best_confidence: float)
        """
        class_names = cooc_result.class_names
        if not class_names:
            return self.config.get("fallback_label", "background"), 0.0

        w_emb = self.config.get("embedding_weight", 0.5)
        w_coc = self.config.get("cooccurrence_weight", 0.3)
        w_nbr = self.config.get("neighbor_weight", 0.2)

        scores = np.zeros(len(class_names), dtype=np.float32)

        # --- Embedding similarity ---
        if patch.embedding is not None:
            emb_scores = self._embedding_score(patch, identified_patches, class_names)
            scores += w_emb * emb_scores

        # --- Co-occurrence context ---
        coc_scores = self._cooccurrence_score(patch, identified_patches, cooc_result)
        scores += w_coc * coc_scores

        # --- Neighbor vote ---
        nbr_scores = self._neighbor_vote_score(patch, identified_patches, cooc_result)
        scores += w_nbr * nbr_scores

        best_idx = int(np.argmax(scores))
        best_confidence = float(scores[best_idx])
        min_conf = self.config.get("min_label_confidence", 0.05)

        if best_confidence < min_conf:
            label = self.config.get("fallback_label", "background")
        else:
            label = class_names[best_idx]

        return label, best_confidence

    def _embedding_score(self, patch, identified_patches, class_names):
        """
        Cosine similarity of patch embedding to mean class embeddings.
        Returns array of shape (n_classes,).
        """
        scores = np.zeros(len(class_names))
        class_embs = {c: [] for c in class_names}

        for p in identified_patches:
            if p.label in class_embs and p.embedding is not None:
                class_embs[p.label].append(p.embedding.flatten())

        for i, c in enumerate(class_names):
            if class_embs[c]:
                mean_emb = np.mean(class_embs[c], axis=0)
                # Cosine similarity
                norm_a = np.linalg.norm(patch.embedding.flatten())
                norm_b = np.linalg.norm(mean_emb)
                if norm_a > 0 and norm_b > 0:
                    scores[i] = float(
                        np.dot(patch.embedding.flatten(), mean_emb) / (norm_a * norm_b)
                    )

        return scores

    def _cooccurrence_score(self, patch, identified_patches, cooc_result):
        """
        Weight class scores using co-occurrence with nearby identified patches.
        Returns array of shape (n_classes,).
        """
        M = cooc_result.M_normalized
        class_to_idx = cooc_result.class_to_idx
        class_names = cooc_result.class_names
        scores = np.zeros(len(class_names))

        k = self.config.get("k_neighbors", 5)
        neighbors = self._find_k_nearest(patch, identified_patches, k)

        for nbr in neighbors:
            if nbr.label in class_to_idx:
                nbr_idx = class_to_idx[nbr.label]
                # Add the co-occurrence row of the neighbor's class
                scores += M[nbr_idx]

        if scores.sum() > 0:
            scores /= scores.sum()

        return scores

    def _neighbor_vote_score(self, patch, identified_patches, cooc_result):
        """
        Simple majority vote from k nearest identified patches.
        Returns array of shape (n_classes,).
        """
        class_to_idx = cooc_result.class_to_idx
        class_names = cooc_result.class_names
        scores = np.zeros(len(class_names))

        k = self.config.get("k_neighbors", 5)
        neighbors = self._find_k_nearest(patch, identified_patches, k)

        for nbr in neighbors:
            if nbr.label in class_to_idx:
                scores[class_to_idx[nbr.label]] += 1

        if scores.sum() > 0:
            scores /= scores.sum()

        return scores

    def _find_k_nearest(self, patch, identified_patches, k: int):
        """Find k nearest identified patches by centroid Euclidean distance."""
        cx, cy = self._centroid(patch.mask)
        dists = []
        for p in identified_patches:
            pcx, pcy = self._centroid(p.mask)
            d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            dists.append((d, p))
        dists.sort(key=lambda x: x[0])
        return [p for _, p in dists[:k]]

    def _centroid(self, mask: np.ndarray):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (0, 0)
        return (int(xs.mean()), int(ys.mean()))
