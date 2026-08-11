"""
Step 2: Co-occurrence Matrix Builder
-------------------------------------
Constructs the co-occurrence matrix M and patch INFO dictionary from
identified patches produced by Step 1.

Co-occurrence Matrix M[i][j] = frequency that class i appears
spatially adjacent or co-located with class j across the image.

INFO dict stores per-patch feature embeddings, labels, and spatial stats
to guide the contextual clustering in Step 4.
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class PatchInfo:
    """Rich feature record for a single patch, stored in the INFO dict."""
    patch_id: int
    label: str
    confidence: float
    centroid: tuple          # (cx, cy) in pixel coords
    area: int
    embedding: Optional[np.ndarray] = None
    neighbor_ids: List[int] = field(default_factory=list)
    neighbor_labels: List[str] = field(default_factory=list)


@dataclass
class CooccurrenceResult:
    """Output container for Step 2."""
    M: np.ndarray               # Co-occurrence matrix (n_classes x n_classes)
    M_normalized: np.ndarray    # Row-normalized version of M
    class_names: List[str]      # Index → class name mapping
    class_to_idx: Dict[str, int]
    info: Dict[int, PatchInfo]  # patch_id → PatchInfo


class CooccurrenceBuilder:
    """
    Builds the co-occurrence matrix M and the INFO dictionary from
    identified patches output by SAMPass (Step 1).
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dict containing:
                - adjacency_method: 'centroid_distance' | 'mask_overlap' | 'bbox_iou'
                - adjacency_threshold: Distance in pixels (for centroid method)
                - normalize: bool — whether to row-normalize M
        """
        self.config = config

    def build(
        self,
        identified_patches,         # List[Patch] from Step 1
        class_names: List[str],
        image_shape: tuple,
    ) -> CooccurrenceResult:
        """
        Build M and INFO from identified patches.

        Args:
            identified_patches: List of Patch objects (must have .label set)
            class_names: Full list of class names (for matrix indexing)
            image_shape: (H, W) of original image

        Returns:
            CooccurrenceResult
        """
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        n_classes = len(class_names)
        M = np.zeros((n_classes, n_classes), dtype=np.float32)
        info: Dict[int, PatchInfo] = {}

        # Build PatchInfo records
        for pid, patch in enumerate(identified_patches):
            cx, cy = self._centroid(patch.mask)
            info[pid] = PatchInfo(
                patch_id=pid,
                label=patch.label,
                confidence=patch.confidence,
                centroid=(cx, cy),
                area=patch.area,
                embedding=patch.embedding if hasattr(patch, "embedding") else None,
            )

        # Compute adjacency and fill M
        adjacency_method = self.config.get("adjacency_method", "centroid_distance")
        threshold = self.config.get("adjacency_threshold", 100)

        for i, pi in info.items():
            for j, pj in info.items():
                if i >= j:
                    continue
                if not (pi.label in class_to_idx and pj.label in class_to_idx):
                    continue

                adjacent = False
                if adjacency_method == "centroid_distance":
                    dist = np.sqrt(
                        (pi.centroid[0] - pj.centroid[0]) ** 2
                        + (pi.centroid[1] - pj.centroid[1]) ** 2
                    )
                    adjacent = dist < threshold
                elif adjacency_method == "mask_overlap":
                    adjacent = self._masks_overlap(
                        identified_patches[i].mask,
                        identified_patches[j].mask,
                    )
                elif adjacency_method == "bbox_iou":
                    adjacent = self._bbox_iou(
                        identified_patches[i].bbox,
                        identified_patches[j].bbox,
                    ) > 0.0

                if adjacent:
                    ri = class_to_idx[pi.label]
                    ci = class_to_idx[pj.label]
                    M[ri, ci] += 1
                    M[ci, ri] += 1
                    info[i].neighbor_ids.append(j)
                    info[i].neighbor_labels.append(pj.label)
                    info[j].neighbor_ids.append(i)
                    info[j].neighbor_labels.append(pi.label)

        # Normalize
        M_norm = self._row_normalize(M) if self.config.get("normalize", True) else M.copy()

        return CooccurrenceResult(
            M=M,
            M_normalized=M_norm,
            class_names=class_names,
            class_to_idx=class_to_idx,
            info=info,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _centroid(self, mask: np.ndarray):
        """Compute (cx, cy) centroid of a binary mask."""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (0, 0)
        return (int(xs.mean()), int(ys.mean()))

    def _masks_overlap(self, mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
        """Check if two binary masks share at least 1 pixel."""
        return bool(np.any(mask_a & mask_b))

    def _bbox_iou(self, bbox_a, bbox_b) -> float:
        """Compute IoU of two (x, y, w, h) bounding boxes."""
        ax1, ay1, aw, ah = bbox_a
        bx1, by1, bw, bh = bbox_b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _row_normalize(self, M: np.ndarray) -> np.ndarray:
        """Row-normalize matrix so each row sums to 1 (or 0 if all-zero)."""
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero
        return M / row_sums
