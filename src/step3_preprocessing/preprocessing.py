"""
Step 3: Preprocessing Module — Fix Over-segmentation
------------------------------------------------------
Takes the unidentified regions from Step 1 and merges over-segmented patches
before passing them to the contextual clustering step (Step 4).

Techniques:
  - Region merging by mask IoU / boundary overlap
  - Small-region suppression (area threshold)
  - Morphological operations (dilation, erosion) to clean boundaries
  - Optional SLIC-based superpixel re-grouping
"""

import numpy as np
from typing import List, Tuple
from scipy import ndimage


class Preprocessor:
    """
    Fixes over-segmentation in unidentified regions.
    Outputs a cleaned list of Patch objects for Step 4.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dict containing:
                - min_area (int): Remove patches smaller than this pixel count
                - merge_iou_threshold (float): Merge patches with IoU above this
                - use_morphology (bool): Apply morphological cleanup
                - morph_kernel_size (int): Kernel size for morphological ops
                - merge_max_iterations (int): Max greedy merge passes
        """
        self.config = config

    def process(self, unidentified_patches, image_shape: Tuple[int, int]):
        """
        Process unidentified patches to fix over-segmentation.

        Args:
            unidentified_patches: List[Patch] from Step 1
            image_shape: (H, W) of original image

        Returns:
            List[Patch] — cleaned, merged patches for Step 4
        """
        patches = list(unidentified_patches)

        # Step A: Remove tiny regions
        patches = self._suppress_small_regions(patches)

        # Step B: Morphological cleanup of each mask
        if self.config.get("use_morphology", True):
            patches = self._morphological_cleanup(patches)

        # Step C: Greedy merge of over-segmented adjacent patches
        patches = self._greedy_merge(patches)

        return patches

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _suppress_small_regions(self, patches):
        """Remove patches whose mask area is below min_area threshold."""
        min_area = self.config.get("min_area", 200)
        kept = [p for p in patches if p.area >= min_area]
        removed = len(patches) - len(kept)
        if removed > 0:
            print(f"[Preprocessor] Removed {removed} small region(s) (area < {min_area}px)")
        return kept

    def _morphological_cleanup(self, patches):
        """Apply morphological open/close to smooth mask boundaries."""
        k = self.config.get("morph_kernel_size", 3)
        struct = ndimage.generate_binary_structure(2, 2)
        kernel_iters = max(1, k // 2)

        for patch in patches:
            mask = patch.mask.astype(bool)
            # Opening: erosion then dilation (removes thin protrusions)
            mask = ndimage.binary_erosion(mask, structure=struct, iterations=kernel_iters)
            mask = ndimage.binary_dilation(mask, structure=struct, iterations=kernel_iters)
            # Closing: dilation then erosion (fills small holes)
            mask = ndimage.binary_dilation(mask, structure=struct, iterations=kernel_iters)
            mask = ndimage.binary_erosion(mask, structure=struct, iterations=kernel_iters)
            patch.mask = mask.astype(np.uint8)
            patch.area = int(mask.sum())

        return patches

    def _greedy_merge(self, patches):
        """
        Iteratively merge pairs of patches whose masks have high overlap
        or whose centroids are very close. Stops when no merges occur
        or max iterations reached.
        """
        max_iters = self.config.get("merge_max_iterations", 5)
        threshold = self.config.get("merge_iou_threshold", 0.3)

        for _ in range(max_iters):
            merged_any = False
            merged_flags = [False] * len(patches)
            new_patches = []

            i = 0
            while i < len(patches):
                if merged_flags[i]:
                    i += 1
                    continue
                j = i + 1
                while j < len(patches):
                    if merged_flags[j]:
                        j += 1
                        continue
                    iou = self._mask_iou(patches[i].mask, patches[j].mask)
                    if iou >= threshold:
                        patches[i] = self._merge_two(patches[i], patches[j])
                        merged_flags[j] = True
                        merged_any = True
                    j += 1
                new_patches.append(patches[i])
                i += 1

            patches = new_patches
            if not merged_any:
                break

        return patches

    def _mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """IoU between two binary masks."""
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    def _merge_two(self, patch_a, patch_b):
        """Merge two patches into one by OR-ing their masks."""
        patch_a.mask = np.logical_or(patch_a.mask, patch_b.mask).astype(np.uint8)
        patch_a.area = int(patch_a.mask.sum())
        # Recompute bbox
        ys, xs = np.where(patch_a.mask)
        if len(xs):
            patch_a.bbox = (int(xs.min()), int(ys.min()),
                            int(xs.max() - xs.min()), int(ys.max() - ys.min()))
        return patch_a
