"""
Step 5: Final Mask Fusion
--------------------------
Merges the labeled identified patches (from Step 1/2) with the contextually
labeled unidentified patches (from Step 4) into a single complete semantic
segmentation mask.

Output: an H x W integer label map where each pixel value is a class index.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class MaskFusion:
    """
    Fuses identified and newly-labeled unidentified patches into a
    complete labeled semantic mask.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dict containing:
                - conflict_resolution: 'confidence' | 'area' | 'identified_priority'
                - background_label: str — label assigned to unassigned pixels
                - smooth_boundaries: bool — apply boundary smoothing post-fusion
        """
        self.config = config

    def fuse(
        self,
        identified_patches,       # List[Patch] from Step 1/2
        labeled_unidentified,     # List[Patch] from Step 4
        class_names: List[str],
        image_shape: Tuple[int, int],
    ) -> Dict:
        """
        Produce a complete labeled mask.

        Args:
            identified_patches: Patches with original SAM labels
            labeled_unidentified: Previously unknown patches, now labeled
            class_names: List of class name strings
            image_shape: (H, W) of the original image

        Returns:
            Dict with:
              - label_map: np.ndarray (H x W) of integer class indices
              - class_names: list mapping index → class name
              - confidence_map: np.ndarray (H x W) of float confidence
              - color_mask: np.ndarray (H x W x 3) RGB visualization
        """
        H, W = image_shape
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        background_label = self.config.get("background_label", "background")

        # Initialize maps
        label_map = np.full((H, W), fill_value=-1, dtype=np.int32)
        confidence_map = np.zeros((H, W), dtype=np.float32)

        conflict = self.config.get("conflict_resolution", "confidence")

        all_patches = []
        if conflict == "identified_priority":
            # Process unidentified first, then identified (so identified wins)
            all_patches = list(labeled_unidentified) + list(identified_patches)
        else:
            all_patches = list(identified_patches) + list(labeled_unidentified)

        for patch in all_patches:
            if patch.label is None:
                continue
            label_idx = class_to_idx.get(patch.label, -1)
            if label_idx < 0:
                continue
            mask_bool = patch.mask.astype(bool)

            if conflict == "confidence":
                # Write only where this patch has higher confidence
                update_region = mask_bool & (patch.confidence > confidence_map)
                label_map[update_region] = label_idx
                confidence_map[update_region] = patch.confidence
            else:
                # Overwrite (last or first wins)
                label_map[mask_bool] = label_idx
                confidence_map[mask_bool] = patch.confidence

        # Fill remaining unassigned pixels with background
        bg_idx = class_to_idx.get(background_label, len(class_names))
        label_map[label_map == -1] = bg_idx

        # Optional boundary smoothing
        if self.config.get("smooth_boundaries", False):
            label_map = self._smooth_boundaries(label_map)

        # Build color visualization
        color_mask = self._colorize(label_map, len(class_names) + 1)

        return {
            "label_map": label_map,
            "class_names": class_names + [background_label],
            "confidence_map": confidence_map,
            "color_mask": color_mask,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _smooth_boundaries(self, label_map: np.ndarray) -> np.ndarray:
        """
        Apply median filter to smooth jagged boundaries between class regions.
        """
        from scipy.ndimage import median_filter
        return median_filter(label_map, size=3).astype(np.int32)

    def _colorize(self, label_map: np.ndarray, n_classes: int) -> np.ndarray:
        """Map an integer label map to an RGB color image."""
        np.random.seed(42)
        palette = np.random.randint(50, 255, size=(n_classes + 1, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]  # background = black

        H, W = label_map.shape
        color = np.zeros((H, W, 3), dtype=np.uint8)
        for cls_idx in range(n_classes + 1):
            color[label_map == cls_idx] = palette[cls_idx]
        return color
