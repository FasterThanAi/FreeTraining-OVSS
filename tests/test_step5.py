"""Tests for Step 5: Final Mask Fusion."""

import numpy as np
import pytest
from src.step5_fusion.mask_fusion import MaskFusion
from src.step1_sam_pass.sam_pass import Patch


def make_patch(label, y1, y2, x1, x2, confidence=0.8):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return Patch(mask=mask, bbox=(x1, y1, x2-x1, y2-y1), area=(y2-y1)*(x2-x1), confidence=confidence, label=label)


def test_label_map_shape():
    fuser = MaskFusion(config={"conflict_resolution": "confidence", "smooth_boundaries": False})
    class_names = ["sky", "road"]
    identified = [make_patch("sky", 0, 50, 0, 100)]
    unlabeled = [make_patch("road", 50, 100, 0, 100)]
    result = fuser.fuse(identified, unlabeled, class_names, image_shape=(100, 100))
    assert result["label_map"].shape == (100, 100)


def test_all_pixels_assigned():
    fuser = MaskFusion(config={"conflict_resolution": "confidence", "smooth_boundaries": False, "background_label": "background"})
    class_names = ["sky"]
    identified = [make_patch("sky", 0, 100, 0, 100)]
    result = fuser.fuse(identified, [], class_names, image_shape=(100, 100))
    # No pixel should be -1
    assert (result["label_map"] == -1).sum() == 0


def test_color_mask_shape():
    fuser = MaskFusion(config={"conflict_resolution": "confidence", "smooth_boundaries": False})
    class_names = ["sky"]
    identified = [make_patch("sky", 0, 100, 0, 100)]
    result = fuser.fuse(identified, [], class_names, image_shape=(100, 100))
    assert result["color_mask"].shape == (100, 100, 3)
