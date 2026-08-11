"""Tests for Step 3: Preprocessing (Over-segmentation Fix)."""

import numpy as np
import pytest
from src.step3_preprocessing.preprocessing import Preprocessor
from src.step1_sam_pass.sam_pass import Patch


def make_patch(area_size, offset=(10, 10)):
    mask = np.zeros((200, 200), dtype=np.uint8)
    x, y = offset
    mask[y:y+area_size, x:x+area_size] = 1
    return Patch(mask=mask, bbox=(x, y, area_size, area_size), area=area_size**2, confidence=0.0, label=None)


def test_small_region_suppression():
    preprocessor = Preprocessor(config={"min_area": 200, "use_morphology": False, "merge_max_iterations": 0})
    patches = [make_patch(5), make_patch(20)]  # areas: 25, 400
    result = preprocessor.process(patches, image_shape=(200, 200))
    assert len(result) == 1  # only the 400-px patch survives
    assert result[0].area == 400


def test_morphological_leaves_valid_mask():
    preprocessor = Preprocessor(config={"min_area": 10, "use_morphology": True, "morph_kernel_size": 3, "merge_max_iterations": 0})
    patches = [make_patch(30)]
    result = preprocessor.process(patches, image_shape=(200, 200))
    # After morphological ops, mask should still be non-empty
    assert result[0].mask.sum() > 0


def test_greedy_merge():
    preprocessor = Preprocessor(config={"min_area": 10, "use_morphology": False, "merge_iou_threshold": 0.1, "merge_max_iterations": 3})
    # Two overlapping patches
    p1 = make_patch(30, offset=(10, 10))
    p2 = make_patch(30, offset=(15, 15))  # overlaps p1
    result = preprocessor.process([p1, p2], image_shape=(200, 200))
    assert len(result) == 1
