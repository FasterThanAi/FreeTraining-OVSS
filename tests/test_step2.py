"""Tests for Step 2: Co-occurrence Matrix Builder."""

import numpy as np
import pytest
from src.step2_cooccurrence.cooccurrence import CooccurrenceBuilder
from src.step1_sam_pass.sam_pass import Patch


def make_patch(label, cx, cy, confidence=0.8):
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[cy-5:cy+5, cx-5:cx+5] = 1
    return Patch(mask=mask, bbox=(cx-5, cy-5, 10, 10), area=100, confidence=confidence, label=label)


def test_cooccurrence_matrix_shape():
    class_names = ["sky", "road", "car"]
    patches = [
        make_patch("sky", 50, 50),
        make_patch("road", 60, 60),  # close → adjacent
        make_patch("car", 150, 150), # far → not adjacent
    ]
    builder = CooccurrenceBuilder(config={"adjacency_method": "centroid_distance", "adjacency_threshold": 50, "normalize": False})
    result = builder.build(patches, class_names, image_shape=(200, 200))
    assert result.M.shape == (3, 3)


def test_cooccurrence_symmetric():
    class_names = ["sky", "road"]
    patches = [
        make_patch("sky", 50, 50),
        make_patch("road", 55, 55),
    ]
    builder = CooccurrenceBuilder(config={"adjacency_method": "centroid_distance", "adjacency_threshold": 30, "normalize": False})
    result = builder.build(patches, class_names, image_shape=(200, 200))
    # Symmetric: M[i,j] == M[j,i]
    np.testing.assert_array_equal(result.M, result.M.T)


def test_row_normalize():
    class_names = ["sky", "road"]
    patches = [
        make_patch("sky", 50, 50),
        make_patch("road", 55, 55),
    ]
    builder = CooccurrenceBuilder(config={"adjacency_method": "centroid_distance", "adjacency_threshold": 30, "normalize": True})
    result = builder.build(patches, class_names, image_shape=(200, 200))
    # Rows that are non-zero should sum to 1
    for row in result.M_normalized:
        if row.sum() > 0:
            assert abs(row.sum() - 1.0) < 1e-5
