"""Tests for Step 4: Contextual Clustering."""

import numpy as np
import pytest
from src.step4_clustering.contextual_clustering import ContextualClusterer
from src.step2_cooccurrence.cooccurrence import CooccurrenceResult
from src.step1_sam_pass.sam_pass import Patch


def make_patch(label=None, cx=50, cy=50, confidence=0.8):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[cy-5:cy+5, cx-5:cx+5] = 1
    emb = np.random.randn(512).astype(np.float32)
    return Patch(mask=mask, bbox=(cx-5, cy-5, 10, 10), area=100, confidence=confidence, label=label, embedding=emb)


def make_cooc_result(class_names):
    n = len(class_names)
    M = np.eye(n, dtype=np.float32)
    return CooccurrenceResult(
        M=M, M_normalized=M,
        class_names=class_names,
        class_to_idx={c: i for i, c in enumerate(class_names)},
        info={},
    )


def test_all_patches_labeled():
    clusterer = ContextualClusterer(config={"embedding_weight": 0.5, "cooccurrence_weight": 0.3, "neighbor_weight": 0.2, "k_neighbors": 3, "min_label_confidence": 0.0})
    class_names = ["sky", "road", "car"]
    identified = [make_patch("sky", cx=30, cy=30), make_patch("road", cx=70, cy=70)]
    unidentified = [make_patch(None, cx=50, cy=50), make_patch(None, cx=20, cy=20)]
    cooc = make_cooc_result(class_names)
    result = clusterer.label(unidentified, identified, cooc)
    assert len(result.labeled_patches) == len(unidentified)
    for p in result.labeled_patches:
        assert p.label is not None


def test_fallback_label_on_low_confidence():
    clusterer = ContextualClusterer(config={
        "embedding_weight": 0.0, "cooccurrence_weight": 0.0, "neighbor_weight": 0.0,
        "k_neighbors": 1, "min_label_confidence": 1.0,  # impossible threshold
        "fallback_label": "background"
    })
    class_names = ["sky"]
    unidentified = [make_patch(None, cx=50, cy=50)]
    cooc = make_cooc_result(class_names)
    result = clusterer.label(unidentified, [], cooc)
    assert result.labeled_patches[0].label == "background"
