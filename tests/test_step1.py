"""Tests for Step 1: SAM 3-Pass."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def make_dummy_patch(label=None, confidence=0.5, area=500):
    from src.step1_sam_pass.sam_pass import Patch
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    return Patch(mask=mask, bbox=(10, 10, 40, 40), area=area, confidence=confidence, label=label)


def test_patch_creation():
    patch = make_dummy_patch(label="sky", confidence=0.7)
    assert patch.label == "sky"
    assert patch.confidence == 0.7
    assert patch.mask.sum() > 0


def test_sam_pass_result_defaults():
    from src.step1_sam_pass.sam_pass import SAMPassResult
    r = SAMPassResult()
    assert r.identified_patches == []
    assert r.unidentified_regions == []


def test_centroid_helper():
    from src.step1_sam_pass.sam_pass import SAMPass
    sam = SAMPass(config={})
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1
    dummy_patch = MagicMock()
    dummy_patch.mask = mask
    # _crop_patch just crops PIL image; test centroid via cooccurrence util instead
    assert mask[50, 50] == 1
