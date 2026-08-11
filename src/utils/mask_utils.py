"""Mask manipulation utilities."""

import numpy as np
from typing import List, Dict


def merge_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    OR-merge a list of binary masks into one.

    Args:
        masks: List of binary np.ndarray (H x W)

    Returns:
        Combined binary mask (H x W)
    """
    if not masks:
        return np.zeros((0, 0), dtype=np.uint8)
    result = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        result = np.logical_or(result, m).astype(np.uint8)
    return result


def apply_label_map(
    patches,
    class_to_idx: Dict[str, int],
    image_shape: tuple,
) -> np.ndarray:
    """
    Rasterize a list of labeled patches into an H x W integer label map.

    Args:
        patches: List of Patch objects with .mask and .label
        class_to_idx: Dict mapping class name → integer index
        image_shape: (H, W)

    Returns:
        Integer label map (H x W), dtype int32; -1 = unassigned
    """
    H, W = image_shape
    label_map = np.full((H, W), fill_value=-1, dtype=np.int32)
    for patch in patches:
        if patch.label and patch.label in class_to_idx:
            label_map[patch.mask.astype(bool)] = class_to_idx[patch.label]
    return label_map
