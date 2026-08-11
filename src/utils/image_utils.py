"""Shared image and visualization utilities."""

import numpy as np
from pathlib import Path
from typing import Optional, List


def load_image(path: str) -> np.ndarray:
    """Load an RGB image from disk as a numpy array (H x W x 3, uint8)."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_mask(label_map: np.ndarray, output_path: str):
    """Save an integer label map as a PNG file."""
    from PIL import Image
    out = Image.fromarray(label_map.astype(np.uint8))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def visualize_result(
    image: np.ndarray,
    color_mask: np.ndarray,
    output_path: Optional[str] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay the color mask on the original image with transparency.

    Args:
        image: Original RGB image (H x W x 3)
        color_mask: RGB color mask (H x W x 3)
        output_path: If given, save the overlay image here
        alpha: Transparency of the mask overlay

    Returns:
        Composite overlay image as numpy array
    """
    blended = (alpha * color_mask.astype(float) + (1 - alpha) * image.astype(float))
    blended = blended.clip(0, 255).astype(np.uint8)

    if output_path:
        from PIL import Image
        Image.fromarray(blended).save(output_path)

    return blended
