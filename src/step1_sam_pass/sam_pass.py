"""
Step 1: Initial SAM 3-Pass
--------------------------
Runs Segment Anything Model (SAM) in three complementary passes:
  Pass 1 — Automatic everything mask
  Pass 2 — Text-prompted / class-guided mask (via CLIP or GLIP)
  Pass 3 — Residual / uncertainty pass on ambiguous regions

Returns:
  - identified_patches: List of (mask, class_label, confidence) for matched regions
  - unidentified_regions: List of masks with no confident class assignment
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Patch:
    """A single segmented patch from SAM."""
    mask: np.ndarray        # Binary mask (H x W)
    bbox: Tuple[int, ...]   # (x, y, w, h)
    area: int
    confidence: float = 0.0
    label: Optional[str] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class SAMPassResult:
    """Output container for the SAM 3-pass step."""
    identified_patches: List[Patch] = field(default_factory=list)
    unidentified_regions: List[Patch] = field(default_factory=list)
    all_patches: List[Patch] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)


class SAMPass:
    """
    Runs SAM in 3 complementary passes and partitions results into
    identified patches and unidentified regions.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dict containing:
                - sam_checkpoint (str): Path to SAM checkpoint file
                - sam_model_type (str): 'vit_h' | 'vit_l' | 'vit_b'
                - device (str): 'cuda' | 'cpu'
                - confidence_threshold (float): Min score to call a patch 'identified'
                - text_encoder (str): 'clip' | 'glip' for pass 2
        """
        self.config = config
        self.sam = None
        self.predictor = None
        self.mask_generator = None
        self.text_encoder = None
        self._initialized = False

    def initialize(self):
        """Load SAM model and text encoder. Call once before run()."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            sam_checkpoint = self.config["sam_checkpoint"]
            model_type = self.config.get("sam_model_type", "vit_h")
            device = self.config.get("device", "cpu")

            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.sam.to(device=device)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=self.config.get("points_per_side", 32),
                pred_iou_thresh=self.config.get("pred_iou_thresh", 0.86),
                stability_score_thresh=self.config.get("stability_score_thresh", 0.92),
                min_mask_region_area=self.config.get("min_mask_region_area", 100),
            )
            self.predictor = SamPredictor(self.sam)
            self._initialized = True
        except ImportError:
            raise ImportError(
                "segment_anything not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray, class_names: List[str]) -> SAMPassResult:
        """
        Execute 3-pass segmentation.

        Args:
            image: RGB image as numpy array (H x W x 3), uint8
            class_names: List of target class name strings

        Returns:
            SAMPassResult with identified_patches and unidentified_regions
        """
        if not self._initialized:
            self.initialize()

        result = SAMPassResult(image_shape=image.shape[:2])

        # --- Pass 1: Automatic Everything ---
        pass1_patches = self._pass1_automatic(image)

        # --- Pass 2: Text-Guided Class Assignment ---
        identified, unidentified = self._pass2_text_guided(image, pass1_patches, class_names)

        # --- Pass 3: Residual Refinement on Unidentified ---
        refined_unidentified = self._pass3_residual(image, unidentified)

        result.all_patches = pass1_patches
        result.identified_patches = identified
        result.unidentified_regions = refined_unidentified

        return result

    # ------------------------------------------------------------------
    # Private Passes
    # ------------------------------------------------------------------

    def _pass1_automatic(self, image: np.ndarray) -> List[Patch]:
        """Pass 1: SAM automatic mask everything."""
        raw_masks = self.mask_generator.generate(image)
        patches = []
        for m in raw_masks:
            p = Patch(
                mask=m["segmentation"],
                bbox=tuple(m["bbox"]),
                area=m["area"],
                confidence=float(m["predicted_iou"]),
            )
            patches.append(p)
        return patches

    def _pass2_text_guided(
        self,
        image: np.ndarray,
        patches: List[Patch],
        class_names: List[str],
    ) -> Tuple[List[Patch], List[Patch]]:
        """
        Pass 2: Match patches to class names using CLIP embeddings.
        Returns (identified_patches, unidentified_patches).
        """
        threshold = self.config.get("confidence_threshold", 0.25)
        try:
            import clip
            import torch
            from PIL import Image

            device = self.config.get("device", "cpu")
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

            # Encode class text
            text_tokens = clip.tokenize(class_names).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            identified, unidentified = [], []
            pil_image = Image.fromarray(image)

            for patch in patches:
                region = self._crop_patch(pil_image, patch)
                region_tensor = clip_preprocess(region).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_features = clip_model.encode_image(region_tensor)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    sims = (img_features @ text_features.T).squeeze(0)
                    best_score, best_idx = float(sims.max()), int(sims.argmax())

                patch.confidence = best_score
                patch.embedding = img_features.cpu().numpy()

                if best_score >= threshold:
                    patch.label = class_names[best_idx]
                    identified.append(patch)
                else:
                    unidentified.append(patch)

            return identified, unidentified

        except ImportError:
            # Fallback — mark all as unidentified if CLIP unavailable
            return [], patches

    def _pass3_residual(
        self, image: np.ndarray, patches: List[Patch]
    ) -> List[Patch]:
        """
        Pass 3: Re-segment unidentified regions with tighter SAM parameters
        to capture fine-grained or previously missed boundaries.
        """
        # TODO: Implement per-region re-segmentation with refined prompts
        # For now, return patches as-is
        return patches

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _crop_patch(self, pil_image, patch: Patch):
        """Crop the image to the patch bounding box."""
        from PIL import Image
        x, y, w, h = patch.bbox
        return pil_image.crop((x, y, x + w, y + h))
