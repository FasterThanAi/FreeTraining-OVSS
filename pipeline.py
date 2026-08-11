"""
Main Pipeline Orchestrator
--------------------------
Runs all 5 steps of the SAM-based zero-shot segmentation pipeline:

  Step 1 → SAM 3-Pass (produces identified patches + unidentified regions)
  Step 2 → Build Co-occurrence Matrix M + INFO
  Step 3 → Preprocessing (fix over-segmentation)
  Step 4 → Contextual Clustering (label unidentified patches)
  Step 5 → Final Mask Fusion (produce complete labeled mask)

Usage:
    from pipeline import SegmentationPipeline
    pipeline = SegmentationPipeline.from_config("configs/config.yaml")
    result = pipeline.run("data/raw/image.jpg", class_names=["sky","road","car"])
"""

import numpy as np
from typing import List, Optional
import yaml

from src.step1_sam_pass import SAMPass
from src.step2_cooccurrence import CooccurrenceBuilder
from src.step3_preprocessing import Preprocessor
from src.step4_clustering import ContextualClusterer
from src.step5_fusion import MaskFusion
from src.utils import load_image, save_mask, visualize_result, get_logger


class SegmentationPipeline:
    """End-to-end pipeline for zero-shot semantic segmentation."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(
            "Pipeline",
            level=config.get("log_level", "INFO"),
            log_file=config.get("log_file", None),
        )

        # Initialize all step modules
        self.step1 = SAMPass(config.get("step1", {}))
        self.step2 = CooccurrenceBuilder(config.get("step2", {}))
        self.step3 = Preprocessor(config.get("step3", {}))
        self.step4 = ContextualClusterer(config.get("step4", {}))
        self.step5 = MaskFusion(config.get("step5", {}))

    @classmethod
    def from_config(cls, config_path: str) -> "SegmentationPipeline":
        """Load pipeline from a YAML config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image_path: str,
        class_names: List[str],
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Run the full 5-step pipeline on a single image.

        Args:
            image_path: Path to input RGB image
            class_names: List of semantic class name strings
            output_dir: If given, saves mask and visualization here

        Returns:
            Dict:
              - label_map: (H x W) integer class index array
              - class_names: list of class names
              - confidence_map: (H x W) float confidence
              - color_mask: (H x W x 3) RGB visualization
        """
        self.logger.info(f"Starting pipeline on: {image_path}")
        self.logger.info(f"Class names: {class_names}")

        # Load image
        image = load_image(image_path)
        self.logger.info(f"Image shape: {image.shape}")

        # ── Step 1: SAM 3-Pass ──────────────────────────────────────────
        self.logger.info("[Step 1] Running SAM 3-Pass...")
        sam_result = self.step1.run(image, class_names)
        self.logger.info(
            f"  Identified patches: {len(sam_result.identified_patches)}  |  "
            f"Unidentified regions: {len(sam_result.unidentified_regions)}"
        )

        # ── Step 2: Co-occurrence Matrix ────────────────────────────────
        self.logger.info("[Step 2] Building co-occurrence matrix...")
        cooc_result = self.step2.build(
            sam_result.identified_patches,
            class_names,
            image_shape=sam_result.image_shape,
        )
        self.logger.info(f"  Matrix shape: {cooc_result.M.shape}")

        # ── Step 3: Preprocessing (Fix Over-segmentation) ───────────────
        self.logger.info("[Step 3] Preprocessing unidentified regions...")
        cleaned_patches = self.step3.process(
            sam_result.unidentified_regions,
            image_shape=sam_result.image_shape,
        )
        self.logger.info(f"  After preprocessing: {len(cleaned_patches)} patch(es)")

        # ── Step 4: Contextual Clustering ───────────────────────────────
        self.logger.info("[Step 4] Contextual clustering...")
        clustering_result = self.step4.label(
            unidentified_patches=cleaned_patches,
            identified_patches=sam_result.identified_patches,
            cooccurrence_result=cooc_result,
        )
        self.logger.info(f"  Labeled {len(clustering_result.labeled_patches)} patch(es)")

        # ── Step 5: Final Mask Fusion ────────────────────────────────────
        self.logger.info("[Step 5] Fusing masks...")
        fusion_result = self.step5.fuse(
            identified_patches=sam_result.identified_patches,
            labeled_unidentified=clustering_result.labeled_patches,
            class_names=class_names,
            image_shape=sam_result.image_shape,
        )
        self.logger.info("  Fusion complete.")

        # ── Save outputs ─────────────────────────────────────────────────
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            save_mask(fusion_result["label_map"], f"{output_dir}/{base}_label_map.png")
            visualize_result(
                image,
                fusion_result["color_mask"],
                output_path=f"{output_dir}/{base}_overlay.png",
            )
            self.logger.info(f"  Saved outputs to {output_dir}/")

        self.logger.info("Pipeline finished.")
        return fusion_result
