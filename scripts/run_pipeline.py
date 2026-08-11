"""
Quick-run script for the SAM segmentation pipeline.

Usage:
    python scripts/run_pipeline.py \
        --image data/raw/sample.jpg \
        --classes sky road car person \
        --config configs/config.yaml \
        --output outputs/

"""

import argparse
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import SegmentationPipeline


def parse_args():
    p = argparse.ArgumentParser(description="SAM Zero-Shot Segmentation Pipeline")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--classes", nargs="+", required=True, help="Class names")
    p.add_argument("--config", default="configs/config.yaml", help="Config YAML path")
    p.add_argument("--output", default="outputs/", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline = SegmentationPipeline.from_config(args.config)
    result = pipeline.run(
        image_path=args.image,
        class_names=args.classes,
        output_dir=args.output,
    )
    print(f"\n✅ Done. label_map shape: {result['label_map'].shape}")
    print(f"   Classes: {result['class_names']}")


if __name__ == "__main__":
    main()
