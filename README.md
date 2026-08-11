# SAM Zero-Shot Semantic Segmentation

A research implementation of a 5-step zero-shot semantic segmentation pipeline using **SAM (Segment Anything Model)** + **CLIP** for class assignment, co-occurrence context modeling, and contextual clustering.

---

## 📐 System Workflow

```
[Input: Unlabeled Image + Class Names]
           ▼
  Step 1: Initial SAM 3-Pass
        /              \
[Identified Patches]  [Unidentified Regions]
        ▼                     ▼
Step 2: Co-occurrence    Step 3: Preprocessing
  Matrix M + INFO         (Fix Over-segmentation)
        \                     /
         ▼                   ▼
       Step 4: Contextual Clustering
          (Label Unidentified Patches)
                  ▼
        Step 5: Final Mask Fusion
                  ▼
     [Output: Complete Labeled Mask]
```

---

## 📁 Project Structure

```
Final_year_project/
├── pipeline.py                        # Main orchestrator (runs all 5 steps)
├── requirements.txt
├── configs/
│   └── config.yaml                    # All hyperparameters & settings
├── src/
│   ├── step1_sam_pass/
│   │   ├── __init__.py
│   │   └── sam_pass.py                # SAM 3-pass + CLIP class assignment
│   ├── step2_cooccurrence/
│   │   ├── __init__.py
│   │   └── cooccurrence.py            # Co-occurrence matrix M + INFO
│   ├── step3_preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessing.py           # Over-segmentation fix
│   ├── step4_clustering/
│   │   ├── __init__.py
│   │   └── contextual_clustering.py   # Label unidentified patches
│   ├── step5_fusion/
│   │   ├── __init__.py
│   │   └── mask_fusion.py             # Final mask fusion
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py             # Load/save/visualize
│       ├── mask_utils.py              # Merge masks, label maps
│       └── logger.py                  # Logging factory
├── scripts/
│   └── run_pipeline.py                # CLI entry point
├── data/
│   ├── raw/                           # Input images
│   ├── processed/
│   └── annotations/                   # Ground truth (for evaluation)
├── outputs/
│   ├── masks/
│   ├── visualizations/
│   └── logs/
├── notebooks/                         # Jupyter notebooks for experiments
└── tests/
    ├── test_step1.py
    ├── test_step2.py
    ├── test_step3.py
    ├── test_step4.py
    └── test_step5.py
```

---

## ⚙️ Installation

```bash
# 1. Clone and enter the project
cd Final_year_project

# 2. Create a virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install base requirements
pip install -r requirements.txt

# 4. Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# 5. Install CLIP
pip install git+https://github.com/openai/CLIP.git

# 6. Download SAM checkpoint (ViT-H)
mkdir -p models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/
```

---

## 🚀 Quick Start

```bash
python scripts/run_pipeline.py \
    --image data/raw/sample.jpg \
    --classes sky road car person building \
    --config configs/config.yaml \
    --output outputs/
```

Or use the Python API:

```python
from pipeline import SegmentationPipeline

pipeline = SegmentationPipeline.from_config("configs/config.yaml")
result = pipeline.run(
    image_path="data/raw/sample.jpg",
    class_names=["sky", "road", "car", "person"],
    output_dir="outputs/",
)

# result["label_map"]    → H x W integer array
# result["color_mask"]   → H x W x 3 RGB visualization
```

---

## 🔧 Key Config Parameters (`configs/config.yaml`)

| Parameter | Step | Description |
|---|---|---|
| `sam_checkpoint` | Step 1 | Path to SAM model weights |
| `confidence_threshold` | Step 1 | Min CLIP similarity to label a patch |
| `adjacency_threshold` | Step 2 | Max centroid distance to count as adjacent |
| `min_area` | Step 3 | Min patch size (pixels) — smaller are removed |
| `merge_iou_threshold` | Step 3 | Merge patches with higher overlap |
| `embedding_weight` | Step 4 | Weight of visual similarity in cluster score |
| `cooccurrence_weight` | Step 4 | Weight of co-occurrence matrix context |
| `neighbor_weight` | Step 4 | Weight of spatial neighbor voting |
| `conflict_resolution` | Step 5 | How to resolve overlapping masks |
