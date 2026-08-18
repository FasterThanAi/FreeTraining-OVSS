# FreeTraining-OVSS

**Training-free open-vocabulary semantic segmentation for remote sensing images**, built on
**SAM 3**.

Given unlabelled satellite imagery and a list of class names, produce fully labelled
segmentation masks — with no training data and no fine-tuning.

> **Terminology note.** This is *training-free open-vocabulary* segmentation, **not
> unsupervised** segmentation. Class names are supplied as input, so the task is zero-shot
> with a known vocabulary. The upstream work (SegEarth-OV, CVPR 2025) makes the same
> distinction, noting the setting is strictly "annotation-free."

---

## Motivation

`SegEarth-OV3` assigns every pixel whose maximum class probability falls below a threshold
τ to the **background** class. On LoveDA (τ = 0.5) this discards a substantial number of
pixels that carry real land-cover labels.

Our reproduction of the baseline makes this measurable:

| Class | Precision | Recall | Gap |
|---|---|---|---|
| water | **89.5** | **54.7** | **+34.8** |
| forest | 57.9 | 44.8 | +13.1 |
| background | **56.9** | **69.4** | **−12.5** |
| building | 77.2 | 78.6 | −1.4 |

SAM 3 is right ~90% of the time it predicts water, yet finds only half the water present.
Background shows the inverse — over-predicted and impure, absorbing pixels that belong to
real classes. The weakness is concentrated in amorphous "stuff"; sharp-boundary "things"
like buildings and roads are balanced.

**Our aim:** recover those discarded pixels using a *semantic co-occurrence prior* over
SAM 3's own region proposals.

See [`WEEK1_RESULTS.md`](WEEK1_RESULTS.md) for the full baseline reproduction.

---

## Status

| Milestone | Status |
|---|---|
| Environment + SAM 3 running | ✅ |
| LoveDA val prepared (1669 images) | ✅ |
| **SegEarth-OV3 baseline reproduced** | ✅ **47.38 mIoU** (paper: 47.4) |
| Discard-rate diagnostic | 🔜 |
| Co-occurrence prior | 🔜 |
| Region-level label assignment | 🔜 |

---

## Environment

⚠️ **The version combination below is not optional.** Three constraints intersect at
exactly one workable point:

| Component | Constraint |
|---|---|
| SAM 3 | **torch ≥ 2.3** (uses `torch.nn.attention`) |
| mmcv prebuilt wheels | available for torch 2.1–2.4; **none for torch 2.5** |
| mmsegmentation 1.2.2 | asserts `mmcv >= 2.0.0rc4, < 2.2.0` |

→ **torch 2.4.1** is the only version satisfying SAM 3 that also has a prebuilt mmcv wheel.
That wheel is mmcv 2.2.0, which mmseg excludes by one patch version, so `MMCV_MAX` must be
raised to `2.3.0`.

### Install

```bash
conda create -n segov3 python=3.11 -y
conda activate segov3

pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"

pip install mmcv==2.2.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install "mmsegmentation==1.2.2"

# raise mmseg's mmcv upper bound (2.2.0 -> 2.3.0)
sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/" \
  $CONDA_PREFIX/lib/python3.11/site-packages/mmseg/__init__.py

pip install einops psutil pycocotools hydra-core iopath timm huggingface_hub omegaconf

python -c "import torch, mmcv, mmseg; from mmcv.ops import point_sample; \
  print('ok', torch.__version__, mmcv.__version__, torch.cuda.is_available())"
```

Or run [`scripts/setup_env.sh`](scripts/setup_env.sh).

### SAM 3 checkpoint

```bash
huggingface-cli download facebook/sam3   # or download sam3.pt manually (~3.45 GB)

mkdir -p weights/sam3
ln -s "$(ls ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt)" \
  weights/sam3/sam3.pt
```

The reference implementation expects the checkpoint at `weights/sam3/sam3.pt` relative to
the repo root.

### Reference implementation

The baseline lives in a separate clone — **do not vendor it into this repo**:

```bash
git clone https://github.com/earth-insights/SegEarth-OV-3.git
```

---

## Dataset — LoveDA

| Item | Value |
|---|---|
| Split used | **Val** (Test has no ground truth — labels withheld for the challenge) |
| Images | Rural 992 + Urban 677 = **1669** |
| Classes | background, building, road, water, barren, forest, agricultural |
| Label encoding | pixel values 1–7; **0 = no-data, ignored** (`reduce_zero_label=True`) |

⚠️ The Kaggle archive nests an extra level: `archive/Val/Val/{Rural,Urban}/`.

```bash
mkdir -p ~/data/loveda/img_dir/val ~/data/loveda/ann_dir/val
SRC=~/Downloads/archive/Val/Val
cp $SRC/{Rural,Urban}/images_png/* ~/data/loveda/img_dir/val/
cp $SRC/{Rural,Urban}/masks_png/*  ~/data/loveda/ann_dir/val/

ls ~/data/loveda/img_dir/val | wc -l   # must be 1669
ls ~/data/loveda/ann_dir/val | wc -l   # must be 1669
```

Both counts must match, or Rural/Urban filenames have collided.

---

## Reproducing the baseline

```bash
cd /path/to/SegEarth-OV-3
ln -s ~/data/loveda data/LoveDA
python eval.py ./configs/cfg_loveda.py
```

Expected: **mIoU ≈ 47.4**. Roughly 24 minutes at 0.85 s/image; peak memory 6.1 GB.

| Observed | Meaning |
|---|---|
| 46–48 | ✅ Reproduced |
| 40–46 | Check prompt wording, τ, decoder confidence threshold |
| < 40 | Structural bug — suspect `reduce_zero_label` or the Rural/Urban merge |
| > 50 | Also a bug — likely mishandling the ignore class |

### Key parameters (`configs/cfg_loveda.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `prob_thd` (**τ**) | 0.5 | Below this, a pixel is discarded to background |
| `confidence_threshold` | 0.5 | Transformer decoder confidence |
| Input resolution | 1024×1024 | LoveDA native; no `Resize` in the test pipeline |

---

## Proposed method

Under active development; the design below supersedes the original five-step sketch.

1. **SAM 3 pass** — dual-head mask fusion + presence-guided filtering, giving per-region
   scores and class-agnostic mask proposals.
2. **Region proposals from SAM 3 itself** — *not* SLIC or Felzenszwalb superpixels. SAM's
   own masks are strictly better region proposals; classical over-segmentation would be a
   regression.
3. **Corpus-level co-occurrence matrix M** — built across the whole unlabelled dataset from
   **spatial adjacency** of high-confidence regions, not per-image co-presence. A per-image
   matrix over 7 classes is nearly information-free.
4. **Label assignment as energy minimisation** over SAM-derived regions:
   - unary — SAM 3 fused probability × presence score
   - pairwise appearance — DINOv3 / SAM 3 feature similarity
   - pairwise semantic — −log M(cᵢ, cⱼ)
   - solved by mean-field or graph cut
5. **Fusion** with the confident predictions.

### Known risk: circularity

M and the class prototypes are both derived from SAM 3's own confident predictions. If
SAM 3 is systematically wrong on a class, M encodes that error and propagation amplifies
it. Mitigation (entropy weighting, symmetric consistency checks) must be explicit and
ablated.

---

## Related work

| Work | Relation |
|---|---|
| [SegEarth-OV3](https://arxiv.org/abs/2512.08730) | Direct baseline. SAM 3 for remote-sensing OVSS |
| [SegEarth-OV](https://openaccess.thecvf.com/) (CVPR 2025) | CLIP-based predecessor |
| **[ConInfer](https://arxiv.org/abs/2603.29271)** | **Closest related work.** Context-at-inference for OVRSS via DINOv3 GMM clustering + KL consensus. Purely *visual* context, patch-level, CLIP-based — no semantic class-pair prior. Names pixel/region-level contextual modelling as future work. |

Our differentiation: a **semantic co-occurrence prior** combined with **SAM 3's
region-level granularity**, versus ConInfer's purely visual patch-level context.

---

## Repository layout

```
FreeTraining-OVSS/
├── WEEK1_RESULTS.md        # baseline reproduction — start here
├── README.md
├── SETUP_SAM3.md           # detailed environment notes
├── ROADMAP.md
├── ANALYSIS.md
├── configs/
├── scripts/
│   └── setup_env.sh
├── src/
└── tests/
```

## Citation

```bibtex
@article{li2025segearthov3,
  title={SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic
         Segmentation in Remote Sensing Images},
  author={Li, Kaiyu and Zhang, Shengqi and Wang, Yujie and Deng, Yupeng
          and Wang, Zhi and Meng, Deyu and Cao, Xiangyong},
  journal={arXiv preprint arXiv:2512.08730},
  year={2025}
}
```
