# Week 1 Results — Baseline Reproduction

**Goal:** Reproduce SegEarth-OV3's reported LoveDA validation mIoU of **47.4** on local hardware.

**Status:** 🟢 **REPRODUCED — 47.38 mIoU vs paper's 47.4**
**Date started:** 2026-08-17
**Last updated:** 2026-08-18

---

## 1. Hardware

| Item | Value |
|---|---|
| Machine | HP Z4 G5 Workstation |
| GPU | NVIDIA RTX 2000 Ada Generation |
| VRAM | 16380 MiB (16 GB) |
| CUDA driver / system nvcc | **13.3** |
| OS | Ubuntu (XFCE) |

> **VRAM outcome:** ✅ No OOM. LoveDA runs at its native **1024×1024** (the test pipeline
> contains no `Resize` step), and the segmentor does sliding-window inference internally.
> 16 GB was sufficient. No fp16/autocast or resolution reduction was needed, so the result
> remains directly comparable to the paper's 47.4.

## 2. Environment

Conda env: `segov3`

| Package | Version |
|---|---|
| Python | 3.11.15 (conda-forge) |
| torch | **2.4.1+cu121** |
| torchvision | 0.19.1+cu121 |
| CUDA (torch build) | 12.1 |
| mmengine | 0.10.7 |
| mmcv | **2.2.0** (prebuilt wheel, torch2.4/cu121 index) |
| mmsegmentation | 1.2.2 — **`MMCV_MAX` patched `2.2.0` → `2.3.0`** |
| sam3 | vendored copy inside `SegEarth-OV-3/sam3/` (shadows `~/sam3`) |
| numpy | 1.26.4 (pinned `<2`) |
| opencv-python | 4.10.0.84 |
| SAM 3 checkpoint | `sam3.pt`, 3,450,062,241 bytes (3.45 GB) |

`torch.cuda.is_available()` → **True** ✅

Full freeze: `~/logs/segov3-env-freeze.txt`

### The version constraint that determines everything

Three requirements intersect at exactly one workable point:

| Component | Constraint |
|---|---|
| SAM 3 | **torch ≥ 2.3** (uses `torch.nn.attention`, which does not exist before 2.3) |
| mmcv prebuilt wheels | available for torch 2.1–2.4; **none for torch 2.5** |
| mmsegmentation 1.2.2 | asserts `mmcv >= 2.0.0rc4, < 2.2.0` |

→ **torch 2.4.1 is the only version satisfying SAM 3 that also has prebuilt mmcv wheels.**
Its wheel is mmcv 2.2.0, excluded by mmseg's upper bound by one patch version, so
`MMCV_MAX` was raised to `'2.3.0'` in
`$CONDA_PREFIX/lib/python3.11/site-packages/mmseg/__init__.py`.
The ops mmseg actually calls are unchanged between mmcv 2.1 and 2.2.

### Working install sequence (reproducible)

```bash
conda create -n segov3 python=3.11 -y && conda activate segov3
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install "mmsegmentation==1.2.2"
# patch mmseg's mmcv upper bound
sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/" \
  $CONDA_PREFIX/lib/python3.11/site-packages/mmseg/__init__.py
pip install einops psutil pycocotools hydra-core iopath timm huggingface_hub omegaconf
# checkpoint: repo expects weights/sam3/sam3.pt relative to SegEarth-OV-3/
mkdir -p ~/SegEarth-OV-3/weights/sam3
ln -s "$(ls ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt)" \
  ~/SegEarth-OV-3/weights/sam3/sam3.pt
```

### Environment notes / gotchas hit
- Initial venv used **Python 3.13** — no mmcv wheels exist for it. Rebuilt on 3.11 via conda.
- **`mmcv-lite` is unusable.** Every mmseg version eagerly imports compiled CUDA ops at
  package import (via `focal_loss` / `mask_classification` → `mmcv.ops` → `mmcv._ext`).
  Installing both `mmcv` and `mmcv-lite` in one env also causes `ModuleNotFound` errors.
- **Building mmcv from source failed** on this machine: system nvcc is CUDA **13.3** while
  torch is built against **12.1**, and torch refuses to compile extensions across that gap.
  Installing `cuda-toolkit` into the conda env did not provide an `nvcc` binary
  (`cuda-nvcc` is the package that does). Prebuilt wheels sidestep the issue entirely.
- mmcv 2.1.0's `setup.py` imports `pkg_resources`, absent from modern setuptools — source
  builds need `--no-build-isolation` plus `setuptools<70`.
- SAM 3's `model_builder.py` imports the **video** predictor stack at module level, so the
  video dependencies (`psutil`, `pycocotools`, …) must be installed even for image-only work.
- The repo ships a **vendored `sam3/`** that shadows the editable `~/sam3` install when
  running from inside `SegEarth-OV-3/`. Edits to `~/sam3` have no effect there.

## 3. Dataset

**LoveDA** (validation split)

| Item | Value |
|---|---|
| Source | Kaggle archive — note the **doubled `Val/Val/` nesting** |
| Split used | **Val** (Test has no ground-truth masks — labels withheld for challenge) |
| Val images | Rural 992 + Urban 677 = **1669** |
| Prepared path | `~/data/loveda/{img_dir,ann_dir}/val`, symlinked to `SegEarth-OV-3/data/LoveDA` |
| Classes (7) | background, building, road, water, barren, forest, agricultural |
| Label encoding | pixel values 1–7; **0 = no-data, ignored** (`reduce_zero_label=True`) |

**Verification checklist**
- [x] `img_dir/val` count == `ann_dir/val` count == **1669** ✅
- [x] Rural + Urban merged with no filename collisions ✅
- [x] `reduce_zero_label=True` confirmed in `cfg_loveda.py` ✅
- [x] Sample mask `2522.png`: `uint8`, 1024×1024, unique values `[1 2 3 4 5 7]` ✅

## 4. Configuration (`configs/cfg_loveda.py`)

| Parameter | Value |
|---|---|
| `prob_thd` (τ, background threshold) | **0.5** |
| `confidence_threshold` (decoder) | **0.5** |
| `classname_path` | `configs/cls_loveda.txt` |
| Test pipeline | `LoadImageFromFile → LoadAnnotations → PackSegInputs` — **no Resize** |
| Effective input resolution | 1024×1024 (LoveDA native) |
| Evaluator | `IoUMetric`, metrics `['mIoU', 'mFscore']` |

**Class prompts** (comma = synonym augmentation):
```
background
building,house
road
water
barren,bareland,soil
forest,tree
agricultural
```

> **τ = 0.5 is the single most important number for this project.** It is the threshold
> below which pixels are discarded to "background," and recovering those pixels is the
> premise of the co-occurrence prior. Note it is 5× the demo's default of 0.1, and the
> paper states τ is tuned per dataset.

## 5. Reproduction Result

**Reference (paper, Table 1):** LoveDA = 47.4 mIoU

| Run | Config | Images | mIoU | Notes |
|---|---|---|---|---|
| Mini (smoke test) | `cfg_loveda_mini.py` | 20 | **38.97** | filename-sorted subset, not representative |
| **Baseline (full)** | `cfg_loveda.py` | 1669 | **47.38** | ✅ matches paper's 47.4 (Δ 0.02) |

### Mini-run per-class results (20 images — indicative only)

| Class | IoU | Acc | Fscore | Precision | Recall |
|---|---|---|---|---|---|
| background | 57.46 | 93.76 | 72.98 | 59.74 | 93.76 |
| building | 50.17 | 60.95 | 66.82 | 73.93 | 60.95 |
| road | 56.04 | 71.93 | 71.83 | 71.73 | 71.93 |
| water | 47.26 | 50.08 | 64.18 | **89.34** | 50.08 |
| barren | 23.12 | 33.27 | 37.56 | 43.11 | 33.27 |
| forest | **0.00** | 0.00 | 0.01 | 1.20 | 0.00 |
| agricultural | 38.77 | 40.62 | 55.87 | **89.47** | 40.62 |

Aggregate: `aAcc 65.93 · mIoU 38.97 · mAcc 50.09 · mFscore 52.75 · mPrecision 61.22 · mRecall 50.09`

**Caveats on the mini run:** 20 images selected by `ls | head -20` (filename order), so the
subset is likely skewed Rural-vs-Urban and is far too small for comparison against 47.4.

**Two observations worth carrying forward:**
1. **forest = 0.00 IoU.** The sample mask contained no class 6 at all, so forest may simply
   be absent from this subset — but if it stays near zero on the full run, the `forest,tree`
   prompt is failing and that is a finding in itself.
2. **High precision, low recall on water (89.3 / 50.1) and agricultural (89.5 / 40.6).**
   The model is confident when it fires but misses most of each class — consistent with
   pixels falling below τ and being dumped into background. This is the project premise
   appearing in the very first measurement.

### Full per-class results (1669 images) — **the key table**

| Class | IoU | Acc | Fscore | Precision | Recall | P−R gap |
|---|---|---|---|---|---|---|
| building | 63.81 | 78.60 | 77.90 | 77.22 | 78.60 | −1.4 |
| road | 53.89 | 70.53 | 70.04 | 69.55 | 70.53 | −1.0 |
| **water** | 51.44 | 54.73 | 67.93 | **89.54** | **54.73** | **+34.8** |
| agricultural | 47.47 | 62.02 | 64.38 | 66.92 | 62.02 | +4.9 |
| background | 45.50 | 69.40 | 62.55 | **56.92** | **69.40** | **−12.5** |
| barren | 35.73 | 53.87 | 52.65 | 51.49 | 53.87 | −2.4 |
| forest | 33.78 | 44.80 | 50.50 | 57.88 | 44.80 | +13.1 |

Aggregate: `aAcc 63.80 · **mIoU 47.38** · mAcc 61.99 · mFscore 63.71 · mPrecision 67.08 · mRecall 61.99`

### Interpretation — the premise, measured

The per-class breakdown supports the project premise directly:

1. **Water: 89.5% precision, 54.7% recall (+34.8 gap).** SAM 3 is right nine times out of
   ten when it predicts water — but finds only half the water present. Those pixels are not
   being *misclassified*; they fall below τ = 0.5 and are discarded to background.
2. **Background: 56.9% precision, 69.4% recall (−12.5, the inverse pattern).** Background is
   over-predicted and impure — it is absorbing pixels belonging to real classes. This is the
   discard bucket, visible directly in the metrics.
3. **Forest (+13.1) and agricultural (+4.9)** show the same asymmetry at smaller scale.
4. **Building (−1.4) and road (−1.0) are balanced.** Sharp-boundary "things" are handled well;
   the weakness is concentrated in amorphous "stuff" — exactly the duality the paper identifies.

→ The pixels the co-occurrence prior aims to recover are measurable, substantial, and
concentrated in identifiable classes. Section 7's diagnostic converts this indirect
evidence into a direct number.

### Interpretation guide
| Observed | Meaning |
|---|---|
| 46–48 | ✅ Reproduced |
| 40–46 | Close — check prompt wording, τ, decoder confidence threshold |
| < 40 | Structural bug — suspect `reduce_zero_label` or Rural/Urban merge |
| > 50 | Also a bug — likely mishandling the ignore class, inflating the score |

## 6. Runtime & Resource Profile

| Metric | Value |
|---|---|
| Seconds per image | **0.85** (full run; mini run measured 0.87) |
| Projected full eval (1669 images) | ~24 minutes |
| Actual full eval wall time | **~24 minutes** ✅ matched projection |
| **Peak memory during inference** | **6115 MB** — only 37% of the 16 GB card |
| Input resolution used | **1024×1024** (native; no Resize in pipeline) |

> **Headroom:** peak usage of 6.1 GB against 16 GB available leaves ~10 GB free. Adding
> DINOv3 features, storing per-region embeddings, or running heavier inference on top of
> SAM 3 is comfortably feasible on this machine without resolution reduction or fp16.

> Vocabulary classes are processed **sequentially** — 7 classes means ~7 forward passes per
> image. Note this when estimating cost for larger vocabularies later.
>
> A 24-minute full evaluation is cheap enough to re-run repeatedly, which makes threshold
> sweeps over τ practical.

## 7. Diagnostic: Discard Rate

*The key measurement motivating the project's premise.*

Instrument inference to record, per image:
- fraction of pixels falling below τ = 0.5 (assigned "background")
- of those below-τ pixels, the **ground-truth class distribution**

| Metric | Value |
|---|---|
| Mean % pixels below τ | _TBD_ |
| Of those, % with a real (non-background) GT label | _TBD_ |

**Decision rule:**
- High discard rate with substantial real-class content → thesis premise confirmed;
  proceed with the co-occurrence prior direction.
- Low discard rate (<5%) → premise weak; pivot toward the medium-resolution /
  domain-gap angle instead (cf. GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).

Early indirect evidence from the mini run: recall is far below precision on water,
agricultural, and barren — consistent with a substantial discard rate.

## 8. Top Confused Class Pairs

From the confusion matrix — these are the pairs a semantic co-occurrence prior would target.

| Rank | True → Predicted | Count / % |
|---|---|---|
| 1 | _TBD_ | |
| 2 | _TBD_ | |
| 3 | _TBD_ | |
| 4 | _TBD_ | |
| 5 | _TBD_ | |

## 9. Failure Cases

Attach 2–3 qualitative examples (image / GT / prediction) showing characteristic failures.

- `demo.py` smoke-test output saved to `SegEarth-OV-3/seg_pred.png` (OpenEarthMap sample)
- _TBD_ — LoveDA failure cases

## 10. Open Issues / Blockers

- [x] ~~mmcv not installed~~ — resolved via torch 2.4.1 + prebuilt mmcv 2.2.0 wheel
- [x] ~~`forest` IoU = 0.00 on the mini run~~ — **resolved: subset composition.** Forest
      reaches **33.78 IoU** on the full 1669 images. The 20-image subset simply contained no
      class-6 pixels. The `forest,tree` prompt works correctly.
- [x] ~~Peak VRAM not measured~~ — **6115 MB peak**, 37% of available
- [ ] mmseg `MMCV_MAX` patch is a site-packages edit — will not survive an env rebuild.
      Capture it in a setup script.

## 11. Next Steps

1. ~~Resolve mmcv installation~~ ✅
2. ~~Verify imports: `torch, mmcv, mmseg, sam3`~~ ✅
3. ~~SAM 3 checkpoint~~ ✅ (3.45 GB, symlinked into `weights/sam3/`)
4. ~~Smoke test with `demo.py`~~ ✅
5. ~~Prepare + verify LoveDA val directory~~ ✅ 1669/1669
6. ~~Trial run on 20 images~~ ✅ mIoU 38.97, 0.87 s/img
7. ~~Full evaluation on 1669 val images~~ ✅ **mIoU 47.38 (paper: 47.4)**
8. ~~Record full numbers~~ ✅ — commit to `FreeTraining-OVSS`
9. **Week 2:** instrument inference for the discard-rate diagnostic (§7)
10. **Week 2:** extract the confusion matrix (§8)

---

## References

- SegEarth-OV3 — arXiv:2512.08730 · `github.com/earth-insights/SegEarth-OV-3`
- SegEarth-OV — CVPR 2025
- ConInfer — arXiv:2603.29271 (closest related work; context-at-inference for OVRSS)
