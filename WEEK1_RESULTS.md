# Week 1–2 Results — Baseline Reproduction & Discard-Rate Diagnostic

**Goal:** Reproduce SegEarth-OV3's reported LoveDA validation mIoU of **47.4** on local hardware,
then measure how much real land cover the baseline discards to "background".

**Status:** 🟢 **REPRODUCED — 47.38 mIoU vs paper's 47.4** · 🟢 **PREMISE CONFIRMED — 29.68% of
real-class pixels discarded at τ = 0.5** · 🟢 **Confusion analysis complete** ·
🔴 **`measure_discard_rate.py` not in version control — §7–§9 unreproducible from this repo (§10)**
**Date started:** 2026-08-17
**Last updated:** 2026-08-21

---

## 1. Hardware

| Item | Value |
|---|---|
| Machine | HP Z4 G5 Workstation |
| GPU | NVIDIA RTX 2000 Ada Generation |
| VRAM | 16380 MiB (16 GB) |
| Driver | 580.173.02 (`nvidia-smi` reports max CUDA **13.0**) |
| System nvcc | **13.3** |
| OS | Ubuntu (XFCE) |

> **VRAM outcome:** ✅ No OOM. LoveDA runs at its native **1024×1024** (the test pipeline
> contains no `Resize` step), and the segmentor does sliding-window inference internally.
> 16 GB was sufficient. No fp16/autocast or resolution reduction was needed, so the result
> remains directly comparable to the paper's 47.4.

> **Note the nvcc/driver mismatch above** — system nvcc (13.3) is *newer* than the driver's
> reported maximum (13.0). This is what defeated the mmcv source build (§2). Prebuilt wheels
> sidestep it, but the discrepancy is worth stating rather than glossing.

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
| Tile size | 1024 × 1024 = 1,048,576 px |
| Classes (7) | background, building, road, water, barren, forest, agricultural |
| Label encoding | pixel values 1–7; **0 = no-data, ignored** (`reduce_zero_label=True`) |

**Verification checklist**
- [x] `img_dir/val` count == `ann_dir/val` count == **1669** ✅
- [x] Rural + Urban merged with no filename collisions ✅
- [x] `reduce_zero_label=True` confirmed in `cfg_loveda.py` ✅
- [x] Sample mask `2522.png`: `uint8`, 1024×1024, unique values `[1 2 3 4 5 7]` ✅

### Class composition (from the confusion matrix, τ = 0.5)

| Class | GT pixels | Share of real-class |
|---|---|---|
| agricultural | 487,082,702 | 44.7% |
| water | 199,567,816 | 18.3% |
| forest | 125,615,647 | 11.5% |
| building | 122,805,791 | 11.3% |
| road | 79,590,500 | 7.3% |
| barren | 74,383,133 | 6.8% |
| **Total real-class** | **1,089,045,589** | 100% |

Total labelled (non-no-data) pixels = **1,704,296,271**; background accounts for the remaining
615,250,682 (36.1%). Since 1669 × 1,048,576 = 1,750,073,344, roughly **2.6% of pixels are
no-data** and correctly excluded.

> **Consistency check:** the confusion-matrix row sums total exactly 1,089,045,589, matching the
> headline figure derived independently by the discard instrumentation. Two separate accounting
> paths, identical totals.

**Agricultural is 44.7% of all real-class pixels** — nearly half the dataset. This dominance
matters for §8: it makes agricultural an attractor for ambiguous predictions, and it means any
per-class improvement there moves mIoU disproportionately.

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
>
> §7.2 and §8.2 now quantify exactly what that choice costs, in both directions.

## 5. Reproduction Result

**Reference (paper, Table 1):** LoveDA = 47.4 mIoU

| Run | Config | Images | mIoU | Notes |
|---|---|---|---|---|
| Mini (smoke test) | `cfg_loveda_mini.py` | 20 | **38.97** | filename-sorted subset, not representative |
| **Baseline (full)** | `cfg_loveda.py` | 1669 | **47.38** | ✅ matches paper's 47.4 (Δ 0.02) |
| Independent recompute | discard instrumentation, τ = 0.5 | 1669 | **47.37** | ✅ Δ 0.01 — **label alignment verified** |

The third row is a correctness gate, not a result. The Week 2 diagnostic recomputes mIoU from its
own confusion matrix using its own class indexing; agreement to 0.01 proves the instrumentation
measures the same quantity the baseline evaluator measures. **Every number in §7–§9 inherits that
guarantee.**

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

→ §7 converts this indirect evidence into direct numbers, and §7.6 decomposes each class's error
budget into its two mechanisms. The `mAcc` column here is reproduced exactly by the confusion
matrix (§7.6), a further consistency check.

### Interpretation guide
| Observed | Meaning |
|---|---|
| 46–48 | ✅ Reproduced ← **we are here (47.38)** |
| 40–46 | Close — check prompt wording, τ, decoder confidence threshold |
| < 40 | Structural bug — suspect `reduce_zero_label` or Rural/Urban merge |
| > 50 | Also a bug — likely mishandling the ignore class, inflating the score |

## 6. Runtime & Resource Profile

| Metric | Value |
|---|---|
| Seconds per image | **0.85** (full run; mini run measured 0.87) |
| Projected full eval (1669 images) | ~24 minutes |
| Actual full eval wall time | **~24 minutes** ✅ matched projection |
| **Peak memory during baseline eval** | **6115 MB** — 37% of the 16 GB card |
| Peak VRAM during Week 2 diagnostic | **8534 MiB** — 52% (extra GT/confusion bookkeeping) |
| Sustained thermals / power | 77 °C at the **70 W cap**, 100% GPU util |
| Input resolution used | **1024×1024** (native; no Resize in pipeline) |

> **Headroom:** even the heavier diagnostic run peaks at 8.5 GB against 16 GB available.
> Adding DINOv3 features, storing per-region embeddings, or running heavier inference on top of
> SAM 3 is comfortably feasible without resolution reduction or fp16.

> **The card is power-limited, not thermally throttled** (64 W drawn against a 70 W cap at
> 100% util). ~24 min per full pass is the floor on this hardware; no tuning will improve it.

> Vocabulary classes are processed **sequentially** — 7 classes means ~7 forward passes per
> image. Note this when estimating cost for larger vocabularies later.

> **Cost note carried into Week 3:** the τ-sweep below ran the vision encoder three times over
> the full split to produce what is arithmetic on a saved confidence map. Caching
> `(conf, pred, gt, S_pres)` per image turns any future threshold or ablation query from ~24 min
> into seconds. See §11 item 3.

## 7. Diagnostic: Discard Rate ✅

*The key measurement motivating the project's premise.*

Instrumented inference (`scripts/measure_discard_rate.py`) records, per image: the fraction of
real-class GT pixels the baseline assigns to "background", and the class distribution of those
discarded pixels. Outputs in `~/outputs/week2_tau{0.5,0.3,0.1}/`.

### 7.1 Headline — at the paper's own operating point (τ = 0.5)

| Metric | Value |
|---|---|
| Labelled (non-no-data) pixels | 1,704,296,271 |
| Pixels with a real class (excl. background) | 1,089,045,589 (63.9%) |
| **Of those, discarded to background** | **323,184,908 — 29.68%** |
| Per-image discard rate | mean **33.79%**, median 18.51%, max 100.00% |

**Nearly one third of all real land-cover pixels in LoveDA val are thrown away by the baseline.**
Against the pivot rule below (`<5% → premise weak`), this clears the bar by a factor of six.

### 7.2 τ-sweep — the residual is *not* threshold-tunable

| τ | mIoU | Real-class px discarded | Per-image mean | Per-image median |
|---|---|---|---|---|
| **0.5** (baseline) | **47.37** | **29.68%** (323,184,908) | 33.79% | 18.51% |
| 0.3 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 0.1 | **41.83** | **10.88%** (118,477,557) | 12.06% | 0.39% |

> Fill the τ=0.3 row: `head -20 ~/outputs/week2_tau0.3/discard_summary.md`

**This table is the argumentative core of the project.** Lowering τ from 0.5 to 0.1 recovers
roughly two-thirds of the discarded pixels — and costs **5.54 mIoU** (47.37 → 41.83).

§8.2 itemises exactly where that loss comes from. The short version: τ is a scalar with no
semantics, so it cannot distinguish a correct recovery from a hallucination. **The residual is
real, it is large, and reaching it requires a mechanism that reasons about *what* a region
plausibly is — which is exactly what a semantic co-occurrence prior supplies.**

### 7.3 Loss by class

| Class | GT pixels | Lost @ τ=0.5 | Lost @ τ=0.3 | Lost @ τ=0.1 |
|---|---|---|---|---|
| forest | 125,615,647 | 43,462,196 (**34.60%**) | 24,343,686 (19.38%) | 9,001,259 (7.17%) |
| water | 199,567,816 | 64,309,668 (**32.22%**) | 44,809,675 (22.45%) | 28,739,040 (14.40%) |
| agricultural | 487,082,702 | 155,414,274 (**31.91%**) | 117,814,134 (24.19%) | 70,119,725 (14.40%) |
| barren | 74,383,133 | ~18,595,783 (**25.0%**) † | _TBD_ | _TBD_ |
| road | 79,590,500 | ~18,464,996 (**23.2%**) † | _TBD_ | _TBD_ |
| building | 122,805,791 | 22,987,367 (**18.7%**) | _TBD_ | _TBD_ |

† derived from the §7.6 error budget (percentage rounded to 1 dp); exact counts in
`~/outputs/week2_tau0.5/discard_per_class.csv`.

Agricultural alone loses **155 million pixels** at τ=0.5 — the largest absolute contributor, and
the most abundant real class in the dataset. The three worst classes by rate (forest, water,
agricultural) are precisely the three with the largest positive P−R gaps in §5, closing the loop
between the two measurements.

### 7.4 The distribution is bimodal, not a smooth tail

At τ=0.1: **958 tiles below 1%** discard, **55 tiles at exactly 100%**, comparatively little in
between. The baseline either works on a tile or collapses on it entirely.

This is why mean (12.06%) and median (0.39%) diverge so violently at τ=0.1, and it reshapes the
contribution: **large gains on a hard subset**, not small gains everywhere. Both statistics must
be reported — a reader shown only the mean will infer broad degradation that isn't there.

At τ=0.5 the median is 18.51%, so at the actual operating point the problem is genuinely broad
*and* carries a catastrophic tail. **Two distinct failure modes, with evidence for both.**

### 7.5 Decision

✅ **Thesis premise confirmed.** Proceed with the co-occurrence prior direction. No pivot to the
medium-resolution / domain-gap angle is warranted.

*(Original decision rule, retained for the record: high discard rate with substantial real-class
content → proceed; low discard rate (<5%) → pivot toward the medium-resolution / domain-gap angle,
cf. GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3.)*

### 7.6 Error budget per class — the two mechanisms, separated ⭐

Each class's GT pixels decompose into three outcomes at τ = 0.5:

| Class | GT px | Correct | → background | → other real class | discard : confusion |
|---|---|---|---|---|---|
| building | 122,805,791 | 78.6% | **18.7%** | 2.7% | 6.9 : 1 |
| road | 79,590,500 | 70.5% | **23.2%** | 6.3% | 3.7 : 1 |
| agricultural | 487,082,702 | 62.0% | **31.9%** | 6.1% | 5.2 : 1 |
| water | 199,567,816 | 54.7% | **32.2%** | 13.1% | 2.5 : 1 |
| barren | 74,383,133 | 53.9% | 25.0% | **21.1%** | 1.2 : 1 |
| forest | 125,615,647 | 44.7% | **34.6%** | **20.7%** | 1.7 : 1 |

*(The `Correct` column reproduces §5's `Acc` exactly — a third independent consistency check.)*

**Three findings.**

**1. Discard dominates confusion for every single class.** Aggregate: ~29.7% of real-class pixels
assigned to background versus ~9.9% confused with another real class — a **3:1 ratio**. The
method targets the larger of the two problems by a wide margin. This is the answer to "why not
just improve the classifier?": the classifier is mostly not wrong, it is silent.

**2. Two distinct failure profiles, requiring different treatment.**
- *Discard-limited*: **building (6.9:1)** and **agricultural (5.2:1)**. When SAM 3 fires it is
  right; it simply does not fire often enough. Recovery alone fixes these — the co-occurrence
  term needs only to supply a plausible label, not to arbitrate between competitors.
- *Both-limited*: **barren (1.2:1)** and **forest (1.7:1)**. These lose ~21% to real-class
  confusion on top of their discards, so recovery must be *discriminative*. This is where the
  signed-PMI exclusion evidence (`ANALYSIS.md` §4.2) earns its place.

**3. Forest is the worst class overall** at 44.7% correct, losing more than half its pixels
across both mechanisms. It is also one of the six sign-flipping pairs in `ANALYSIS.md` §4.4.
**If the method improves forest materially, that is the headline per-class result.**

**The honest ceiling.** Perfect recovery of all background-assigned pixels — with no new false
positives — would move building 78.6% → 97.3% and forest 44.7% → 79.3%. Large, but bounded.
State this in the paper before a reviewer computes it.

> **Terminology caution:** `→ background` here counts every real-class pixel predicted as
> background, which is a superset of "fell below τ". The two coincide numerically for the three
> classes cross-checked against `discard_per_class.csv` (water 32.22%, forest 34.60%,
> agricultural 31.91%), but write "assigned to background" rather than "discarded by τ" unless
> the identity has been verified for every class.

## 8. Confusion Matrix Analysis ✅

Source: `~/outputs/week2_tau{0.5,0.3,0.1}/confusion_matrix.npy`, 7×7, rows = true,
columns = predicted. Orientation verified: `C[agricultural, background]` = 155,414,274, matching
`discard_per_class.csv` exactly.

### 8.1 Top real-class confusions (background excluded on both sides)

These — not the `→ background` cells — are the pairs a semantic co-occurrence prior can
**arbitrate**, as opposed to the discards it must **recover**.

| Rank | True → Predicted | τ=0.5 | τ=0.3 | τ=0.1 |
|---|---|---|---|---|
| 1 | **forest → agricultural** | 23,765,826 | 34,341,313 | 41,212,112 |
| 2 | **water → agricultural** | 19,332,270 | 24,927,270 | 28,587,550 |
| 3 | **agricultural → barren** | 16,105,731 | 19,131,204 | 31,057,381 |
| 4 | **barren → agricultural** | 13,564,250 | 18,862,728 | 20,805,928 |
| 5 | agricultural → forest | 7,859,015 | 10,288,842 | 12,559,116 |
| 6 | water → barren | 4,148,715 | 5,479,005 | 7,430,931 |
| 7 | road → forest | 2,627,343 | 3,221,434 | — |
| 8 | agricultural → road | 2,139,670 | — | — |

**The ranking is stable across all three thresholds.** These are structural properties of the
model, not artefacts of where τ happens to sit — which makes them a legitimate design target.

**Three observations that change the method design:**

**(a) The top confusions are exactly the domain-unstable pairs.** `ANALYSIS.md` §4.1 measured
forest–agriculture at **+1.17 PMI in urban but −1.62 in rural**, and water–barren at **−0.22
urban / +1.70 rural** — both among the six sign-flippers of §4.4. So the single largest real-class
confusion in the baseline is a pair where a global M would actively mislead on one domain or the
other. **This is the strongest available justification for the hierarchical
`M_eff = λ·M_global + (1−λ)·M_image`** — far more specific than the generic argument in §4.4.
Cite these counts in the method section.

**(b) The confusions are directional, so M should be too.** `water → agricultural` is 19.3M at
τ=0.5 while `agricultural → water` does not appear in any top-8. `forest → agricultural` (23.8M)
runs 3× its reverse (7.9M). `ROADMAP.md` Week 7 lists "directed or symmetric?" as open, with the
current script symmetrising. **Resolve in favour of directed** — a symmetric M cannot express
"a low-confidence region bordering water is probably not agricultural" independently of the
converse.

**(c) `agricultural` is a prediction-side attractor.** It appears in four of the top five
confusions, as the predicted class in three of them. At 44.7% of real-class pixels (§3) it is the
majority class, and ambiguous predictions drift toward it. This is the hub problem `ANALYSIS.md`
§4.3 identified for `road` on the *neighbour* side, appearing here on the *prediction* side —
verify the discriminability weighting addresses both.

### 8.2 Where the 5.54 mIoU actually goes ⭐

Tracking the agricultural column across the sweep isolates the cost of threshold relaxation:

| τ | agricultural → background (missed) | background → agricultural (hallucinated) |
|---|---|---|
| 0.5 | 155,414,274 | 91,125,950 |
| 0.3 | 117,814,134 | 167,098,884 |
| 0.1 | 70,119,725 | **238,983,179** |

Going 0.5 → 0.1 **recovers 85,294,549 agricultural pixels and creates 147,857,229 new false
positives — a ratio of 1.73 wrong for every 1 right.**

In aggregate the picture is worse. Background totals 615,250,682 pixels (§3); at τ=0.1 the
visible `background → X` rows already sum to ~439M, meaning **over 70% of all true-background
pixels are misassigned to some real class.** At τ=0.1 the model has effectively stopped
predicting background at all.

> **This is the paper's motivating result, stated precisely:** the discarded residual is
> recoverable in principle, but recall bought by threshold relaxation is paid for at worse than
> one-to-one in precision, because a scalar threshold carries no information about *which* class
> a region plausibly is. A co-occurrence prior conditions on neighbourhood semantics and can
> therefore recover selectively — that is the whole claim, and it is now quantified rather than
> asserted.

## 9. Failure Cases

- `demo.py` smoke-test output saved to `SegEarth-OV-3/seg_pred.png` (OpenEarthMap sample)

### 9.1 Worst tiles (τ = 0.1, 100% of real-class pixels discarded)

| Tile | real_px | discarded_px | % |
|---|---|---|---|
| 3031 | 1,048,576 | 1,048,576 | 100.0 |
| 3003 | 1,004,599 | 1,004,599 | 100.0 |
| 2752 | 916,226 | 916,226 | 100.0 |
| 3175 | 898,188 | 898,188 | 100.0 |
| 2994 | 894,700 | 894,700 | 100.0 |
| 2625 | 845,017 | 845,017 | 100.0 |

Tile 3031 has **every pixel** in the tile carrying a real class, and every one discarded. These
are genuine total failures, not small-denominator artefacts — only tile 3480 (`real_px` = 1048)
is small enough to dismiss as noise.

### 9.1a Qualitative panels ✅

The mmseg registry error is fixed (`import segearthov3_segmentor` before `init_model`). Six
four-panel figures — image / GT / SegEarth-OV3 @ τ=0.5 / discard mask — are committed at
`docs/25{22,23,24,25,26,27}.png`.

**They show something the aggregate numbers do not: the discard has two distinct morphologies.**

| Tile | Discard | Shape of the discard |
|---|---|---|
| `2525` | **0.9%** | healthy tile; one small barren region missed wholesale |
| `2524` | **19.0%** | **whole contiguous regions** dropped — large water blocks |
| `2522` | **21.0%** | **thin seams along class boundaries** + building outlines; region interiors intact |

This distinction is load-bearing and is **not yet quantified**. A region-level co-occurrence
prior assigns labels to *regions*; boundary-seam pixels are the seams *between* regions and have
no atom to be assigned to. So the 323,184,908-pixel residual is an **upper bound** on what this
method can address, not an estimate of it.

> **Open — do before Week 8.** Erode each GT class region by k pixels and re-measure the discard
> rate split into interior vs. boundary band, as a function of k. That converts "29.68%
> discarded" into "X% addressable by a region-level method" — the number the paper actually
> needs. If the interior fraction is high, this is a *stronger* claim than 29.68%, because it is
> the fraction the method can genuinely reach. Cheap: pure numpy once the §11 `.npz` cache exists.

### 9.2 Root cause — presence-head collapse

Probed tile **3487** with `sam3_smoke_test.py --raw`, all six real classes:

| Class | S_pres | `semantic_seg` max logit | → sigmoid | Ceiling on P_final | Instances returned |
|---|---|---|---|---|---|
| building | 0.1309 | +6.44 | 0.998 | 0.131 | 0 |
| road | 0.0757 | **+10.13** | 1.000 | 0.076 | 0 |
| water | 0.0298 | +2.28 | 0.907 | 0.027 | 0 |
| barren | 0.0481 | +2.77 | 0.941 | 0.045 | 0 |
| forest | 0.0094 | +5.03 | 0.993 | 0.009 | 0 |
| agricultural | 0.0200 | +5.44 | 0.996 | 0.020 | 0 |

Since `P_final = P_fused · S_pres`, the presence score is a **hard ceiling on every pixel in the
tile** for that class. Road is the clearest case: the semantic head emits a +10.13 logit —
effectively certain the class is present — and the presence gate crushes the tile's best
achievable score to 0.076. Nothing survives even τ=0.1, which is exactly why this tile reports
100% discard.

**The mechanism is not visual ambiguity.** `P_sem` is confident. The dense evidence exists and is
being destroyed multiplicatively by a single global scalar.

Two consequences:

1. This is a **second failure mode**, distinct from the low-confidence residual, and it accounts
   specifically for the 100%-discard tail in §7.4.
2. It is **favourable for the method**. Had these tiles failed because `P_sem` was genuinely
   uncertain, there would be no signal to recover. Instead the signal is present and suppressed —
   and a co-occurrence prior, which aggregates *local neighbour* evidence, is a natural mechanism
   for overriding a *global* scalar that is wrong.

### 9.2a Generalised to all 1669 tiles ✅ — *21 Aug*

The n=1 caveat is resolved. The instrumented run
(`~/outputs/week2_tau0.5_instrumented`, τ=0.5) records per-class `S_pres` for every tile.
`spres_max` = highest presence score over the six real classes, max across sliding-window crops.

| Tile set (τ=0.5) | n | mean `spres_max` | median | p90 |
|---|---|---|---|---|
| **catastrophic** (≥99% discard) | 198 | 0.3125 | **0.2734** | 0.5699 |
| **healthy** (<1% discard) | 77 | 0.8886 | **0.9180** | 0.9672 |

**Correlation(`spres_max`, discard %) = −0.750 over all 1669 tiles.**

The separation is decisive: the **p90 of the catastrophic set (0.5699) lies below the median of
the healthy set (0.9180)** — even the best-presence catastrophic tiles score worse than a typical
healthy one. Tile 3487 was not a fluke.

Two refinements to how §9.2 must be described:

1. **It is a gradient, not a switch.** Catastrophic tiles average 0.31, not the ≲0.2 originally
   guessed. Tile 3487 (`S_pres` 0.0094–0.1309) sits at the *extreme* end. Present it as an
   illustrative worst case, never as the typical one.
2. **Quote the threshold.** 198 catastrophic / 77 healthy are **τ=0.5** counts. §7.4's 55 / 958
   are **τ=0.1**. Both correct; neither is meaningful without its τ.
3. **Two different code paths — say so.** `sam3_smoke_test.py`, which produced the §9.2 table,
   does a **single whole-image forward**. The eval path runs **sliding-window** inference, so
   `_inference_single_view` is called once per *crop* and there is one `S_pres` per crop, not
   per tile. The figures above take max-over-crops; `per_image_presence.csv` also carries the
   mean, and the `.npz` cache keeps the full `(n_views, n_cls)` array. Tile 3487's numbers and
   the 1669-tile distribution are therefore *not* measurements of the same object. Do not
   present them as one series.

> ### ⚠️ The correlation is partly mechanical — do not over-claim it
>
> `P_final = P_fused · S_pres`, and a pixel is discarded iff `P_final < τ`. **Low `S_pres`
> mechanically forces discard.** So `spres_max` and discard % would correlate even if the
> presence head were perfectly calibrated and those tiles were simply hard.
>
> The claim "presence gating destroys recoverable tiles" needs two legs:
>
> | | Claim | Status |
> |---|---|---|
> | (a) | `S_pres` is low on catastrophic tiles | ✅ **proved, n=1669** (this section) |
> | (b) | `P_fused` was *good* on those tiles anyway | ⚠️ **proved only for tile 3487, n=1** (§9.2) |
>
> Without (b), the competing reading survives: *hard tiles are hard, and low presence is a
> symptom rather than a cause.* **A reviewer will raise this.** State it before they do.

### 9.2b The counterfactual — pending

`measure_discard_rate.py --no-presence` disables the `S_pres` multiply (`P_final = P_fused`) and
re-measures. One 25-min run converts the correlation above into a causal claim:

| Outcome on the 198 catastrophic tiles | Reading |
|---|---|
| discard collapses, IoU jumps | **Presence gating caused it.** Leg (b) established at scale. This is the strongest result available in the project — and it argues *for* the method, since a local co-occurrence prior is the natural correction for a wrong *global* scalar (`ANALYSIS.md` §3.5). |
| they stay bad | Those tiles are genuinely hard; `S_pres` was a symptom. §9.2 scopes down to "an illustrative failure case", and a claim that would not have survived review is avoided. |

Expect **overall mIoU to fall** — presence gating exists because it helps on average
(SegEarth-OV3 Fig. 3). That is not a refutation. The question is not whether gating is net
positive, but what it costs on the tail.

```bash
cd ~/SegEarth-OV-3
nohup python ~/FreeTraining-OVSS/scripts/measure_discard_rate.py \
  --tau 0.5 --no-presence --no-cache \
  --out ~/outputs/week2_tau0.5_nopresence \
  > ~/logs/week2_tau0.5_nopresence.log 2>&1 &
```

Compare per-tile against `~/outputs/week2_tau0.5_instrumented/per_image_discard.csv`, restricted
to the 198 catastrophic tiles — **not** on aggregate mIoU, which answers a different question.

⚠️ **Scope risk, decide before Week 8.** The method labels unidentified regions by conditioning on
the labels of *identified* neighbours. On a 100%-discard tile there are **no identified patches** —
no seeds, an empty `M_image`, no neighbour labels. `M_global` alone cannot place a label without
an anchor. The 55 catastrophic tiles may be unreachable unless the method can bootstrap from
presence-corrected evidence.

## 10. Open Issues / Blockers

- [x] ~~mmcv not installed~~ — resolved via torch 2.4.1 + prebuilt mmcv 2.2.0 wheel
- [x] ~~`forest` IoU = 0.00 on the mini run~~ — resolved, but **the stated reason was wrong and
      is corrected here (21 Aug).** The old explanation — "the 20-image subset simply contained
      no class-6 pixels" — generalised from a single mask (`2522.png`, values `[1 2 3 4 5 7]`)
      to the whole subset. The instrumented smoke run shows the subset holds **1,258,983 forest
      pixels**, of which **96.46% are discarded to background**. Forest was present all along
      and almost entirely thrown away. The `forest,tree` prompt works; forest reaches **33.78
      IoU** on the full 1669 images. *Lesson: one mask is not a subset.*
- [x] ~~Peak VRAM not measured~~ — **6115 MB peak** (baseline), 8534 MiB (diagnostic)
- [x] ~~Discard rate unmeasured~~ — **29.68% at τ=0.5**, §7
- [x] ~~Confusion matrix unanalysed~~ — §8, all three τ
- [x] ~~mmseg `MMCV_MAX` patch is a site-packages edit~~ — **captured** in
      `scripts/setup_env.sh:40-43`; survives an env rebuild via that script.
- [x] ~~`KeyError: SegEarthOV3Segmentation is not in the mmseg registry`~~ — **fixed**
      (`import segearthov3_segmentor` before `init_model`). §9.1a figures produced and committed.
- [x] ~~`sam3_smoke_test.py` stranded in a scratch directory~~ — **tracked** at
      `scripts/sam3_smoke_test.py` (commit `e43a49b` added the `--raw` flag).
      `scripts/cooccurrence_gt.py` is also tracked (`c1bac57`), so `ANALYSIS.md` §4's
      "reproducible from this repo" claim holds for §4.
- [x] ~~`ANALYSIS.md` §3.5 contradicted by measurement~~ — **corrected 21 Aug**; §3.5 now
      documents presence-head collapse as a second failure mode.
- [x] ~~`ROADMAP.md` Week 7 "directed or symmetric?"~~ — **closed as directed**, 21 Aug.

- [x] ~~`measure_discard_rate.py` is not in version control~~ — **committed 21 Aug**, together
      with `reference/{segearthov3_segmentor.py, cfg_loveda.py, cls_loveda.txt}`, which also
      pins the exact baseline code that produced 47.38. §7–§9 are now reproducible from the repo.
- [x] ~~Instrument for per-class `S_pres` + `.npz` cache~~ — **done 21 Aug.**
      **Validation gate passed exactly:** mIoU **47.37**, discard **323,184,908 (29.68%)**,
      per-image mean/median/max **33.79 / 18.51 / 100.00** — every figure identical to the
      pre-instrumentation run. The patch is observation-only. Cache written for all 1669 tiles,
      so every future τ and ablation is now a numpy pass, not a 25-min encoder run.

### Still open

- [ ] **§9.2b counterfactual** — `--no-presence` run. Converts the −0.750 correlation into a
      causal claim, or scopes §9.2 down honestly. One 25-min run. **Highest value remaining.**
- [ ] τ=0.3 mIoU and headline discard % not yet transcribed into §7.2 (values in the CSVs).
- [ ] Exact per-class discard counts for road / barren at all τ, and building at τ=0.3/0.1
      (§7.3) — values in the CSVs, not yet transcribed.
- [ ] **§9.1a boundary-vs-interior decomposition** — the addressable-residual number. Do before
      Week 8.
- [ ] **`ANALYSIS.md` §4 PMI uses a mismatched null model.** `P_obs` is a *boundary*-frequency
      distribution; `P_exp = outer(p, p)` is built from *area* marginals
      (`scripts/cooccurrence_gt.py:118-131`). High-perimeter classes are systematically inflated,
      low-perimeter ones deflated, independent of semantics. The premise (1.3–1.7 bits vs a 0.004
      floor) is far too large to be at risk, but the **per-pair** values are — and §4.3's
      "road is a hub" finding is derived from precisely the thinnest, highest-perimeter class,
      then used to justify the discriminability weighting. The existing random control does not
      catch this: scattering classes uniformly destroys blob geometry, so it tests "is there
      signal", not "is this pair's signal geometric or semantic". **Fix:** structure-preserving
      permutation null — keep each image's mask geometry, permute class labels across regions,
      ~100 draws → per-pair z-scores with confidence intervals. Cheap, CPU-only, and strictly
      better for the paper than raw bits.

## 11. Next Steps

1. ~~Resolve mmcv installation~~ ✅
2. ~~Verify imports: `torch, mmcv, mmseg, sam3`~~ ✅
3. ~~SAM 3 checkpoint~~ ✅ (3.45 GB, symlinked into `weights/sam3/`)
4. ~~Smoke test with `demo.py`~~ ✅
5. ~~Prepare + verify LoveDA val directory~~ ✅ 1669/1669
6. ~~Trial run on 20 images~~ ✅ mIoU 38.97, 0.87 s/img
7. ~~Full evaluation on 1669 val images~~ ✅ **mIoU 47.38 (paper: 47.4)**
8. ~~Week 2: discard-rate diagnostic~~ ✅ **29.68% at τ=0.5**
9. ~~Week 2: τ-sweep (0.3, 0.1)~~ ✅ **−5.54 mIoU to recover ⅔ of the residual**
10. ~~Week 2: confusion matrix analysis~~ ✅ **§8 — directional confusions, 1.73:1 recovery cost**

11. ~~Correct `ANALYSIS.md` §3.5; close directed/symmetric in `ROADMAP.md`; produce §9.1
    figures~~ ✅ **done 21 Aug** (commit below)

**Remaining, in order.** Rationale for the ordering: *save the work, then make experiments cheap,
then validate the two shaky claims, then build.*

1. 🔴 **Commit `measure_discard_rate.py`, `cfg_loveda.py` and the summary CSVs.** No GPU, ~15 min.
   Until this is done the project's empirical core exists on one untracked filesystem. Everything
   below is lower priority than this.
2. Fill the last `_TBD_` fields in §7.2 and §7.3 from those CSVs — **no re-running required.**
3. **Instrument `measure_discard_rate.py` to dump per-class `S_pres` per image, and add an
   `.npz` cache of `(conf, pred, gt, S_pres)` in the same edit** (`INSTRUMENTATION_PATCH.md`).
   One re-run at τ=0.5 then yields: the presence distribution over all 1669 tiles, the
   catastrophic-vs-healthy comparison that generalises §9.2 from n=1, and a cache making every
   future τ value and ablation a sub-minute numpy pass (~2.5 GB for the split).
   **Validation gate: the instrumented run must still reproduce 47.37 and 29.68%** — if either
   moves, the patch changed behaviour. ~25 min, one GPU run.
4. **Two validation experiments, CPU-only once the cache exists** — both target claims that are
   currently load-bearing and unvalidated, and both are expensive to unwind after the method is
   built on top of them:
   - §9.1a boundary-vs-interior decomposition → the addressable-residual number.
   - §10 permutation null for PMI → per-pair z-scores instead of raw bits.
5. Begin Week 3: `M_global` construction against the GT co-occurrence reference
   (`ANALYSIS.md` §4), with §8.1's confusion ranking as the validation target — a useful M should
   assign low compatibility to exactly the pairs the baseline confuses.

---

## 12. Framing note for the paper

With 29.68% of real-class pixels discarded, the headroom is enormous — and so is the reader's
expectation. **Two numbers must be reported separately:**

- **Recovery rate** — what fraction of the discarded residual the method labels at all.
- **Precision of recoveries** — what fraction of those labels are correct.

§8.2 is the proof that this separation is necessary: threshold relaxation recovers 85.3M
agricultural pixels while creating 147.9M false positives, **1.73 wrong per 1 right**.
**Recovering pixels carelessly is worse than not recovering them.** Any headline gain claimed
without the precision column invites exactly the objection the τ-sweep already answers — and a
reviewer who has read SegEarth-OV3 will raise it.

The strongest table remains the one specified in `ANALYSIS.md` §6: **mIoU restricted to pixels
SegEarth-OV3 assigns to background.** That isolates the contribution and cannot be confused with
backbone effects. §7 gives that table its denominator: 323,184,908 pixels.

Report the **median alongside the mean** everywhere (§7.4). The bimodality is a real property of
the problem, and disclosing it is more persuasive than a mean that overstates the typical case.

Lead the results section with §7.6's **3:1 discard-to-confusion ratio**. It establishes in one
number that the baseline's dominant error mode is silence rather than error, which is precisely
the gap the method fills.

---

## References

- SegEarth-OV3 — arXiv:2512.08730 · `github.com/earth-insights/SegEarth-OV-3`
- SegEarth-OV — CVPR 2025
- ConInfer — arXiv:2603.29271 (closest related work; context-at-inference for OVRSS)