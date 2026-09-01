# ConInfer comparison — measured 1 Sep 2026

## ⛔ VALIDATED, AND THE REPRODUCTION IS IMPERFECT — read this first

Their published figures (Table 1 of arXiv:2603.29271):

| | published | our reproduction | gap |
|---|---|---|---|
| **LoveDA** *(full official split, 1669 tiles)* | **39.33** | 36.99 | **−2.34** |
| **OpenEarthMap** *(our 384 of 500)* | **41.95** | 29.90 | ⛔ **−12.05** |

**LoveDA is usable, OpenEarthMap is not.** The split that reproduces closely is the
one we evaluate in full; the one that fails badly is the one we hold only 77% of.
That is not a coincidence, and it decides which comparison goes in the paper.

⛔ **Drop OpenEarthMap from the ConInfer comparison.** A 12-point gap on a subset we
cannot complete is not a measurement.

✅ **Report LoveDA with BOTH numbers** — theirs and ours — so the reader sees the
reproduction gap rather than trusting whichever figure suits us:

| method | LoveDA mIoU |
|---|---|
| ConInfer (published) | 39.33 |
| ConInfer (our reproduction) | 36.99 |
| SegEarth-OV3 (our baseline) | 47.37 |
| **+ per-class τ (ours)** | **48.53** |

The conclusion holds under either ConInfer figure.

### What was eliminated as the cause

| hypothesis | verdict |
|---|---|
| label encoding | ⛔ values are `uint8` 0–8 with `reduce_zero_label=False`; correct |
| class-name order | ⛔ their prompts match our label values exactly |
| image format | ⛔ RGB 8-bit 1000×1000, no 16-bit or alpha |
| **`feature_up=False`** | ⛔ setting `True` changes **nothing** — identical to the digit, so the flag does not touch the `predict()` path |
| alphabetical subset bias | ⛔ 73 cities, `aachen`→`zanzibar`, all continents |

What remains — the missing 116 OpenEarthMap tiles, torch 2.4.1 vs their 2.7.1, a
CLIP weight revision — is unverifiable without the full split. **We stopped here
deliberately** rather than spend the timebox on it.

### ⭐ Why this does not block §7.1a

Applying per-class τ to ConInfer measures a **delta on our own run**, and a delta is
robust to a reproduction offset. If per-class τ adds +X to ConInfer's scores as we
measured them, that is a valid statement about the method generalising to a CLIP
backbone whether their absolute is 36.99 or 39.33. **The experiment worth more than
the comparison row is unaffected by the comparison row's imperfection.**

## What was run

Their code, their configs, their hyperparameters, on **our exact tiles** — the same
1669 LoveDA and 384 OpenEarthMap images behind every other number in this project.
Only the dataloader batch size was overridden (8 instead of 50, for one 16 GB card);
nothing about the method, the thresholds or the data was altered.

| | LoveDA | OpenEarthMap |
|---|---|---|
| **ConInfer** | **36.99** | **29.90** |
| SegEarth-OV3 (our baseline) | 47.37 | 44.16 |
| **+ per-class τ (ours)** | **48.53** | **44.47** |
| *gap, ours − ConInfer* | *+11.54* | *+14.57* |

Iterations confirm the splits: 209 × 8 ≈ 1669, and 48 × 8 = 384.

### Per class

| LoveDA | ConInfer | ours (fitted) | | OpenEarthMap | ConInfer | ours (fitted) |
|---|---|---|---|---|---|---|
| background | 24.60 | 45.48 | | background | 17.00 | 5.83 |
| building | 47.89 | 64.08 | | bareland | 9.32 | 15.68 |
| road | 36.10 | 53.99 | | grass | 22.12 | 43.27 |
| water | 53.72 | 58.06 | | pavement | 16.67 | 30.59 |
| barren | 18.73 | 36.70 | | road | 22.89 | 48.82 |
| forest | 29.38 | 34.14 | | tree | 47.88 | 63.95 |
| agricultural | 48.54 | 47.27 | | water | 45.38 | 65.81 |
| | | | | cropland | 44.44 | 46.58 |
| | | | | building | 43.41 | 79.67 |

## ⛔ The open gate

**Their published LoveDA and OpenEarthMap mIoU are not yet known to us**, so we
cannot say whether 36.99 and 29.90 are a faithful reproduction or a broken setup.
⛔ **Nothing above goes in the paper until that check passes.** Publishing a
competitor 10+ points below their claim, unverified, is the worst available
outcome — worse than reporting no comparison at all.

**Plausibility, pending that check.** ConInfer claims **+2.80 over SegEarth-OV**,
which is CLIP ViT-B/16. If SegEarth-OV scores ≈34 on LoveDA and ≈27 on
OpenEarthMap, then ≈37 and ≈30 land exactly on their claim. That is consistent,
but it is arithmetic on a number we have not read.

## ⭐ The framing this forces, and it is important

**ConInfer and our method improve different backbone generations.** They are
CLIP ViT-B/16 at 448²; we build on SAM 3 at 1024². Both are training-free
open-vocabulary segmentation, both evaluated on identical tiles against identical
ground truth — so the comparison is legitimate as *state of the art on this task*,
but it is **not** an apples-to-apples method comparison, and the caption must say
so rather than let the reader assume otherwise.

The honest reading: **most of our margin is the backbone, not our contribution.**
Our contribution is the +1.16 from per-class thresholds on top of SegEarth-OV3 —
not the 11.54 gap to ConInfer. Presenting the gap as if it were ours would be the
same category of error as OpenEarthMap's +2.28 in §8.1, and we should not make it
after building a whole paper on catching it.

## Also worth reporting

- **Their `prob_thd` is 0.8 (method) and 0.3 (baseline configs) on LoveDA, 0.1 on
  OpenEarthMap.** They tune the same global threshold this paper is about, which
  supports the framing that a single confidence threshold is a real design choice
  in this family rather than an incidental detail.
- **Speed:** 0.58 s/iter at batch 8, ~2 minutes for LoveDA, against ~24 minutes for
  SAM 3 at 1024². Their approach is an order of magnitude cheaper at inference,
  which belongs in the comparison alongside accuracy.
- **Deployment cost differs in kind:** ConInfer needs a licence-gated
  satellite-pretrained DINOv3 ViT-L/16 (1.13 GB) in addition to CLIP; ours needs
  ~200 labelled tiles and no additional weights.

## Reproducibility notes (all verified, all resolved)

1. `segearth_segmentor.py` is **absent from their repo**; `eval.py` imports it at
   line 4. Copied from upstream `likyoo/SegEarth-OV`.
2. Three dependencies are declared by neither their requirements nor the packages
   importing them: `ftfy`/`regex` (mmsegmentation), `psutil`
   (fast-pytorch-kmeans), `openpyxl` (their own `utils.py`).
3. `ConInfer_segmentor.py` **hardcodes the first author's filesystem** for the
   DINOv3 repo and weights. Repointed by `scripts/patch_coninfer_paths.sh`, which
   keeps a `.orig` and edits **paths only**.
4. `torch==2.7.1` is unusable here — no prebuilt mmcv wheel exists for it. Run on
   torch 2.4.1+cu121 with mmcv 2.2.0. ⚠️ **A deviation, and the reason their own
   number must be reproduced before ours is compared to it.**
5. `configs_baseline/` sets `gmm_temp=0` but `gmm_fitting` computes `1/temp`
   unconditionally → `ZeroDivisionError`. Their baseline rows appear to require
   manually renaming `predict1` → `predict`; there are three `predict*` methods and
   no dispatch between them. **Not pursued** — that is a logic edit, and validating
   against their published ConInfer number serves the same purpose.


---

# ⭐⭐ §7.1a — per-class τ TRANSFERS to a CLIP backbone

Both gates passed: the instrumented run printed **36.99**, identical to the
un-instrumented one, and `conf` was observed in **[0.1601, 0.9611]**, inside the
[0,1] grid the threshold search assumes.

Fitted at **their** operating point (`prob_thd = 0.8`), five-fold, calibration and
evaluation tiles always disjoint, baseline recomputed on the same held-out tiles.

| | SAM 3 (SegEarth-OV3) | **CLIP+GMM (ConInfer)** |
|---|---|---|
| five-fold Δ full mIoU | +1.18 ± 0.45 | ⭐ **+2.51 ± 0.34** |
| worst fold | +0.84 | **+2.22** |
| Δ catch-all-excluded mIoU | +1.36 | ⭐ **+1.94** |
| Δ catch-all IoU | −0.02 | +6.01 |
| calibration tiles for a reliable gain | ~200 | ⭐ **~25** |

**Held-out, one partition, both metrics:** 37.00 → **39.52** full, 39.06 → **41.00**
excluding the catch-all, `background` 24.61 → 30.61.

**Every class improves**, which our own SAM 3 result does not manage:
`water` +4.20, `road` +2.52, `forest` +2.04, `agricultural` +1.69, `barren` +0.78,
`building` +0.43, `background` +6.01.

## What this changes

> ⭐ **The claim is no longer about SAM 3.** Per-class thresholding is a property of
> **any pipeline that thresholds per-class scores**: +1.18 on a SAM 3 pipeline at
> τ=0.5, +2.51 on a CLIP+GMM pipeline at τ=0.8. Two architectures, two methods, two
> operating points, same correction.

That answers *"is this a SAM 3 quirk?"* — a question the paper could not answer this
morning, and the first one a reviewer would ask of a single-backbone result.

⭐ **And ConInfer stops being a competitor.** Our method *composes* with it: their
pipeline goes 36.99 → **39.52** with our calibration on top. We do not beat the
nearest published work, we improve it. That is a better answer to *"how is this
different from ConInfer?"* than any margin would have been.

## Honest notes

- ⚠️ **`background` gains +6.01, a third of the full-mIoU gain.** Not the 110%
  artefact of §8.1, and the catch-all-excluded column is independently positive at
  **+1.94** — but quote both, as everywhere else in this paper.
- ⚠️ **Measured on our reproduction (36.99), not their published 39.33.** This is a
  *delta*, which is robust to a constant offset, but say so.
- ⚠️ `tau_cv.py`'s closing verdict repeats the "LoveDA train → val gives −0.12"
  caveat. That is **boilerplate from the SAM 3 run** and was never tested here.
  Do not quote it for this result.
- ⚠️ ConInfer's baseline is *itself* depressed 2.07 points by its catch-all
  (`background` 24.61 against a real-class mean of 39.06) — the same metric
  distortion this paper documents, appearing in a competitor's published setup.
