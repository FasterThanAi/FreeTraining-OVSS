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


---

# ⛔ §7.1a on OpenEarthMap — it does NOT transfer, and the reason replicates

Gates passed: instrumented run **29.90** exactly, `conf` in **[0.1042, 0.9195]**.
Fitted at their OEM threshold (`prob_thd = 0.1`).

**Five-fold: −0.39 ± 1.28, one fold of five positive. Null.**

| | published τ | fitted | Δ |
|---|---|---|---|
| full mIoU | 29.90 | 29.81 | **−0.09** |
| catch-all-excluded | 31.51 | **32.78** | **+1.27** |
| `background` | 17.00 | **6.06** | **−10.94** |

The full-mIoU movement decomposes as **+1.13 from the eight real classes** and
**−1.22 from `background` alone**. And almost nothing else moves: `grass` −0.00,
`pavement` +0.00, `road` +0.02, `tree` +0.02, `building` +0.04. The fit changed one
threshold that mattered (`bareland` +8.18, a class that is 1.28% of the scene) and
paid for it with the catch-all.

## ⭐ Why this is a good negative: it replicates our own OEM result on a different model

| OpenEarthMap | SAM 3 (ours) | ConInfer (CLIP) |
|---|---|---|
| Δ full mIoU | +0.30 | −0.09 |
| Δ excl. catch-all | +1.75 | +1.27 |
| `background` IoU | 17.13 → **5.83** | 17.00 → **6.06** |

⭐ **Two unrelated models, and the fit drives the catch-all to almost the same
value.** That is the dataset's metric pathology, not a property of either pipeline:
OpenEarthMap's `background` sits near 17 IoU with poor precision and owns 11.1% of
a nine-class mean, so an optimiser told to maximise land cover trades it away and
full mIoU cannot move. §9b said exactly this about OEM — *"you can improve land
cover or the metric, not both"* — and it now holds for a competitor's pipeline too.

## The claim, correctly scoped

⛔ **Not** *"per-class τ transfers across backbones."* The honest 2×2:

| | LoveDA | OpenEarthMap |
|---|---|---|
| **SAM 3** | **+1.18 ± 0.45** ✅ | +0.30 *(land cover +1.75)* |
| **ConInfer (CLIP)** | **+2.51 ± 0.34** ✅ | −0.09 *(land cover +1.27)* |

> **The correction is not backbone-specific — it is dataset-specific, and the axis
> is the catch-all's calibration.** Where the catch-all is competently predicted
> (LoveDA, ~45 IoU) the fit improves land cover *and* the headline, on both
> architectures. Where it is pathological (OpenEarthMap, ~17 IoU) the fit improves
> land cover on both and the headline on neither, because the catch-all it must
> sacrifice owns a ninth of the metric.

That is a **stronger and more falsifiable** statement than "it transfers", and it
ties the method back to the paper's own mechanism instead of standing beside it.

⭐⭐ **THEIR OEM THRESHOLD CANNOT FIRE — measured 4 Sep, and it reframes this row.**
`reachability.py` finds mechanism (A) is **exactly zero** across all 384 tiles: the cache's
`conf` floor is **0.1042** against their published `prob_thd` of **0.1**, so the threshold
sits *below the score floor* and never fires on a single pixel. Every catch-all assignment
on that dataset is an argmax loss.

⚠️ **So this is not the like-for-like test the row implies.** Their OEM baseline is an
**un-thresholded argmax**, which means our fit could only ever *raise* thresholds, never
lower them — the opposite regime from LoveDA, where τ=0.8 discards 20.7% and 89.4% of that
is threshold-reachable. Say this beside the −0.39 rather than letting it read as "the
method does not transfer here".

⭐ It is also a finding about a published configuration in its own right, and a reviewer
can verify it from their own released code: a tuned hyperparameter that has no effect on
the dataset it was tuned for. Report it factually, without editorialising — the likeliest
explanation is that `prob_thd` was carried over from a setting where it did bite.

⚠️ **Caveat that must travel with this row:** the OEM reproduction is off by 12.05
against their published figure. The delta is robust to a constant offset, but this
row rests on far weaker footing than the LoveDA one (−2.34), and should be labelled
supporting evidence rather than an independent confirmation.

⚠️ `tau_cv.py`'s verdict again printed SAM 3 boilerplate — *"the single-split +1.44
was a favourable draw"* — which refers to a LoveDA number and is meaningless here.
Third time generated verdict text has needed checking against its own table.
