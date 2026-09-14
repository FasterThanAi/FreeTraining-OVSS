# UAVid — baseline reproduced at 56.86, which is 2.16 ABOVE the published 54.7

**14 Sep 2026.** 70 official val frames, `cfg_uavid.py`'s own vocabulary, thresholds
(`prob_thd=0.3`, `confidence_threshold=0.3`) and 7-class `UAVidDataset` definition.

---

## 1. The numbers

| `max_side` | frame given to SAM 3 | mIoU | vs published **54.7** |
|---|---|---|---|
| 1008 | 1008x567 | **56.86** | **+2.16** |
| 1344 | 1344x756 | **57.29** | **+2.59** |

⭐ **Resolution runs the WRONG way to explain the gap.** More pixels scores *higher*
(+0.43 for 1.33x the linear size), so the full 3840x2160 frame their config implies would
land further above 54.7, not closer to it. The downscale is not inflating us -- it costs
about half a point, and `max_side=1008` is therefore a conservative operating point.

⚠️ **The gap is UNEXPLAINED.** Same code, same config, same `cls_uavid.txt`, same
thresholds, same class definition. The only free variable left is *which frames*: their
`data_prefix` reads `img_dir/test`, and UAVid's official test split has **no public
labels**, so they must have placed labelled frames under a directory named `test`.
Whether those are these 70 is not determinable from the released config.

⭐ **This does not block the work.** Everything the method claims is a **delta** measured
on our own run, which is robust to a constant offset -- the same argument that made
ConInfer usable at a −2.34 reproduction gap (@CONINFER_RESULTS.md). Report both numbers.

## 2. ✅ The labels are verified exactly, not approximately

70 label PNGs, 597,196,800 px: **0 pixels at 255 (ignore), 0 unexpected values.** Nothing
was excluded from the metric, so the mIoU is over the whole split.

⭐ **And the total is self-verifying.** 70 frames at 3840x2160 would be 580,608,000. It is
not -- the split is **40 frames at 3840x2160 plus 30 at 4096x2160**:

    8,294,400 x 40  +  8,847,360 x 30  =  597,196,800   (exact)

UAVid ships two frame widths. Every label is intact at full resolution, and the arithmetic
closing to the pixel is stronger evidence than the histogram alone.
⛔ **Consequence: `tta='d4'` is unusable here** -- it refuses non-square input by design,
and these are 16:9 and 1.896:1. `hflip` is the available TTA.

## 3. ⭐⭐ The per-class profile is the widest spread in the project

| class | GT share | IoU | P | R | **P−R** | what per-class tau should do |
|---|---|---|---|---|---|---|
| **human** | **0.19%** | 17.62 | 78.91 | 18.49 | ⭐ **+60.4** | barely fires, right when it does -> **down** |
| **tree** | 23.55% | 53.15 | 92.13 | 55.68 | **+36.5** | LoveDA `water`'s profile (+34.8) -> **down** |
| car | 1.81% | 63.28 | 86.12 | 70.46 | +15.7 | down |
| building | 33.98% | 90.79 | 95.98 | 94.37 | +1.6 | leave |
| background | 16.16% | 55.23 | 67.88 | 74.77 | −6.9 | up |
| road | 10.17% | 67.79 | 76.67 | 85.41 | −8.7 | up |
| **vegetation** | 14.15% | 50.19 | 55.20 | 84.68 | ⛔ **−29.5** | over-fires -> **up** |

*(IoU/P/R at `max_side=1008`.)*

**+60.4 to −29.5 against LoveDA's +34.8 to −12.5.** Four classes want the threshold lowered
and three raised -- the "one global tau is wrong for different classes in opposite
directions" claim stated more loudly than any dataset held so far. WEEK3 §9g measured that
the **P−R gap** ranks which classes calibration moves (rho +0.713, p 0.013) while precision
alone does not (+0.168, p 0.60), so this table is a direct prediction, not a hope.

⛔ **`human` is 0.19% of pixels and owns 1/7 = 14.3% of mIoU.** One IoU point there moves
the headline 0.14. **Never quote a UAVid mean without this table** -- it is WEEK3 §9h's
leverage argument with a rare class in the catch-all's role.

## 4. ⭐ A new interior point on §7's share axis, for free

`background` (UAVid's clutter) is **16.16%** of ground truth, landing in an empty gap:

| dataset | catch-all share |
|---|---|
| OpenEarthMap | 0.84% |
| Potsdam | 4.29% |
| ⭐ **UAVid** | ⭐ **16.16%** |
| LoveDA urban | 26.0% |
| LoveDA (pooled) | 36.1% |
| LoveDA rural | 42.9% |

WEEK3 §7c found the detectability effect **saturates by ~35%** share, with the action at the
low end and only two points (0.84%, 4.29%) below 26%. UAVid is a third.

## 5. ⛔ The blocker ahead: 70 val frames is below the calibration budget

§9b's learning curve: ~200 calibration tiles for every draw to be positive, and **below 50 it
actively hurts** (−2.14 at n=10). A 5-fold inside 70 frames leaves 56 calibrating and 14
evaluating, which is inside the harmful region, and frames drawn from the same flight
sequence are strongly correlated on top of that.

✅ **UAVid ships 600 labelled TRAIN frames** (verified: `train: 600 images, 600 labels`), so
the budget exists. ⚠️ But WEEK3 §9b's rule is **calibrate on the distribution you evaluate
on** -- LoveDA train->val gives **−0.12**. Decide the protocol before caching, not after.

⚠️ **Cache size.** These frames are ~8.5 M px against LoveDA's 1.05 M. 670 frames is 5.7 G px
against LoveDA's 1.75 G -- **3.3x the LoveDA cache** at the same per-pixel layout. Measure the
estimate before launching, per the fix in `measure_discard_rate.py`.

## 6. Engineering note worth keeping

⛔ **`predict()` ignores mmseg's `inputs` tensor and re-opens the original file from disk**
(`segearthov3_segmentor.py:404-405`, *"Load original image to preserve details for SAM3"*).
**A `Resize` in the test pipeline therefore does nothing at all to what SAM 3 sees.** Adding
one left the failing CUDA allocation at exactly 2.63 GiB with 12.9 GB resident -- an 8x
smaller input that moves no memory is the tell. `max_side` exists because that is the only
point where the frame can be shrunk; it defaults to 0 and its condition is a strict `>`, so
every recorded LoveDA and Potsdam number is untouched (`scripts/test_max_side.py`).


---

## 7. Per-class tau on UAVid — the ceiling, and a LEAK in the first 5-fold

`tau_oracle.py` / `tau_cv.py` on the 70-frame val cache at their published tau = 0.3.
✅ Cross-check passes: the histogram path recomputes the published-tau row at **56.87**,
matching `measure_discard_rate.py` and `eval.py`'s 56.86.

### The oracle ceiling — +1.51

| rung | free params | mIoU | delta |
|---|---|---|---|
| published tau = 0.3 | 0 | 56.87 | — |
| best global tau = 0.230 | 1 | 57.34 | +0.47 |
| **best per-class tau** | 6 | **58.38** | **+1.51** |

Comparable to LoveDA's **+1.46**, and the same shape: one global value is wrong for
different classes in **opposite** directions. Fitted span **0.000 to 0.700**.

| class | oracle tau | IoU before | after | delta |
|---|---|---|---|---|
| **human** | **0.000** | 17.59 | 23.82 | ⭐ **+6.23** |
| building | 0.100 | 90.76 | 91.54 | +0.77 |
| road | **0.700** | 67.88 | 68.37 | +0.48 |
| tree | 0.125 | 53.12 | 53.24 | +0.12 |
| car | 0.035 | 63.33 | 63.44 | +0.11 |
| vegetation | 0.265 | 50.17 | 50.13 | −0.04 |
| `background` | *(no effect)* | 55.22 | 58.13 | +2.91 |

⭐ **`human` alone is 59% of the ceiling (6.23 of the 10.58 IoU points, and 81%
of the real-class gain)**, and its oracle threshold is **0.000** -- the
optimum is to stop thresholding it entirely. At 62.07% of its pixels discarded on 78.91%
precision that is exactly what WEEK3 §9g predicts from the precision-recall gap (+60.4).
⭐ `road` moves the other way to **0.700**, so UAVid reproduces the opposite-directions
result on a dataset whose failure mode is small objects rather than amorphous stuff.
⚠️ `background` gains **+2.91** without a threshold of its own -- it is the residue of the
real classes' moves, not something the fit optimised (the objective is `real`).

### ⛔ The 5-fold number is CONTAMINATED — do not quote it

**+1.05 +/- 0.86, 5/5 folds positive, range +0.24 to +2.51.** Fails the project gate
(`mean − 2*sd > 0` gives **−0.67**) -- but that is not the reason to withhold it.

⛔ **UAVid val is 7 flight sequences of 10 CONSECUTIVE frames.** `tau_cv.py` split at the
frame level, so `scripts/test_group_folds.py` measures that a frame-level 5-fold straddles
**all 7 of 7 groups**: near-duplicate frames of every evaluation scene were in the
calibration set. The fit could memorise the scene it was scored on.
⛔ **The learning curve is contaminated the same way**, and more severely -- its flatness
(n=10 **+0.85**, n=25 +0.98, n=50 +0.80, no trend where LoveDA needs 200 tiles to turn
reliably positive) is exactly what leakage looks like. **Do not report "UAVid calibrates
from 10 tiles" until it is re-measured group-disjoint.**

✅ **Fixed:** `tau_cv.py --group-re` permutes GROUPS and deals them into folds whole, and
does the same for the learning curve. Absent, the flag is a no-op -- the ungrouped path is
the original permutation split verbatim and the grouping block draws no randomness, so
every recorded LoveDA / Potsdam / OEM / ConInfer number is untouched. Both properties are
asserted in `scripts/test_group_folds.py`.

⚠️ **Two honest notes for whatever the grouped number turns out to be.**
- With 7 groups a 5-fold gives folds of 20/20/10/10/10 tiles, so the spread will be *wider*,
  not narrower. UAVid is **underpowered, not null** -- the same position OpenEarthMap holds
  (384 tiles swinging 10 points against a ~1 mIoU effect).
- `road` **−1.18** across folds against an oracle **+0.48**: the fit moves road's threshold
  to 0.700 on calibration data and it does not transfer. A per-class rule can hurt a
  per-class result; quote the table, never the mean alone.

### Disk, measured

**43 MB per frame compressed** (3.0 GB for 70), against the 118.7 MB/tile uncompressed
estimate -- 2.7x pessimistic. So the 600 labelled train frames would need **~26 GB** against
~20 GB free: ⛔ not cacheable whole, ~300 frames fits at ~13 GB.
⛔ `--cache-full`, which lever 2 requires, roughly doubles the per-pixel cost and is
infeasible at 3840x2160 on this disk. Caching at the inference resolution (1008x567, a
14x reduction) is the obvious answer and is not yet implemented.
