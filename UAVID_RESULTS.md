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


---

## 8. ⭐⭐ The leak was worth +0.54 mIoU and two folds — measured, not argued

`--group-re '([a-z]+[0-9]+)(?=[-_][0-9]+$)'`. ✅ Confirmed: **7 groups from 70 tiles,
sizes 10-10.** The Kaggle repack flattened all seven val sequences into one directory
called `seq16` and renamed six of them `fileNN-k.png`, so scene identity survives in the
filename even though the directory structure lost it.

| protocol | mean | sd | folds+ | gate (`mean − 2sd > 0` and 5/5) |
|---|---|---|---|---|
| frame-level folds *(leaking)* | **+1.05** | 0.86 | **5/5** | ⛔ |
| ⭐ **group-disjoint folds** | **+0.51** | 0.95 | **3/5** | ⛔ |

⭐⭐ **Half the apparent gain was the model being scored on near-duplicates of frames it
calibrated on**, and the leak also flipped two folds from negative to positive. This is a
clean measurement of how much a correlated-tile split inflates a result, on a protocol
this project uses everywhere else, and it is worth a methods paragraph in its own right.

⭐ **The learning curve is the tell, and it reverses completely:**

| calib tiles | leaking | **group-disjoint** |
|---|---|---|
| 10 | +0.85 | ⭐ **−0.69** |
| 25 | +0.98 | +0.85 |
| 50 | +0.80 | +0.17 |

Leaking, it was flat and positive from 10 tiles — which would have been a headline
("UAVid calibrates from 10 tiles" against LoveDA's 200). Clean, it is **negative at n=10**,
exactly the shape LoveDA has (−2.14 at n=10). ⛔ **The flat curve was the leak**, and the
suspicion was recorded before the re-run.

## 9. ⚠️ Underpowered, not null — and the reason is 7 scenes, not 70 tiles

⛔ **+0.51 ± 0.95 with 3/5 folds fails the gate.** Do not report UAVid as a positive result.

But the fold **baselines** span **45.68 to 63.83 — 18 mIoU points** — against an effect of
about half a point. That is OpenEarthMap's position exactly (384 tiles swinging 10 points
against a ~1 mIoU effect), and it is what "underpowered" looks like: 70 frames is really
**7 scenes**, and a 5-fold leaves 10-20 frames from 1-2 flights to score on.

⭐ **`human` gets STRONGER under the clean protocol: +9.38** (leaking: +7.64), against an
oracle of +6.23. Its threshold move is the one thing that transfers across scenes.
⛔ **`road` −3.11**, against an oracle of **+0.48** at τ 0.700. The fit pushes road's
threshold far up on the calibration scenes and it does not transfer — WEEK3 §9e's
"thresholds are domain-specific" appearing at the level of individual **flights**.

| class | Δ IoU |
|---|---|
| **human** | ⭐ **+9.38** |
| building | +0.67 |
| tree | +0.06 |
| car | −0.89 |
| vegetation | −1.14 |
| `background` *(catch-all)* | **−1.41** |
| **road** | ⛔ **−3.11** |

⭐ **Report both metrics, per §9h.** Real classes sum to **+4.96**, i.e. **+0.83
catch-all-excluded mIoU** against **+0.51** full — the catch-all is *deflating* the
headline here, the LoveDA-urban direction rather than the OpenEarthMap one.

## 10. The decisive next test

Cache the **train** split: 600 labelled frames, and ~300 fit on disk at 43 MB/frame. If
train carries many more scenes than val's 7, a group-disjoint 5-fold there separates
"underpowered" from "null" — which 70 frames from 7 flights cannot.


---

## 11. ⭐⭐ Train and val are the SAME distribution — which LoveDA's never were

600 train frames cached at `--cache-stride 4` (7.4 MB/tile against 118.7, 4.3 GB against
34.8 -- the full-resolution cache was storing an upsample of a 1008x1008 forward pass).

| | UAVid train (600) | UAVid val (70) | gap |
|---|---|---|---|
| **discard rate** | **6.73%** | **6.81%** | ⭐ **0.08 pp** |
| `human` lost to bg | **62.47%** | 62.07% | 0.40 pp |
| `car` lost to bg | 28.92% | 22.73% | ⚠️ **6.19 pp** |
| baseline mIoU | 55.48 | 56.87 | 1.39 |
| real-class pixels | 4,191,571,414 | 500,664,688 | 8.4x |

⭐⭐ **WEEK3 §9b's train->val failure on LoveDA (−0.12) was diagnosed as "identical
background share, 2.04x different discard" (14.54% vs 29.68%). UAVid's two splits differ
by 0.08 percentage points.** So UAVid is the first dataset in this project where
calibrate-on-train / evaluate-on-val is a *fair* transfer test rather than a domain shift
in disguise -- and a committed prediction: **it should work here.** If it does not, the
rule "calibrate on the distribution you will evaluate on" needs a sharper statement than
discard-rate matching.

⚠️ `car` is the one class that moves materially between the splits, and it is already
−0.89 IoU under the fitted rule on val. Watch it.

⭐ **And the pixel budget stops being the constraint:** 4.19 G real-class pixels, **3.8x
LoveDA's entire val split** (1.09 G). Whether UAVid is underpowered is now purely a
question of how many independent SCENES those 600 frames carry, not of sample size.


---

## 12. ✅✅ LEVER 1 PASSES ON UAVid — +1.34 ± 0.39, 5/5 folds

200 real train frames (400 augmented copies excluded), **20 sequences**, folds
**group-disjoint** so no flight appears on both sides.

| | mean | sd | folds+ | **mean − 2sd** | gate |
|---|---|---|---|---|---|
| LoveDA (recorded) | +1.18 | 0.45 | 5/5 | +0.28 | ✅ |
| ⭐ **UAVid train** | ⭐ **+1.34** | **0.39** | **5/5** | ⭐ **+0.56** | ✅ **PASS** |
| UAVid val *(7 scenes)* | +0.51 | 0.95 | 3/5 | −0.67 | ⛔ |

⭐⭐ **UAVid clears the project gate more comfortably than LoveDA does**, and it is the
fourth pipeline/dataset where per-class τ works (SAM 3 on LoveDA, SAM 3 on Potsdam,
ConInfer on LoveDA, now SAM 3 on UAVid).

### ⭐ val was UNDERPOWERED, not null — the prediction was on the record and held

Going from 7 scenes to 20: **sd collapses 0.95 → 0.39** and the mean *rises* 0.51 → 1.34.
⭐ **The rise is `road`**, which was the diagnostic class: **−3.11 on 7 scenes, +0.72 on
20.** The fit was pushing road's threshold to 0.700 on one or two flights and failing to
transfer; with four flights per fold it stabilises. **So per-flight threshold specificity
is a small-sample artefact here, not a property** -- a question WEEK3 §9e could not settle
and this does.

| class | val (7 scenes) | **train (20 scenes)** |
|---|---|---|
| **human** | +9.38 | ⭐ **+6.47** |
| **road** | ⛔ **−3.11** | ✅ **+0.72** |
| car | −0.89 | +0.39 |
| vegetation | −1.14 | +0.06 |
| building | +0.67 | +0.24 |
| tree | +0.06 | +0.03 |
| `background` | −1.41 | +1.45 |

⭐ **No class loses under the well-powered protocol.** LoveDA's fit costs `road` 0.54 and
`agricultural` 0.25; UAVid's costs nothing. That is the only dataset in the project where
that is true of SAM 3.

### The ceiling is essentially reached

Oracle on the same 200 clean frames: **+1.40**. Fitted, held out: **+1.34** — **96%**.
LoveDA captures 81%, Potsdam 73%, OpenEarthMap 11%.
⚠️ **Flag the protocol mismatch rather than celebrating the number**: the oracle is fitted
and scored on the pooled 200, the CV averages five held-out sets of 40, and mIoU does not
decompose across subsets -- which is why the fitted real-class aggregate (**+7.91**) can
exceed the oracle's (**+7.61**). Read it as "the fit reaches essentially all of the
available headroom", not as a precise ratio.

### Calibration cost — 25 tiles, and now measured on group-disjoint draws

| calib tiles | mean Δ | sd | worst draw |
|---|---|---|---|
| 10 | ⛔ **−0.19** | 1.53 | −2.84 |
| **25** | ✅ **+0.84** | 0.36 | **+0.26** |
| 50 | +0.99 | 0.25 | +0.72 |
| 100 | +1.18 | 0.23 | +0.86 |

⭐ **Positive from ~25 tiles against LoveDA's ~200**, matching ConInfer's ~25. ⚠️ Because
draws take whole sequences, "25 tiles" is really **3 scenes** -- quote it that way.
⛔ And n=10 is **negative** (worst draw −2.84), reproducing LoveDA's shape. The earlier
frame-level curve said **+0.85** at n=10; that was the leak, and it is now gone.

### Both metrics agree, and the catch-all points a THIRD way

| | full mIoU | catch-all-excluded | `background` IoU |
|---|---|---|---|
| published τ | 56.48 | 56.06 | 59.03 |
| per-class τ | **57.80** | **57.29** | 60.86 |
| **Δ** | **+1.32** | **+1.23** | +1.83 |

✅ Full +1.32 and land cover +1.23 agree, so the gain is **not** a repaired catch-all.
⭐ **New direction for §9h:** UAVid's `background` sits at **59.03 against a real-class mean
of 56.06**, so the catch-all *inflates* the published headline by **0.42**. LoveDA's
deflates (45.50 against ~48), OpenEarthMap's deflates severely (17.13 against 47.54).
**Three datasets, and UAVid is the first where the catch-all is the BETTER-than-average
class** — because UAVid's clutter is a real visual category (rooftop equipment, walls,
fences), not LoveDA's "everything else".

## 13. What is NOT done

- ⛔ **Lever 2 (per-class scale)** — needs `--cache-full`. Now feasible: at stride 4 that is
  ~14 MB/tile, so 200 train + 70 val is under 4 GB. **Not run.**
- ✅ **train→val transfer — DONE, §14. +1.03, prediction confirmed.**
- ⛔ **End-to-end `eval.py` verification.** Everything above is cached-histogram arithmetic.
  WEEK3 §9c's rule: verify in the segmentor before it is written up.


---

## 14. ✅⭐⭐ THRESHOLDS TRANSFER train→val — +1.03, and the diagnosis is confirmed

Fitted on all 200 real train frames, applied **unchanged** to all 70 val frames. No
val label is touched by the fit, and the two splits share no filenames.
✅ **Gate exact**: the published-τ row reproduces **56.87** against the known 56.87,
so the two caches agree on the label convention despite sitting at different
`--cache-stride` settings (train 4, val 1).

| arm | τ | full mIoU | Δ | excl. catch-all | Δ |
|---|---|---|---|---|---|
| A published τ = 0.3 | global | 56.87 | — | 57.14 | — |
| B best **global** τ fitted on train | 0.190 | 56.99 | **+0.12** | 57.59 | +0.45 |
| ⭐ **C per-class τ fitted on train** | per class | **57.90** | ⭐ **+1.03** | 58.15 | **+1.01** |
| D per-class oracle *on val* | per class | 58.32 | *+1.45* | 58.44 | *+1.30* |

⭐ **C captures 71% of the destination oracle**, and ⭐⭐ **arm B shows the per-class part
is 89% of it**: a single global threshold fitted the same way on the same data buys
**+0.12**. The gain is the *shape* of the vector, not its level.

### ⭐⭐ The permutation control settles that independently

Arm C's thresholds shuffled among the 6 real classes, 200 draws:

| | Δ mIoU |
|---|---|
| **real assignment** | **+1.03** |
| shuffled, mean | ⛔ **−3.08** |
| shuffled, p95 | +0.53 |
| shuffled, max | +0.92 |
| **shuffles matching or beating it** | ⭐ **0 of 200** |

**A shuffled vector is actively harmful (−3.08) and not one of 200 draws reached
+1.03.** What crosses the split boundary is *which class gets which threshold*.
⛔ Note this is why `--deploy-cfg` writes the vector straight from the fit: the
segmentor's length check cannot catch a **permutation**, and a permuted vector would
produce a perfectly plausible table 3 points lower.

### ⭐ And it needs almost no source data

| source tiles | mean Δ | sd | worst draw |
|---|---|---|---|
| **25** *(3 scenes)* | ⭐ **+0.81** | 0.11 | **+0.72** |
| 50 | +0.96 | 0.12 | +0.78 |
| 100 | +1.02 | 0.04 | +0.96 |
| 200 | +1.03 | 0.00 | +1.03 |

**25 tiles from three flights, from a different split, reach 79% of the full-budget
transfer — and the worst of five draws is +0.72.**

### ⭐⭐ The transfer rule now has evidence, and it is LABEL-FREE on both sides

WEEK3 §9b/§9e's rule was *"calibrate on the distribution you will evaluate on"*, with
no way to know in advance whether two distributions were close enough. The discard
rate is computable with **no ground truth at all**, and across four split pairs it
orders the outcome:

| split pair | discard rates | ratio | transfer Δ |
|---|---|---|---|
| ⭐ **UAVid train→val** | 6.73% / 6.81% | **1.01×** | ⭐ **+1.03** |
| LoveDA train→val (§9b) | 14.54% / 29.68% | 2.04× | −0.12 |
| LoveDA urban→rural (§9e) | 18.5% / 39.3% | 2.1× | −0.40 |
| LoveDA rural→urban (§9e) | 39.3% / 18.5% | 2.1× | ⛔ −1.11 |

⚠️ **Four points, two datasets, and the LoveDA rows come from a different protocol**
(5-fold within a domain at a 200-tile budget) than the UAVid row (a single fit on one
split applied whole to another). It is a consistent ordering, **not a fitted law**, and
it must be stated that way.
⚠️ It also does **not** overturn §9f, which asked a different question — *whether
calibration pays at all* — and answered no. This asks whether a calibration set built
somewhere else still applies, which §9f never tested.

### ⛔ Per class — and my named prediction was wrong

| class | τ from train | IoU published | transferred | Δ |
|---|---|---|---|---|
| **human** | **0.015** | 17.59 | 23.78 | ⭐ **+6.19** |
| `background` | *(no effect)* | 55.22 | 56.39 | +1.17 |
| building | 0.150 | 90.76 | 91.41 | +0.65 |
| tree | 0.180 | 53.12 | 53.21 | +0.10 |
| car | 0.000 | 63.33 | 63.40 | **+0.07** |
| vegetation | 0.185 | 50.17 | 49.88 | −0.29 |
| **road** | 0.405 | 67.88 | 67.21 | ⛔ **−0.68** |

⛔ **I named `car` as the class to watch** — it is the only one whose discard rate moves
materially between the splits (28.92% train vs 22.73% val) — and it comes back at
**+0.07**, essentially neutral. **The per-class discard gap did not predict which class
fails.**
⭐ **`road` is the fragile class, and it is fragile everywhere**: −3.11 on val's own
7-scene CV, +0.72 on train's 20-scene CV, −0.68 under transfer. Train fits it at
**0.405** where val's oracle wants **0.700**. Its optimum genuinely differs between the
splits, and it is the one class the transfer costs.
