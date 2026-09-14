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

- ✅ **Lever 2 — DONE, §16. +5.89 ± 1.51, 5/5 folds. The largest lever-2 gain in the project.**
- ✅ **train→val transfer — DONE, §14. +1.03, prediction confirmed.**
- ✅ **End-to-end `eval.py` verification — DONE, §15. 57.92 measured against 57.90 predicted.**


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


---

## 15. ✅✅ VERIFIED END-TO-END BY THE PIPELINE — 57.92 against a predicted 57.90

Everything in §12–§14 is arithmetic on a cached `(gt, pred, conf-bin)` histogram. WEEK3 §9c's
rule is to run it through the actual segmentor before it is written up. LoveDA and Potsdam both
did; UAVid now has too.

`configs/cfg_uavid_val_perclass.py` was written **by `tau_transfer.py --deploy-cfg`, straight
from the fit** — not transcribed. The log shows the segmentor accepting it:
`per-class prob_thd: 0.300, 0.150, 0.405, 0.000, 0.180, 0.185, 0.015`.

| | predicted from the cache | **measured by `eval.py`** | Δ |
|---|---|---|---|
| published τ = 0.3 | 56.87 | **56.86** | 0.01 |
| **per-class τ (transferred from train)** | **57.90** | ⭐ **57.92** | **0.02** |
| **gain** | +1.03 | ⭐ **+1.06** | 0.03 |

### Per class, cached prediction against the pipeline

| class | predicted | measured | Δ |
|---|---|---|---|
| background | 56.39 | 56.53 | 0.14 |
| building | 91.41 | 91.53 | 0.12 |
| human | 23.78 | 23.69 | 0.09 |
| car | 63.40 | 63.32 | 0.08 |
| tree | 53.21 | 53.26 | 0.05 |
| road | 67.21 | 67.24 | 0.03 |
| vegetation | 49.88 | 49.90 | 0.02 |

⚠️ **Max disagreement 0.14, against LoveDA's ≤0.04.** Larger, and the reason is known rather than
guessed: `conf` is cached as **float16** and binned into 200 buckets, so a pixel sitting on a bin
edge can be scored on either side of a threshold. Two of UAVid's fitted thresholds (`car` 0.000,
`human` 0.015) sit at the very bottom of the range where float16 spacing is finest and the
quantisation is relatively coarsest. **Immaterial at 0.02 on the headline, but quote the
measured column, not the predicted one.**

⭐ **This validates the histogram as an instrument on a FOURTH dataset** (LoveDA, Potsdam,
ConInfer, UAVid), so every cached result above inherits it.

### The deployable claim, in one line

> **56.86 → 57.92 (+1.06) on UAVid val, with thresholds fitted on a different split, never
> touching a validation label, and verified by the unmodified evaluation pipeline.**


---

## 16. ✅✅ LEVER 2 IS THE BIGGEST RESULT ON UAVid — +5.89 ± 1.51 over lever 1

200 real train frames, 20 flights, **group-disjoint folds**, augmented copies excluded.
Predictions committed in `prereg/predict_uavid_lever2.md` (`9289fb1`) **before the cache
existed**. All three rungs evaluated **exactly, over every pixel**; the 40k-px subsample
drove only the search for `w`, and its gate passes at **0.099 against a 0.15 bar**.

| rung | | mean | sd | folds+ | mean−2sd |
|---|---|---|---|---|---|
| **B − A** | per-class τ (lever 1) | +1.54 | 0.30 | 5/5 | +0.94 |
| ⭐ **C − B** | **+ per-class scale (lever 2)** | ⭐ **+5.89** | **1.51** | **5/5** | ⭐ **+2.87** |

⭐⭐ **Largest lever-2 gain measured in this project**: LoveDA +1.16, Potsdam +4.92,
ConInfer −0.10, **UAVid +5.89**. Range +4.02 to +7.53, every fold positive.
✅ Rung B reproduces lever 1 independently (+1.54 ± 0.30 against §12's +1.34 ± 0.39,
different cache and code path), so the run is sound.

### ⭐⭐ The pre-registered mechanism was RIGHT, and it is two classes

| | prediction (committed in advance) | measured | |
|---|---|---|---|
| **U1** | clears the gate | +5.89 ± 1.51, 5/5, mean−2sd +2.87 | ✅ |
| ⭐ **U2** | **`vegetation` w < 1 and `tree` w > 1** | ⭐ **0.40 and 2.56** | ✅ **decisive** |
| **U3** | `human` w > 1 | **2.55** | ✅ |
| **U4** | `building` within 25% of w = 1 | **1.63** | ⛔ **fail** |
| **U5** | lever 1 holds within ±0.4 of +1.34 | **+1.54** (Δ 0.20) | ✅ |

U2 named two classes and two signs from the precision/recall table alone, before any fit:
`vegetation` 53.6/87.8 fires far too readily, `tree` 91.8/55.9 far too rarely, and they are
the two green classes competing for the same pixels. **The fit separates them 6.4×.**

⭐ **And the IoU confirms the mechanism, not just the weights:**

| class | Δ IoU (C over B) | w |
|---|---|---|
| ⭐ **tree** | ⭐ **+26.13** | **2.56** |
| ⭐ **vegetation** | ⭐ **+13.50** | **0.40** |
| human | +2.93 | 2.55 |
| building | +0.09 | 1.63 |
| road | −0.23 | 0.40 |
| car | −0.44 | 0.50 |
| `background` | −0.77 | 1.23 |

> ⭐⭐ **`tree` + `vegetation` are 96.2% of the gain.** This is the family the completeness
> argument explicitly excludes: **no threshold can recover a pixel another class already won
> in the argmax.** Lowering `tree`'s τ cannot take back a pixel `vegetation` took.

### ⛔ U4 failed — the suspect-check the pre-registration demanded, run

The prereg says an U4 failure means *treat the run as suspect and check the fit*, because
lever 5's `w` once drifted to a uniform 0.40 by tie-break and produced a false null. Five
checks, all passing:

| check | result |
|---|---|
| gauge (geometric mean of `w` must be 1) | **1.0066** ✅ |
| subsample vs exact on rung B | max **0.099** against a 0.15 bar ✅ |
| lever 1 reproduces | +1.54 ± 0.30 vs the recorded +1.34 ± 0.39 ✅ |
| ⭐ decisive classes stable across 5 disjoint flight groups | `road` sd **0.01**, `vegetation` sd **0.01**, `car` sd **0.02** ✅ |
| which classes wander | `building` 0.39, `human` 0.38, `background` 0.31 |

⭐ **U4 was a badly-posed prediction, and that is the honest verdict.** A scale vector is
defined only up to a global constant — with `road`, `car` and `vegetation` all driven to
0.40–0.50, geometric-mean normalisation *necessarily* lifts everything else. "Stays near 1"
is a gauge-dependent claim about an inherently relative quantity.
⭐ And `building` is simply **not identified**: it moves 1.14–2.11 across folds and changes
its IoU by **+0.09**. A flat direction of the objective, exactly like `background`'s γ in
lever 4 (@PRESENCE_POWER_RESULTS). **Do not quote `building`'s scale.**

### ⚠️ Three caveats that must travel with this number

1. ⚠️ **96.2% of the gain is two classes** out of seven. That is this project's own §9h
   leverage warning turned on itself: an unweighted mean lets two classes carry a headline.
   **Never quote +5.89 without the per-class table.**
2. ⚠️ ⭐ **It may be repairing the VOCABULARY, not the model.** UAVid's class is *low
   vegetation* and the prompt is the bare word `vegetation`, which plainly also describes
   trees. A 6.4× reweighting between two prompts that do not separate is what a bad prompt
   pair looks like from the inside. `PROMPT_ENSEMBLE_RESULTS` measured the vocabulary as
   worth **+4.94 mIoU** on LoveDA — larger than every calibration lever combined.
   ⭐ **Testable and cheap: change the prompt to `low vegetation` or `grass` and re-measure.
   If the lever-2 gain shrinks, part of it was a prompt choice.** Not yet run.
3. ⛔ **Cached-histogram result on the TRAIN split.** It is a prediction until the segmentor
   reproduces it, and the val number is unmeasured.

⚠️ The generated report contains both *"the fitted scales move substantially between folds,
which is what overfitting looks like"* and *"non-uniqueness, not overfitting"*. The first is
a generic warning template; the second is the correct reading here, and the evidence is the
sd column — the three classes that decide the result are stable to **0.01–0.02** across five
disjoint flight groups. Quote the sd, not the warning.


---

## 17. ✅⭐⭐ LEVER 2 TRANSFERS train→val — +5.64 over lever 1 on tiles never seen

`scale_transfer.py`. Fit `(w, τ)` on all 200 real train frames, evaluate **exactly over
every one of 37,324,800 labelled pixels** of the 70 val frames. ✅ Gate exact: published τ
reproduces **56.87**. ✅ The val cache is verified by arithmetic — 37,324,800 =
40×(960×540) + 30×(1024×540), matching UAVid's two frame widths to the pixel.

| arm | mIoU | Δ vs published |
|---|---|---|
| A published τ = 0.3 | 56.87 | — |
| B lever 1 — per-class τ from train | 57.91 | +1.04 |
| ⭐ **C + lever 2 — per-class scale from train** | ⭐ **63.55** | **+6.68** |
| D `(w, τ)` fitted **on val** *(a bound, not a method)* | 64.17 | *+7.29* |

> ⭐⭐ **Lever 2 adds +5.64 over lever 1 across a split boundary, reaching 90% of the
> destination bound (+5.64 of +6.26).**

Compare the within-split 5-fold (§16): **+5.89 ± 1.51**. The transfer keeps **96%** of it.
⭐ **The scale transfers better than the threshold does** — lever 1 keeps +1.03 of its own
+1.34 (77%) — which is the opposite of what `road`'s threshold failure suggested.

### Per class — and no real class loses

| class | w | τ | published | lever 1 | **lever 2** | Δ (C−B) |
|---|---|---|---|---|---|---|
| ⭐ **tree** | **2.522** | 0.115 | 53.15 | 53.25 | ⭐ **74.04** | **+20.79** |
| ⭐ **vegetation** | **0.383** | 0.180 | 50.20 | 49.92 | ⭐ **63.04** | **+13.12** |
| **human** | **2.522** | 0.005 | 17.57 | 23.75 | **29.20** | **+5.45** |
| car | 0.495 | 0.005 | 63.32 | 63.40 | 63.65 | +0.25 |
| building | 1.788 | 0.160 | 90.73 | 91.40 | 91.58 | +0.18 |
| road | 0.383 | 0.405 | 67.92 | 67.25 | 67.40 | +0.15 |
| `background` | 1.212 | — | 55.21 | 56.39 | 55.96 | −0.43 |

⭐ **Every real class is at or above its lever-1 value**, and `road` — which lever 1 *lost*
0.68 on — is recovered to +0.15. The scale repairs the class the threshold hurt.

### ⚠️ The permutation control is weaker than lever 1's, and the reason is in the vector

| | Δ vs published |
|---|---|
| **real assignment** | **+6.68** |
| shuffled, mean | **−1.58** |
| shuffled, p95 | +6.57 |
| shuffled, max | +6.72 |
| matching or beating it | **1.0%** *(2 of 200)* |

Lever 1's control gave **0.0%**. Here two draws match, and the vector says why: `road` and
`vegetation` are **both 0.383**, `tree` and `human` **both 2.522**. Many rearrangements
therefore preserve the structure that matters — *greens split, small objects up, road down*
— and score nearly as well.
⭐ **That refines the claim rather than weakening it: what transfers is the GROUPING, not
seven individually meaningful numbers.** A shuffle that breaks the grouping costs 1.58 mIoU
below the published baseline, an 8-point swing from the real assignment.

✅ **VERIFIED END-TO-END — see §18.** `eval.py` reports **63.5500** against a predicted
**63.55**. ⚠️ The first attempt CUDA-OOMed because another process held 10.61 GB of the
16 GB card — a scheduling collision, not a fault in the run.


---

## 18. ✅✅✅ UAVid IS COMPLETE — 56.86 → 63.55, verified by the pipeline, exactly as predicted

| | predicted from the cache | **`eval.py`** | Δ |
|---|---|---|---|
| **mIoU** | **63.55** | ⭐ **63.5500** | ⭐ **0.00** |
| tree | 74.04 | **74.04** | **0.00** |
| road | 67.40 | 67.39 | 0.01 |
| vegetation | 63.04 | 63.00 | 0.04 |
| car | 63.65 | 63.59 | 0.06 |
| background | 55.96 | 56.09 | 0.13 |
| building | 91.58 | 91.73 | 0.15 |
| human | 29.20 | 29.00 | 0.20 |

The deployed vectors were written by `scale_transfer.py --deploy-cfg` straight from the fit,
and the segmentor echoed both back (`per-class prob_thd: …`, `class_scale: …`) before the run.

### The full chain on UAVid val

| rung | mIoU | aAcc | mPrecision | mRecall |
|---|---|---|---|---|
| published τ = 0.3 | 56.86 | 79.24 | 78.99 | 69.12 |
| + per-class τ *(lever 1)* | 57.92 | 79.52 | 77.92 | 70.32 |
| ⭐ **+ per-class scale *(lever 2)*** | ⭐ **63.55** | ⭐ **84.81** | **79.59** | **74.04** |
| **total** | ⭐ **+6.69** | **+5.57** | **+0.60** | **+4.92** |

Both parameter sets are fitted on the **train** split and applied unchanged; no validation
label is touched by either fit.

### ⭐⭐ Two checks that answer the obvious objections

**1. "The mIoU gain is two classes in an unweighted mean of seven."** ⛔ **That caveat was
too pessimistic and is corrected here.** `aAcc` — overall pixel accuracy, weighted by pixel
count and therefore impossible to move with a rare class — rises **79.24 → 84.81, +5.57**.
And `tree` (23.55%) plus `vegetation` (14.15%) are **37.7% of every pixel in the dataset**.
They are the two *largest* classes after `building`, not rare ones. The caveat applies to
`human` (0.19% of pixels), not to the classes carrying this result.

**2. ⭐ Precision AND recall both rise.** `TTA_RESULTS` records that as the signature of
better decisions rather than redistributed ones, and called dihedral TTA *"the only
intervention in this project where precision AND recall both rise"*. **The two levers
together now do it as well: mPrecision +0.60, mRecall +4.92.**
⭐ And the split is informative: **lever 1 alone TRADES** (precision −1.07, recall +1.20) —
it lowers thresholds to collect discarded pixels. **Lever 2 pays the precision back** and
adds recall on top, because it is not lowering a bar, it is handing the pixel to the class
that should have won it.

Visible per class, and it is exactly the pre-registered mechanism:

| class | precision → | recall → |
|---|---|---|
| **tree** | 91.96 → **84.65** | ⭐ **56.43 → 85.53** |
| **vegetation** | ⭐ **55.71 → 79.78** | 84.04 → 74.97 |
| human | 76.61 → 62.96 | **20.28 → 34.97** |

**The two green classes move toward each other from opposite extremes** — `tree` buys 29
points of recall for 7 of precision, `vegetation` buys 24 points of precision for 9 of
recall. That is the argmax competition between them being rebalanced, which is what the
pre-registration predicted and what no threshold can do.

### The deployed configuration

| class | `class_scale` | `prob_thd` |
|---|---|---|
| background | 1.212 | 0.300 *(no effect)* |
| building | 1.788 | 0.160 |
| road | 0.383 | 0.405 |
| car | 0.495 | 0.005 |
| **tree** | **2.522** | 0.115 |
| **vegetation** | **0.383** | 0.180 |
| human | 2.522 | 0.005 |

⚠️ **Against SegEarth-OV3's published 54.7 this is +8.85 — do NOT quote that as our gain.**
Our own reproduction is 2.16 above their number for reasons we cannot explain, so **the
honest claim is +6.69 over our own reproduced baseline**, exactly as the ConInfer row is
handled in the other direction.


---

## 19. ⭐⭐⭐ ONE WORD IS WORTH +3.53 mIoU — V1 and V2 confirmed

`prereg/predict_uavid_vocabulary.md` (`b20eaf8`), committed before any cache existed.
**One line of `cls_uavid.txt` changed: `vegetation` → `low vegetation`.** ✅ Same line count,
so still one prompt per class — no arity confound (`WEEK3 §7b`).

| | `vegetation` | **`low vegetation`** | Δ |
|---|---|---|---|
| **mIoU** | 56.86 | ⭐ **60.39** | ⭐ **+3.53** |
| aAcc | 79.24 | 82.96 | +3.72 |

### ⭐⭐ And it moves ONLY the two classes lever 2 was repairing

| class | `vegetation` | `low vegetation` | Δ |
|---|---|---|---|
| ⭐ **tree** | 53.15 | **67.81** | ⭐ **+14.66** |
| ⭐ **vegetation** | 50.19 | **61.33** | ⭐ **+11.14** |
| `background` | 55.23 | 54.16 | −1.07 |
| building | 90.79 | 90.75 | −0.04 |
| road | 67.79 | 67.77 | −0.02 |
| human | 17.62 | 17.64 | +0.02 |
| car | 63.28 | 63.29 | +0.01 |

**Every class except `tree`, `vegetation` and the catch-all moves by at most 0.04.** A
two-word prompt cannot be confused with a general improvement: it is surgical.

### V2 — the diagnostic, predicted in advance and both directions right

| | before | after | predicted |
|---|---|---|---|
| `vegetation` **precision** | 53.62 | ⭐ **71.51** | rises ✅ |
| `tree` **recall** | 55.90 | ⭐ **72.76** | rises ✅ |

⭐ The bare word `vegetation` was matching trees, so the class over-fired (precision 53.6)
and stole `tree`'s pixels (recall 55.9). Naming the class *low vegetation* separates them.
**That is the same confusion lever 2 repaired with a 6.4× scale ratio — reached instead by
typing two words.**

### ⚠️ What this already changes, before V3 is known

⭐ **This is a second dataset confirming `PROMPT_ENSEMBLE_RESULTS`' finding that the
vocabulary is the largest single lever in this pipeline.** LoveDA: `barren` 35.73 → 1.17 on
two words, **+4.94 mIoU**. UAVid: **+3.53 on one word**, larger than the entire calibration
method delivers on LoveDA (+2.32). The vocabulary is inherited, hand-written and unreported
by the baseline, and it is worth more than anything else measured in this project.

⛔ **Reporting discipline, decided in the pre-registration rather than now:**
- **The reproduction row keeps THEIR vocabulary** — 56.86 against their published 54.7 — or
  it is no longer a reproduction of their configuration.
- **Our method's reported configuration uses the corrected prompt**, because keeping a number
  obtained with a prompt we know to be wrong, on the grounds that it flatters the method,
  is precisely what the pre-registration exists to prevent.
- **The +3.53 is reported as its own result**, not folded into the method's gain.

⏳ **V3 — the actual test — is still running.** Lever 2 was **+5.64** with the bad prompt.
The new baseline already reaches `tree` 67.81 and `vegetation` 61.33, where lever 2 with the
old prompt reached 74.04 and 63.00 — so some headroom remains, but much of the mass lever 2
was collecting has now been collected by the word. **V3 asks whether lever 2's gain falls
below +2.8.**
