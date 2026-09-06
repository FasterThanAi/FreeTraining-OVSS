# Scaling before the argmax — a second lever, verified end to end

**5 Sep 2026** · `argmax_reorder.py`, `reorder_deploy.py` · LoveDA 1669 tiles @ τ=0.5

⭐ **The method roughly doubles.** Per-class thresholds are worth +1.16 mIoU; a per-class
*scale applied before the argmax* is worth **+1.16 more on top**, five-fold, every fold
positive. Confirmed by `eval.py`, not only by cache arithmetic.

---

## Why this family, and why it is not a restatement

The paper's method section proves per-class thresholds are the **complete** parameterisation
of the decision — but complete *given the argmax*. Every per-class monotone recalibration
composed with a global threshold collapses to a threshold vector, so nothing acting **after**
the argmax can do better. What acts **before** it is strictly larger, and the mass there is
not small:

| error type | LoveDA | reachable by per-class τ? |
|---|---|---|
| below the threshold | 94.0% of the residual | ✅ |
| catch-all won the argmax | 6.0% | ⛔ |
| confused with another real class | 9.9% of all real-class px | ⛔ |

`forest → agricultural` alone is 23.8M px, `water → agricultural` 19.3M (WEEK1 §8.1). No
threshold vector can touch either.

**The rule.** `pred = argmax_c (w_c · s_c)`, then keep if `s_pred ≥ τ_pred`, fitting
`(w, τ)` together. ⭐ `conf` reads the **raw** score deliberately: a scale applied after the
argmax is monotone and folds into τ, so reading raw confines `w` to the reorderings and
makes the increment attributable to them alone.

⚠️ **Per-class logit scaling is not new** — it is prior correction from the long-tail
literature. What is new here is that it is the family the completeness argument excludes,
that it pays at the same supervision budget, and that it pays *on top of* thresholds.

---

## The result — LoveDA, 5-fold

| rung | | mIoU | Δ |
|---|---|---|---|
| **A** | published τ = 0.5 | 47.63 | — |
| **B** | per-class τ | 48.79 | **+1.16 ± 0.15** |
| **C** | + class scale | **49.95** | **+1.16 ± 0.19** over B |
| | | | **+2.32 total** |

**C − B: 5/5 folds positive**, range +0.96 to +1.45, and **mean − 2·sd = +0.78 > 0**, so it
clears the bar used elsewhere in this project (`tau_domain.py`).

### ⭐ Every real class improves and the catch-all does not move

| class | Δ IoU (C over B) |
|---|---|
| **water** | **+4.41** |
| **forest** | **+2.63** |
| barren | +0.63 |
| building | +0.18 |
| agricultural | +0.18 |
| road | +0.13 |
| *background (catch-all)* | *−0.03* |

**+1.36 catch-all-excluded against +1.16 full.** Both positive and the catch-all inert, so
this is structurally immune to the artefact that caught three earlier results (§8.1, §9e).

---

## ✅ Verified end to end

`reorder_deploy.py` fitted on 200 tiles and predicted three numbers; `eval.py` ran all three
on the same 1469 held-out tiles. The segmentor printed its scale vector on load, so the
config demonstrably reached the model.

| | predicted | measured |
|---|---|---|
| baseline *(exact)* | 47.64 | **47.65** |
| per-class τ *(exact)* | 47.66 | **47.68** |
| ⭐ **the increment** | **+1.31** | **+1.34** |

Per class the agreement is near-exact: `background` +1.55 / **+1.55**, `building` +0.09 /
**+0.09**, `agricultural` +0.17 / **+0.17**, `forest` +3.06 / **+3.08**, `water` +3.97 /
**+3.89**.

⚠️ **A subsampling subtlety, now fixed in the script.** Rung 3 is computed on a 40k-px/tile
subsample while rungs 1–2 are exact. The subsample carried a **+0.28** absolute offset, so
the naive rung-3 prediction of 49.27 missed the measured 49.02 by 0.25 — while the
*increment* matched to 0.03, because the offset cancels. `reorder_deploy.py` now reports the
debiased absolute (exact rung 2 + subsampled increment), which for this run gives **48.97
against a measured 49.02**. **Quote the increment; treat a subsampled absolute with care.**

⚠️ **This deployment draw was unlucky for thresholds** — τ alone gave only +0.03 here
(47.65 → 47.68) against the five-fold +1.16. It is a verification of the *instrument*, not a
second estimate of the gain. Quote the five-fold.

---

## ⭐ The calibration budget does NOT go up

The obvious objection to 2N−1 parameters instead of N−1. Both rules, same draws, 3 repeats:

| tiles | τ only | + scale | **increment** | worst draw |
|---|---|---|---|---|
| 100 | +0.83 | +1.05 | +0.22 | **−0.21** |
| **200** | +0.66 | +1.60 | **+0.94** | **+0.66** |
| 400 | +1.01 | +2.13 | +1.12 | +0.85 |
| 800 | +1.26 | +2.38 | +1.12 | +0.97 |

**200 tiles is where every draw turns positive — the same figure as for thresholds alone**,
despite twice the parameters. Saturates by 400.

⭐⭐ **The combined rule on 200 tiles (+1.60) beats thresholds alone on 800 (+1.26).**
Adding the scale is worth more than quadrupling the labelling budget.

---

## The fitted scales, and why they are believable

| | background | building | road | water | barren | forest | agricultural |
|---|---|---|---|---|---|---|---|
| mean | **0.41** | **0.59** | 0.98 | **2.55** | 1.07 | **1.54** | 1.02 |
| sd across folds | 0.03 | 0.01 | 0.03 | 0.08 | 0.04 | 0.10 | 0.02 |

Largest relative spread **6.7%**. Renormalised to geometric mean 1, since only ratios matter
to an argmax.

⭐⭐ **The fit rediscovered WEEK1 §7.7 without being told.** It drives `background` **down to
0.41** and `water` **up to 2.55** — and §7.7 measured, six weeks earlier, that *every*
mechanism-(B) pixel (catch-all winning the argmax at `conf ≥ τ`) is **water**: 19,378,177 px
in 24 tiles. Two independent measurements, the same fault.

⭐ **`agricultural` stays at 1.02, and that is the subtle part.** §8.1 named it the
prediction-side attractor, present in four of the top five confusions, so the obvious fix is
to suppress it. The optimiser declined — agricultural is 44.7% of all pixels and suppressing
it would destroy its own recall. It raised `water` and `forest` instead, which corrects
`forest→agricultural` and `water→agricultural` **by the same ratio at no cost elsewhere**.
That is the coupled objective of §9d being solved directly, and the gains land exactly there:
water +4.41, forest +2.63.

---

## ⛔ OpenEarthMap — inconclusive, and the "gain" is one class

**5 Sep, 384 tiles @ τ=0.1, all rungs evaluated exactly over every pixel.** Cache verified:
384 tiles, rung A **43.64** against OEM's published 44.16.

| | LoveDA | **OpenEarthMap** |
|---|---|---|
| per-class τ (B − A) | +1.16 ± 0.15 | +0.41 ± 0.26 |
| **scale (C − B)** | **+1.16 ± 0.19** | **+0.95 ± 0.85** |
| folds positive | **5/5** | 4/5 |
| mean − 2·sd | **+0.78** ✅ | **−0.75** ⛔ |
| scale stability, worst class | **6.7%** | **15.7%** |

**Verdict: not readable.** The fitted scales move 16% between folds, so each fold is fitting
a different rule and the sign of the increment is not evidence either way.

### ⭐ And the per-class table says why, which the aggregate hides

| class | Δ IoU |
|---|---|
| **cropland** | **+8.57** |
| pavement | +2.98 |
| water | +2.15 |
| road | +0.78 |
| *background (catch-all)* | *−0.02* |
| building | −0.65 |
| tree | −0.80 |
| grass | −1.77 |
| bareland | −2.66 |

⛔ **`cropland` alone is +8.57. The other seven real classes sum to +0.03, and four of
eight get worse.** Catch-all-excluded works out to +1.08 and full mIoU to +0.95 — both
positive, both produced by a single class in a nine-class unweighted mean.

⭐⭐ **This is §8.1 happening a third time on the same benchmark.** There, recovery's
apparent +2.28 was 110% `background` while real classes lost 2.11. Here it is `cropland`.
**OpenEarthMap keeps producing headline mIoU movements that are one class**, and the
per-class table keeps being the only thing that reveals it. Compare LoveDA, where every
real class improved and the two largest gains were 4.41 and 2.63.

### Two causes, not yet separated

1. **The dataset is small.** 384 tiles over five folds is ~77 evaluation tiles each against
   LoveDA's 334. The baseline itself swings **41.0 → 46.0** across folds — a spread larger
   than the effect being measured.
2. **The search subsample was too thin.** The gate disagreed by 0.199 against a 0.15 bar.
   Results are exact, but `w` was *searched* on that subsample, so the fit optimised a
   noisier objective than it was scored against. A failed gate and unstable scales are
   plausibly the same problem, and `--subsample` is the lever.

⚠️ **Neither is established.** A re-run at `--subsample 150000` would separate them, at
roughly 5× the search cost.

### Calibration curve — every worst draw is negative

| tiles | τ only | + scale | increment | worst draw |
|---|---|---|---|---|
| 50 | −0.48 | +0.10 | +0.58 | **−0.16** |
| 100 | +0.22 | +0.29 | +0.07 | **−0.29** |
| 200 | +0.01 | +0.28 | +0.27 | **−0.71** |

Against LoveDA, where every worst draw from 200 tiles up was positive (+0.66 / +0.85 /
+0.97). Note also that **per-class τ itself barely works here** (+0.01 at 200 tiles), which
matches §9b: OEM resists both levers, not just this one.

---

## Pre-registered predictions, scored

`prereg/predict_oem_scaling.md`, committed at `7a14924` before the run.

| | prediction | measured | |
|---|---|---|---|
| **S1** | `background` scale < 0.7 | **0.56** | ✅ |
| **S2** | catch-all-excluded increment > +0.30 | **+1.08** | ⚠️ numerically, but not established |
| **S3** | full increment < catch-all-excluded | +0.95 < +1.08 | ✅ |
| **S4** | increment < LoveDA's +1.36 | +1.08 | ✅ |
| **S5** | scales span ≥ 2× | **3.2×** (1.52 / 0.48) | ✅ |

⚠️ **Four of five hold, and the fifth is the one that matters.** S1, S3, S4 and S5 describe
the *shape* of the fit, which is measurable whether or not the effect is real. S2 is the
only prediction about the effect itself, and the increment it names does not clear the
stability bar — so it is satisfied by a number that is not established.

⛔ **My pre-registration did not have a branch for this outcome.** It named "S5 holds, S2
fails" as *"the scales move but buy nothing — LoveDA-specific"*, and "S5 fails, S2 fails" as
an explained null. It did not anticipate **S2 numerically satisfied by a single class while
the fit is too unstable to read**. Recorded as a gap in the pre-registration rather than
forced into a branch it does not fit.

---

### ⛔ The denser re-run settles it: OpenEarthMap cannot measure this

`--subsample 150000`, 3.75x the pixels for the `w` search. If a thin search subsample were
the cause, the scales should have stabilised. **They got worse.**

| | 40k | **150k** |
|---|---|---|
| C − B | +0.95 ± 0.85 | **+0.40 ± 1.41** |
| folds positive | 4/5 | **3/5** |
| scale instability, worst class | 15.7% | **28.6%** |

⭐ **The decisive number is rung A**, the published baseline, which does not involve the
subsample at all: it swings **39.64 → 49.79** across folds. A **10-point** spread against an
effect of 0.4. And **per-class τ becomes unreadable too** — **+0.40 ± 1.34**, with fold 1 at
**−1.87** for a method verified end-to-end on LoveDA at +1.16 ± 0.15.

> **OpenEarthMap's 384 tiles cannot resolve a ~1 mIoU effect under 5-fold cross-validation.
> The fold-to-fold variance of the baseline is an order of magnitude larger than the
> quantity being measured, and this applies to BOTH levers equally.** It is a limit of the
> measurement, not a finding about the rule.

`cropland` still carries what movement there is (+7.70, with four of eight real classes
worse), so the §8.1 reading is unchanged.

⚠️ **A design flaw this comparison exposed, now fixed.** The fold partition was drawn from
the same RNG stream as the pixel subsampling, so `load_full` consumed randomness in
proportion to `--subsample` and **raising the subsample silently reshuffled the folds**. The
40k and 150k runs were therefore never comparable — rung A differed between them for a
reason that had nothing to do with the knob under test. The partition is now drawn first,
from its own stream, and is a function of `--seed` alone; verified by two subsample settings
producing identical rungs A and B. **Two settings of one knob must differ in that knob
alone**, and this one did not.

⭐ **Where the transfer test should go instead: Potsdam.** 2016 tiles — *more than LoveDA's
1669* — giving 403 evaluation tiles per fold against OEM's 77. Its tiles are 512², so a
`--cache-full` is roughly 6 GB rather than LoveDA's 42, and about 35 GPU-minutes. It is both
the bigger and the cheaper test, and OpenEarthMap should be dropped from this question.

---

## ⭐⭐ Potsdam — a second positive, and it explains a September anomaly

**6 Sep, 2016 tiles @ τ=0.1, all rungs exact.** Cache gate passed: **57.87 mIoU / 4.68%
discard**, matching `POTSDAM_RESULTS.md`. Rung A **57.84**. Search-subsample gate passed at
**0.052** against a 0.15 bar.

| fold | A published τ | B per-class τ | C + scale | C − B |
|---|---|---|---|---|
| 1 | 58.33 | 58.07 | 62.59 | +4.52 |
| 2 | 58.00 | 58.74 | 63.21 | +4.47 |
| 3 | 57.35 | 58.24 | 63.40 | +5.16 |
| 4 | 57.88 | 58.85 | 64.07 | +5.21 |
| 5 | 57.65 | 58.28 | 63.24 | +4.96 |

**C − B = +4.86 ± 0.35, 5/5 folds, mean − 2·sd = +4.16.** Clears the gate by a wide margin.
**Every real class improves.**

| road | building | grass | **tree** | car | *clutter* |
|---|---|---|---|---|---|
| +1.08 | +0.59 | +3.97 | **+21.56** | +2.81 | *−0.82* |

Catch-all-excluded **+6.00**, full **+4.86**.

### ⭐⭐ This is the answer to `tree`, which broke §9g

`POTSDAM_RESULTS.md` recorded the anomaly and could not explain it:

> ⚠️ **`car` carries it, and `tree` does not** — despite `tree` having precision 93.34 /
> recall 38.63, a **+54.7** gap, larger than LoveDA's `water` (+34.8) that drove the entire
> LoveDA result. **The precision–recall asymmetry did not predict which class would move.**
> That weakens §9g's ρ = +0.713.

**`tree` fires correctly 93% of the time and finds 39% of what is there — and per-class τ
recovered +0.32 of it.** The scale recovers **+21.56** at w = 4.04.

⭐ **The reason is visible in the reachability run**, which measured Potsdam `tree` at **96.2%
reachable but only 19.2% SELF-reachable**. Nearly all of `tree`'s residual is below the
threshold — but with a *different class* winning the argmax. Lowering `tree`'s own threshold
therefore returns those pixels wearing the wrong label, which is precisely why the largest
precision–recall gap in the project moved nothing. Scaling flips the argmax and collects it.

> **§9g's precision–recall gap identifies a class that is under-firing. It does not say
> whether a THRESHOLD can do anything about it. `tree` is the case where those two come
> apart, and the second lever is what reaches it.** The anomaly is not a defect in §9g; it
> is the signature of the family §9g's rule cannot address.

### ⚠️ `tree` is 72% of the gain — and the result survives without it

| | |
|---|---|
| real-class mean, all 5 | **+6.00** |
| real-class mean **excluding `tree`** | **+2.11** |
| full mIoU **excluding `tree`'s contribution** | **+1.27** |

Comparable to LoveDA's +1.16 with `tree` removed entirely. **Unlike OpenEarthMap, where four
of eight real classes got worse and one class was the whole effect, every Potsdam class
improves and the result does not depend on the largest one.**

### Calibration cost: lower than for thresholds alone

| tiles | τ only | + scale | increment | worst draw |
|---|---|---|---|---|
| 100 | −0.09 | +4.71 | **+4.81** | **+4.42** |
| 200 | +0.73 | +5.09 | +4.36 | +4.11 |
| 400 | +0.83 | +5.52 | +4.69 | +4.67 |
| 800 | +0.15 | +5.18 | +5.03 | +4.41 |

⭐ **Positive at every size with worst draws above +4.1, from 100 tiles.** Flat thereafter —
this needs *less* calibration data than per-class τ does, not more. Note also that per-class
τ alone is worth only +0.15 to +0.83 here; **on Potsdam the scale is the method and the
threshold is the ablation.**

### ⚠️ Two checks that fired, and why they were wrong

**`clutter` spreads 51% across folds.** It is the catch-all, and `--objective real` does not
score it, so its scale is only weakly identified — it moves the objective indirectly, through
which pixels it takes from real classes. Every real class sits between **2.6%** (`car`) and
**15.3%** (`road`). The stability gate now covers the real classes and reports the catch-all
beside them.

**`road` still sits at 15.3%, marginally over the 15% bar — and the bar was not moved.**
Instead the check order was fixed. Parameter instability was vetoing the verdict *before the
gain was considered*, and those measure different things: overfitting shows up as an unstable
**gain**, and the gain here is +4.86 ± 0.35 with a range of 0.74 over five folds. Unstable
parameters with a steady gain is **non-uniqueness** — coordinate ascent reaching equivalent
optima by different routes on a coupled objective. Instability now vetoes only when the gain
*also* fails, and otherwise annotates it. ⛔ **Consequence: report the gain, never a single
fitted vector as "the" scales.**

⚠️ Both changes are post-hoc, made after seeing a result I wanted to keep. The arguments are
about what the statistics mean rather than about this run, and are stated in the code, but
the ordering is recorded honestly.

---

## ✅ Potsdam verified end to end — 57.60 → 63.27

**6 Sep.** Three `eval.py` passes on the same 1816 held-out tiles, after a config bug
described below was fixed.

| rung | predicted from cache | **measured by `eval.py`** | miss |
|---|---|---|---|
| A published τ = 0.1 | 57.63 | **57.60** | −0.03 |
| B per-class τ | 58.40 | **58.35** | −0.05 |
| C **+ class scale** | 63.31 | **63.27** | −0.04 |

| | |
|---|---|
| per-class τ (B − A) | **+0.75** |
| ⭐ **scale (C − B)** | **+4.92** — against +4.86 ± 0.35 from the 5-fold |
| both together (C − A) | **+5.67** |

**Every class within 0.23 IoU of prediction**, worst `road` at 0.23.

### ⭐⭐ `tree`: 37.92 → 59.09, measured

**+21.17 against a predicted +21.11.** Recall 39.26 → 66.83.

That is the class `POTSDAM_RESULTS.md` recorded in September as an unexplained anomaly:
the largest precision–recall gap in the project (**+54.7**, precision 93.34 / recall 38.63),
which per-class τ moved by **+0.32** and which weakened §9g's ρ = +0.713. **The mechanism is
now confirmed in the pipeline, not inferred from a cache.**

Also confirms the histogram as an instrument on a second dataset, so the 5-fold **+4.86 ±
0.35** inherits the guarantee — the same argument §9c made for LoveDA.

---

## ⛔ The config bug this uncovered, and why it hid for a week

The first three passes gave 57.60 / **57.18** / **61.71** — rung A exact on every class,
rungs B and C low by 1.2 and 1.6 with three of six classes off by 2–3 IoU.

**Cause: `tau_deploy.py` and `reorder_deploy.py` hardcoded `confidence_threshold=0.5` into
every generated config.** That is SAM 3's *decoder instance-confidence* threshold, passed
straight into `Sam3Processor` — a different knob from `prob_thd`, and one that changes
`seg_logits` **themselves** rather than the decision applied on top of them.

So pass 1 (the original `cfg_potsdam.py`) and passes 2–3 (generated configs) **were running
different models.** Both templates now inherit it from `_base_`.

⚠️ **`cfg_loveda.py` uses 0.5, so this was a no-op on LoveDA** — which is exactly why §9c's
end-to-end verification matched to 0.04 per class and the bug stayed invisible. **It could
only surface on a second dataset**, and it did, the first time one was tried.
⛔ It is in `tau_deploy.py` too, the script behind the *published* §9c verification. That
result is unaffected because the value coincides there, but any future per-class τ
deployment on another dataset would have been wrong.

**How it was localised, and the step that mattered:** `diagnose_vector_tau.py` applied the
segmentor's rule **per pixel with no binning** and compared against `confusion_at` on the
same tiles and the same vector. They agreed **exactly** — for the vector *and* the scalar.
That eliminated the histogram, which was the alternative and the far worse outcome: had it
failed, every cached sweep in the project would have inherited the error, including the
+1.18 headline. With the arithmetic exonerated the only remaining candidate was the cache
vs the pipeline, and the only thing that differs between a scalar run and a vector run is
which config file was loaded.

⭐ **The tell was that rung A was exact.** A scalar threshold is invariant to whatever else
changed; only the fitted rungs used a generated config. **A control that reproduces is worth
more than a result that does not**, and it is the reason this was findable at all.

---

## ⛔ ConInfer (CLIP) — a clean null, and my main prediction was FALSIFIED

**6 Sep, 1669 tiles @ their published τ = 0.8, all rungs exact.** Both gates passed:
instrumented run **36.99** exactly, `conf` in **[0.1601, 0.9611]**. Rung A **37.00**.

| | |
|---|---|
| per-class τ (B − A) | **+2.51 ± 0.34** — reproduces §7.1a exactly |
| **scale (C − B)** | **−0.10 ± 0.14**, 1/5 folds positive |
| catch-all-excluded | **+0.27** (real classes), catch-all **−2.31** |

**Scaling before the argmax adds nothing on a CLIP backbone.** The learning curve agrees at
every size (−0.18 / +0.12 / −0.27 / −0.05).

### Pre-registered predictions, scored — `prereg/predict_coninfer_scaling.md` (`4b5ee2e`)

| | prediction | measured | |
|---|---|---|---|
| **C1** | increment **> +0.50** | **−0.10** | ⛔ **FALSIFIED** |
| C2 | smaller than Potsdam's +4.92 | −0.10 | ✅ |
| C3 | no class > 60% of the real gain | `water` 48% | ✅ |
| C4 | scales span ≥ 2× | **15.9×** | ✅ |
| C5 | positive worst draw at 50 tiles | +0.06 | ✅ |

⭐ **The pre-registration named this exact outcome in advance.** Its branch table reads:
*"C4 holds, C1 fails → the scales move but buy nothing: CLIP's argmax is comparatively
clean, and τ = 0.8 already collects what is available."* The fit **does** move the scales
hard — `water` at **7.94**, a 15.9× span — and it buys nothing.

### ⛔ And the REASONING behind C2 is contradicted, which matters more than C1

I argued the gain should track the **argmax-lost share**: less argmax-blocked mass, less for
the scale to recover. Across three datasets that is simply wrong.

| dataset | published τ | argmax-lost | τ gain | **scale gain** |
|---|---|---|---|---|
| Potsdam | 0.1 | **6.10%** (lowest) | +0.75 | **+4.92** (highest) |
| LoveDA (SAM 3) | 0.5 | 11.31% | +1.16 | +1.16 |
| ConInfer (CLIP) | 0.8 | 8.56% | **+2.51** | **−0.10** |

**Potsdam has the least argmax-lost mass and the largest scale gain.** The stated mechanism
does not survive contact with three datasets, and is withdrawn.

### ⭐ What the three datasets do show: the two levers SUBSTITUTE

Read the last two columns instead. **Where the threshold collects a lot, the scale collects
nothing; where the threshold collects little, the scale collects a great deal.** ConInfer's
+2.51 / −0.10 and Potsdam's +0.75 / +4.92 are the two extremes; LoveDA's +1.16 / +1.16 sits
between them.

⚠️ **Stated as an observation over three points, not a law.** It is post-hoc — it was not
pre-registered, and the pre-registered mechanism failed. It should be tested on a fourth
dataset before it appears in the paper as anything more than a description of these three.

⚠️ Note also that the *totals* are not constant (+5.67, +2.32, +2.41), so the two levers are
not simply splitting a fixed pool.

---

## Status and limits

| | |
|---|---|
| ✅ LoveDA, 5-fold, verified end to end | +1.16 ± 0.19 over per-class τ |
| ✅ Calibration budget | unchanged at ~200 tiles |
| ⛔ **Other datasets** | **untested** — OEM, Potsdam, ConInfer all still open |
| ⛔ Domain transfer | untested; §9e showed τ alone does not transfer, and there is no reason to assume `w` does |
| ⚠️ Larger families | a per-class scale is the *simplest* reordering. A general one is larger still, and unbounded by anything here |

⛔ **Do not quote the +2.32 as a multi-dataset result.** It is LoveDA, and the honest headline
until the transfer runs exist is *"on LoveDA the two levers give +2.32 together, of which
+1.16 is the scale"*.
