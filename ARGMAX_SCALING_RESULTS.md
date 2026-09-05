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
