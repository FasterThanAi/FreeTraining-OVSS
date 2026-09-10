# Separability of the per-class threshold objective — 11 Sep 2026

**Status:** ✅ separability **proved and confirmed on three datasets** ·
⚠️ **a claim in the paper (§9d) is wrong and needs restating** ·
⛔ **the selection rule this suggested does NOT work — reported as a negative** ·
⭐ **an unexploited oracle gap on OpenEarthMap is now measured**

Scripts: `tau_curves.py`, `separability_proof.py`, `tau_select.py`.
Outputs: `docs/fig10_tau_curves{,_potsdam,_oem}.{png,pdf}`,
`~/outputs/week3/tau_{curves,select}_*.md`.

---

## 1. ⭐ The objective separates across real classes — proved, then confirmed

**The argument.** For a fixed `argmax`, a pixel predicted class *c* either clears
`τ_c` and stays *c*, or fails and becomes the catch-all *b*. **It can never become
another real class.** Hence for every real *c*:

```
TP_c = #{gt=c, pred=c, conf ≥ τ_c}
FP_c = #{gt≠c, pred=c, conf ≥ τ_c}
FN_c = #{gt=c} − TP_c
```

All three depend on **`τ_c` alone**. So the objective `real` (mIoU over real
classes, which is what `tau_cv.py` and `tau_deploy.py` actually fit) is a **sum of
single-variable terms**.

**Measured** — max |Δ IoU| of any *other* real class while one threshold sweeps
the whole grid:

| dataset | classes swept | max |Δ IoU| of any other real class | catch-all moves |
|---|---|---|---|
| LoveDA | 6 | **0.000000** | 2.03 – 17.38 |
| Potsdam | 5 | **0.000000** | 2.89 – 12.52 |
| OpenEarthMap | 8 | **0.000000** | 0.60 – 4.03 |

**Exactly zero on all 19 sweeps.** All coupling runs through the catch-all.

### Three consequences

- ⭐ **Coordinate ascent is EXACT, not greedy.** One sweep per class *is* the global
  optimum. `rounds=1 == rounds=6` on LoveDA, Potsdam and OEM, and on 20 random
  histograms.
- ⭐ **The search is provably cheap:** `N × (bins+1)` evaluations instead of
  `(bins+1)^N`. On LoveDA, **1,206 against 201⁶ ≈ 6.6 × 10¹³**.
- ⭐ **A per-class rule can be validated one class at a time** without touching any
  other threshold — which is what made §3 below testable at all.

### ⛔ It does NOT hold for the `all` objective

The catch-all collects rejected pixels from every class, so `IoU_b` depends on the
whole vector. `separability_proof.py` finds an explicit counterexample in three
random tables: the best `τ` for one class differs depending on where the others
sit. **Potsdam confirms it in real data** — `objective all` gives
`max |Δτ| = 0.0050` between one round and six, where `real` gives `0.0000`.

⚠️ **LoveDA's 0.0000 under `all` is a coincidence of that dataset.** Stating
separability without naming the objective would be wrong.

---

## 2. ⚠️ §9d's stated reason is false and must be restated

The paper currently says:

> *"raising one class's threshold moves its pixels into the catch-all, changing
> the catch-all's IoU **and therefore every other class's optimum**"*

**The second half is false.** Other real classes' optima do not move — measured at
exactly zero, and provably so.

⭐ **The conclusion survives; the mechanism was wrong.** The correct version:

> The objective is **separable across real classes**: each threshold is the peak of
> that class's own IoU curve, whose location depends on the full score distribution
> over true and false pixels. **A summary statistic such as precision does not
> locate that peak** — which is why eleven label-free rules failed.

This is *stronger* than the original: it is provable rather than asserted, and it
explains an observation the coupling story could not — that the fitted thresholds
are **not monotone in precision** (`road` at 69.7% precision takes 0.675 while
`barren` at 50.9% takes 0.375). The **shape** of the curve decides, not a scalar
summary of it.

---

## 3. How much of the oracle the fit actually captures

`tau_curves.py`, held-out real-class mIoU. "Oracle" chooses each threshold on the
**held-out** labels — an upper bound, not achievable.

| dataset | published | fitted | oracle | **captured** |
|---|---|---|---|---|
| LoveDA | 47.46 | 48.70 | 48.96 | **83%** |
| Potsdam | 66.25 | 67.15 | 67.48 | **73%** |
| ⛔ **OpenEarthMap** | 47.28 | **47.56** | **49.80** | ⛔ **11%** |

### ⛔ OEM's shortfall is one class

| OEM class | published τ | fitted | oracle | IoU @pub | @fitted | @oracle |
|---|---|---|---|---|---|---|
| `road` | 0.100 | 0.310 | 0.550 | 46.05 | 49.09 | 49.64 |
| `cropland` | 0.100 | 0.550 | 0.520 | 42.12 | 45.41 | 45.59 |
| `building` | 0.100 | 0.375 | 0.385 | 75.44 | 79.66 | 79.66 |
| ⛔ **`water`** | 0.100 | **0.710** | **0.240** | **69.18** | ⛔ **55.02** | 69.84 |

**`water` loses 14.16 IoU.** Its calibration curve peaks near 0.7; its held-out
curve peaks near 0.1. With 100 calibration tiles and water unevenly distributed,
the peak was noise and the argmax rule followed it.

⭐ **Fix `water` alone and OEM's +0.28 becomes +2.14.** The dataset's weak result is
not a property of the dataset.

⚠️ **And it is not a sample-size problem.** OEM calibrates on 26% of its tiles
(100/384) against LoveDA's 12% (200/1669), and still overfits. ⛔ **Re-downloading
the full 500-tile split would not fix it, and would invalidate every recorded OEM
number.**

---

## 4. ⛔ The proposed fix FAILS — a negative, with a mechanism

The deployed rule takes the argmax of the calibration curve however flat or noisy
it is. Three rules were tested, **all using calibration data only**:

- **plain** — argmax of the pooled calibration curve *(deployed today)*
- **cv** — argmax of the mean of 5 inner-fold curves
- ⭐ **1se** — among thresholds within one standard error of the best, the one
  **closest to the published τ**. The classic one-standard-error rule.

⚠️ **`1se` was named as the principled choice *before* the run**, on the grounds
that when evidence cannot separate candidates one should prefer the least movement
from the prior. It was not chosen because it won.

| Δ real-class mIoU vs published | plain | cv | **1se** | oracle |
|---|---|---|---|---|
| LoveDA | **+1.23** | **+1.23** | ⛔ **+0.76** | +1.50 |
| Potsdam | +0.90 | **+1.00** | ⛔ **+0.55** | +1.23 |
| OpenEarthMap | +0.28 | +0.15 | ⭐ **+0.54** | +2.51 |
| **mean vs plain** | — | **−0.01** | ⛔ **−0.19** | — |

> ⛔ **1se is worse than what is already deployed.** It wins on OEM by +0.26 and
> loses on LoveDA by 0.47 and Potsdam by 0.35. `cv` is indistinguishable from
> plain. **Neither is an improvement.**

### ⭐ Why it fails — the useful part

**1se is too blunt.** On OEM it set six of eight classes to exactly the published
0.100, avoiding `water`'s −14.16 but discarding four genuine gains (`road` +3.04,
`cropland` +3.29, `bareland` +3.14, `pavement` +2.61).

⛔ **And on LoveDA it did the opposite of what was needed**: it moved `water` from
**0.175 → 0.320**, costing 2.66 IoU, because 0.320 lies inside the admissible band
*and* is nearer the published 0.5. **The tiebreak dragged back the one class that
genuinely needed moving.**

> ⭐ **1se cannot distinguish "flat because uninformative" from "flat near a
> genuinely displaced optimum."** Both look identical on the calibration curve.

### ⭐⭐ What the run reframes

OEM `water` at the three candidate thresholds:

| τ | 0.100 *(published)* | 0.240 *(oracle)* | 0.710 *(fitted)* |
|---|---|---|---|
| held-out IoU | **69.18** | 69.84 | ⛔ **55.02** |

**The published threshold was already within 0.66 of optimal.** `water` never
needed calibrating. `road` did: 46.05 → 49.64 available.

> **The open question is not "how do we pick a better threshold" but "which classes
> should be calibrated at all?"**

⚠️ That is a per-class restatement of **§9f**, which asked the same question at
dataset level and returned a negative on two datasets. Expect the per-class version
to fail for the same reason; if it does, the honest finding is that the bound
applies at both granularities.

⛔ **Do not try a fourth selection rule.** Shrinkage with a tuned λ, a different SE
multiplier, a variance-weighted blend — all founder on the same obstacle. This is
`CLAUDE.md`'s "do not look for a thirteenth statistic" one level down.

---

## 5. What to change in the paper

1. ⭐ **Add the separability result.** It is provable, exact on 19 sweeps, and makes
   coordinate ascent exact rather than greedy. State it for the `real` objective and
   note the counterexample for `all`.
2. ⚠️ **Restate §9d's mechanism** per §2 above. The conclusion is unchanged.
3. ⭐ **Report the captured-share table** (83 / 73 / 11%) in limitations. The OEM
   oracle gap is a measured bound on the method, and naming it is stronger than
   leaving a reader to compute it.
4. ⛔ **Report the 1se negative in one sentence**, as with the other bounded
   families. It cost an hour and it closes a question a reviewer would otherwise ask.
