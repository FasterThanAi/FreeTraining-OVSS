# Should some classes keep the published τ? — a selection test

- cache `/home/priyanshu/outputs/loveda_full_all/cache` | tiles **1669** | published τ **0.5** | 5-fold held out | objective **`real`**
- hardcoded arm freezes: **road, agricultural**

⭐ Under `--objective real` the objective **separates**, so "freeze class *c*" is a clean per-class decision and the histogram is a sufficient statistic. Rung B only — lever 2 changes the argmax and would break separability.

| arm | how classes are chosen | mIoU | sd | **vs B** | deployable? |
|---|---|---|---|---|---|
| **A** | none fitted — the baseline | 47.31 | 1.46 | **-1.18** | — |
| **B** | all fitted — **the method** | 48.49 | 1.28 | — | ✅ |
| **hardcode** | named by hand | 48.42 | 1.28 | **-0.07** | ⛔ **no** |
| **cv** | inner-CV on calibration only | 48.42 | 1.21 | **-0.07** | ✅ |
| **oracle** | chosen on the evaluation fold | 48.57 | 1.28 | **+0.08** | ⛔ no |

## Per-class held-out Δ IoU at rung B (fitted − published)

| class | f1 | f2 | f3 | f4 | f5 | mean | folds − | frozen by cv | by oracle |
|---|---|---|---|---|---|---|---|---|---|
| background | -0.55 | -0.14 | +0.67 | +0.04 | -0.05 | **-0.01** | 3/5 | 5/5 | 3/5 |
| building | +1.03 | +0.27 | +0.64 | -1.36 | +0.80 | **+0.28** | 1/5 | 2/5 | 1/5 |
| road | +0.38 | +0.26 | -0.12 | +0.18 | -0.22 | **+0.10** | 2/5 | 1/5 | 2/5 |
| water | +6.63 | +9.64 | +3.96 | +5.96 | +7.73 | **+6.78** | 0/5 | 0/5 | 0/5 |
| barren | +1.36 | +3.01 | +0.97 | +1.35 | -1.82 | **+0.98** | 1/5 | 0/5 | 1/5 |
| forest | +0.59 | +0.12 | +0.63 | +0.18 | +0.34 | **+0.37** | 0/5 | 2/5 | 0/5 |
| agricultural | +0.14 | +0.08 | -0.59 | -0.75 | +0.08 | **-0.21** | 2/5 | 4/5 | 2/5 |

## Verdict

⛔ **The information exists and is not reachable.** A peeking oracle gains **+0.08** mIoU by freezing; the honest inner-CV rule gets **-0.07**. Same shape as the one-standard-error negative (SEPARABILITY_RESULTS §4): which classes to calibrate is decidable *after* the fact and not *before*. **Fit every class.**

⛔ **The hardcoded arm is not a result at any value.** `road, agricultural` were named by reading LoveDA’s own held-out table, so its -0.07 is a measurement of how much peeking is worth, not of a method. It is reported only to show that gap — a rule chosen this way cannot transfer to a dataset whose losing classes differ, and §9e already showed the fitted thresholds themselves do not transfer across a domain.

⚠️ Rung B only. The deployed method is rung C (+ per-class scale), where the argmax moves and the per-class decision is no longer separable.