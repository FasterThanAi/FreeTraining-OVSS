# An affine reordering — does a per-class BIAS add anything to a scale?

- cache `/home/priyanshu/outputs/loveda_full_all/cache` | tiles **1669** | τ **0.5** | 5-fold | objective **`real`**
- `b` grid **-0.25–0.25**, step 0.05, symmetric about **b = 0 = lever 2 unchanged**

`pred = argmax_c (w_c · s_c + b_c)`, then keep `pred` if `s_pred ≥ τ_pred`. ⭐ The threshold reads the **raw** score, so `b` is confined to the reordering — shifting the score τ sees would merely reparameterise τ.

⭐ Where scores are large `w·s` dominates and `b` is irrelevant; at **low-confidence** pixels `b` decides who wins, and a multiplier cannot reach there. That is the regime the residual lives in.

| fold | A published | B +τ | C +scale | D +bias | **D − C** |
|---|---|---|---|---|---|
| 1 | 48.12 | 49.49 | 50.51 | 50.37 | **-0.14** |
| 2 | 45.76 | 47.59 | 49.13 | 49.20 | **+0.07** |
| 3 | 48.75 | 49.60 | 50.63 | 50.29 | **-0.34** |
| 4 | 46.41 | 47.20 | 49.32 | 49.42 | **+0.10** |
| 5 | 49.04 | 50.02 | 50.39 | 50.33 | **-0.06** |

- **D − C = -0.07 ± 0.18**, 2/5 folds positive, mean − 2·sd = **-0.43**

## Fitted `b` per class

| class | f1 | f2 | f3 | f4 | f5 | mean |
|---|---|---|---|---|---|---|
| background | -0.20 | +0.00 | -0.25 | +0.00 | -0.15 | **-0.12** |
| building | +0.20 | +0.00 | +0.00 | +0.00 | +0.20 | **+0.08** |
| road | +0.05 | +0.00 | +0.00 | +0.00 | +0.00 | **+0.01** |
| water | +0.00 | +0.10 | -0.05 | +0.10 | +0.00 | **+0.03** |
| barren | +0.05 | +0.05 | +0.05 | +0.05 | +0.05 | **+0.05** |
| forest | +0.05 | +0.05 | +0.10 | +0.05 | +0.05 | **+0.06** |
| agricultural | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | **+0.00** |

`b < 0` = this class is suppressed where scores are small; `b > 0` = favoured there; `b = 0` = lever 2 alone was already right.

## Verdict

⛔ **Null: -0.07 ± 0.18**, 2/5 folds positive. A per-class bias buys nothing over a per-class scale. ⭐ The decision side is then exhausted **empirically as well as structurally** — a stronger closing statement than the one that was over-claimed, and reached by testing the family rather than by asserting it.

⭐ **1 of 7 classes keep `b` = 0 in every fold** (within half a grid step) — lever 2 alone was already right for those, whichever way D − C goes.