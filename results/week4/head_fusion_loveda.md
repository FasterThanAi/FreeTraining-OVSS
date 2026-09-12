# Per-class head fusion — is `max(P_sem, P_inst)` right for every class?

- cache `/home/priyanshu/outputs/loveda_heads/cache` | tiles **800** | τ **0.5** | 5-fold | objective **`real`**
- ρ grid **0.25–4.0**, 13 points, symmetric about **ρ = 1 = the published rule**

`s_c = max(a_c · P_sem_c, b_c · P_inst_c)`, `max(a_c, b_c) = 1`, `ρ_c = b_c / a_c`. The overall per-class scale is left to lever 2, which already fits it — ρ carries only the part lever 2 cannot express, the **ratio** between the heads.

✅ **Identity gate: 0.00000** against a bar of 0.01. At ρ = 1 the reconstruction reproduces the cached `logits`, so ρ ≠ 1 measures the deployed pipeline and not an approximation of it. Views per tile: [1].

| fold | A published | B +τ | C +scale | D +fusion | **D − C** |
|---|---|---|---|---|---|
| 1 | 46.84 | 47.99 | 49.02 | 49.59 | **+0.57** |
| 2 | 46.32 | 47.85 | 48.98 | 49.00 | **+0.02** |
| 3 | 48.79 | 49.07 | 50.56 | 50.74 | **+0.18** |
| 4 | 46.87 | 48.10 | 48.01 | 48.13 | **+0.11** |
| 5 | 48.56 | 48.31 | 50.01 | 50.24 | **+0.22** |

- **D − C = +0.22 ± 0.21**, 5/5 folds positive, mean − 2·sd = **-0.20**

## Fitted ρ per class

| class | f1 | f2 | f3 | f4 | f5 | mean |
|---|---|---|---|---|---|---|
| background | 1.59 | 1.59 | 1.59 | 1.00 | 1.59 | **1.47** |
| building | 0.50 | 0.50 | 0.50 | 0.63 | 0.40 | **0.51** |
| road | 3.17 | 4.00 | 4.00 | 4.00 | 4.00 | **3.83** |
| water | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| barren | 0.32 | 0.32 | 0.40 | 0.32 | 0.32 | **0.33** |
| forest | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | **0.25** |
| agricultural | 0.79 | 1.00 | 1.00 | 0.79 | 0.50 | **0.82** |

`ρ > 1` = this class wants the **instance** head; `ρ < 1` = the **semantic** head; `ρ = 1` = the published `max` is already right.

## Verdict

⚠️ **Promising, not established: +0.22 ± 0.21**, every fold positive (1 in 32 by sign test) but the spread covers zero. Do not write it up at this width.

⭐ **1 of 7 classes keep ρ within ±30% of 1 in every fold** — those are classes for which the published `max` is already the right rule, and that is a result about the baseline whichever way D − C goes.

⚠️ Check the ρ table before concluding: if ρ is stable across folds while the gains scatter, the fit is stable and the evaluation is noisy, which argues for more tiles rather than against the rule.