# Per-class head fusion — is `max(P_sem, P_inst)` right for every class?

- cache `/home/priyanshu/outputs/potsdam_heads/cache` | tiles **2016** | τ **0.1** | 5-fold | objective **`real`**
- ρ grid **0.25–4.0**, 13 points, symmetric about **ρ = 1 = the published rule**

`s_c = max(a_c · P_sem_c, b_c · P_inst_c)`, `max(a_c, b_c) = 1`, `ρ_c = b_c / a_c`. The overall per-class scale is left to lever 2, which already fits it — ρ carries only the part lever 2 cannot express, the **ratio** between the heads.

✅ **Identity gate: 0.00000** against a bar of 0.01. At ρ = 1 the reconstruction reproduces the cached `logits`, so ρ ≠ 1 measures the deployed pipeline and not an approximation of it. Views per tile: [1].

| fold | A published | B +τ | C +scale | D +fusion | **D − C** |
|---|---|---|---|---|---|
| 1 | 58.38 | 58.14 | 62.66 | 62.53 | **-0.13** |
| 2 | 58.23 | 58.77 | 63.07 | 63.24 | **+0.17** |
| 3 | 57.50 | 58.31 | 63.26 | 63.35 | **+0.09** |
| 4 | 57.90 | 59.27 | 63.83 | 63.90 | **+0.07** |
| 5 | 57.68 | 58.39 | 63.23 | 63.21 | **-0.02** |

- **D − C = +0.04 ± 0.11**, 3/5 folds positive, mean − 2·sd = **-0.19**

## Fitted ρ per class

| class | f1 | f2 | f3 | f4 | f5 | mean |
|---|---|---|---|---|---|---|
| road | 1.26 | 1.00 | 0.79 | 1.26 | 1.26 | **1.11** |
| building | 0.63 | 0.50 | 0.50 | 0.50 | 0.50 | **0.53** |
| grass | 1.26 | 1.00 | 1.00 | 0.79 | 0.79 | **0.97** |
| tree | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| car | 2.52 | 2.52 | 1.59 | 2.52 | 2.52 | **2.33** |
| clutter | 0.79 | 0.79 | 1.00 | 0.79 | 1.00 | **0.88** |

`ρ > 1` = this class wants the **instance** head; `ρ < 1` = the **semantic** head; `ρ = 1` = the published `max` is already right.

## Verdict

⛔ **Null: +0.04 ± 0.11**, 3/5 folds positive. Per-class head fusion buys nothing on top of per-class τ and per-class scaling.

⭐ **4 of 6 classes keep ρ within ±30% of 1 in every fold** — those are classes for which the published `max` is already the right rule, and that is a result about the baseline whichever way D − C goes.