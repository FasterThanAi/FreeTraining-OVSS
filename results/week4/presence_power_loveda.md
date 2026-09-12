# Per-class presence weight — should the gate apply equally to every class?

- cache `/home/priyanshu/outputs/loveda_full_all/cache` | tiles **1669** | τ **0.5** | 5-fold | objective **`real`**
- γ grid **0.0–2.0**, symmetric about **γ = 1 = the published rule**

`s_c = P_fused_c · S_pres_c^γ_c`. γ = 0 turns gating off for that class alone; γ = 1 is the baseline. ⭐ Not absorbed by lever 2: `w_c` is one constant per dataset, `S_pres` varies tile by tile.

✅ **Identity gate 0.00000** (bar 0.01). Views per tile [1].

| fold | A published | B +τ | C +scale | E +presence | **E − C** |
|---|---|---|---|---|---|
| 1 | 48.12 | 49.49 | 50.51 | 50.89 | **+0.38** |
| 2 | 45.75 | 47.59 | 49.13 | 49.28 | **+0.14** |
| 3 | 48.75 | 49.60 | 50.63 | 50.23 | **-0.40** |
| 4 | 46.41 | 47.20 | 49.32 | 49.88 | **+0.56** |
| 5 | 49.04 | 50.02 | 50.39 | 50.48 | **+0.09** |

- **E − C = +0.15 ± 0.36**, 4/5 folds positive, mean − 2·sd = **-0.57**

## Fitted γ per class

| class | f1 | f2 | f3 | f4 | f5 | mean |
|---|---|---|---|---|---|---|
| background | 2.0 | 0.2 | 2.0 | 0.2 | 2.0 | **1.28** |
| building | 2.0 | 1.4 | 2.0 | 1.4 | 2.0 | **1.76** |
| road | 2.0 | 2.0 | 1.4 | 1.4 | 1.4 | **1.64** |
| water | 1.0 | 1.0 | 0.8 | 1.0 | 0.8 | **0.92** |
| barren | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | **0.80** |
| forest | 0.4 | 0.6 | 0.4 | 0.4 | 0.4 | **0.44** |
| agricultural | 1.2 | 1.0 | 1.2 | 1.2 | 1.2 | **1.16** |

`γ < 1` = this class wants **less** presence gating; `γ > 1` = **more**; `γ = 1` = the published gate is already right.

## Verdict

⛔ **Null: +0.15 ± 0.36**, 4/5 folds positive. A per-class presence weight buys nothing on top of levers 1 and 2.

⭐ **0 of 7 classes keep γ = 1 in every fold** — the published gate is already right for those, and that is a result about the baseline whichever way E − C goes.

⚠️ **`background`'s γ is only weakly identified under `--objective real`**, which does not score it. Read its row as what the fit did to help the real classes, not as a statement about the catch-all.