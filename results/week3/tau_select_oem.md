# Threshold selection: plain argmax vs inner-CV vs one-standard-error

Published τ = 0.1. Fitted on 100 calibration tiles (5 inner folds); every IoU below is on the 284 held-out tiles.

| class | τ plain | τ cv | τ 1se | τ oracle | IoU pub | plain | cv | 1se | oracle |
|---|---|---|---|---|---|---|---|---|---|
| `bareland` | 0.185 | 0.185 | 0.100 | 0.250 | 11.53 | 14.67 | 14.67 | **11.53** | 16.57 |
| `grass` | 0.050 | 0.050 | 0.100 | 0.040 | 43.30 | 43.64 | 43.64 | **43.30** | 43.65 |
| `pavement` | 0.015 | 0.015 | 0.085 | 0.020 | 27.65 | 30.26 | 30.26 | **28.65** | 30.26 |
| `road` | 0.310 | 0.305 | 0.100 | 0.550 | 46.05 | 49.09 | 49.06 | **46.05** | 49.64 |
| `tree` | 0.220 | 0.220 | 0.100 | 0.140 | 62.99 | 62.76 | 62.76 | **62.99** | 63.16 |
| `water` | 0.710 | 0.720 | 0.100 | 0.240 | 69.18 | 55.02 | 54.08 | **69.18** | 69.84 |
| `cropland` | 0.550 | 0.555 | 0.100 | 0.520 | 42.12 | 45.41 | 45.33 | **42.12** | 45.59 |
| `building` | 0.375 | 0.375 | 0.275 | 0.385 | 75.44 | 79.66 | 79.66 | **78.79** | 79.66 |

| rule | real-class mIoU | Δ vs published | share of oracle |
|---|---|---|---|
| published τ | 47.28 | +0.00 | — |
| plain argmax *(deployed today)* | 47.56 | +0.28 | 11% |
| inner-CV mean | 47.43 | +0.15 | 6% |
| ⭐ one standard error | 47.83 | +0.54 | 22% |
| oracle *(bound)* | 49.80 | +2.51 | — |

⚠️ The rule is chosen on PRINCIPLE, not on this table. Selecting whichever column scores best on the held-out tiles would be selection on the evaluation set — the same error, one level up. All three are reported so the reader can see the spread.
