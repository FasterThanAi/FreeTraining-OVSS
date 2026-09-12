# Threshold selection: plain argmax vs inner-CV vs one-standard-error

Published τ = 0.5. Fitted on 200 calibration tiles (5 inner folds); every IoU below is on the 1469 held-out tiles.

| class | τ plain | τ cv | τ 1se | τ oracle | IoU pub | plain | cv | 1se | oracle |
|---|---|---|---|---|---|---|---|---|---|
| `building` | 0.190 | 0.190 | 0.335 | 0.430 | 64.12 | 64.25 | 64.25 | **64.77** | 64.82 |
| `road` | 0.675 | 0.675 | 0.500 | 0.540 | 53.98 | 53.44 | 53.44 | **53.98** | 54.07 |
| `water` | 0.175 | 0.175 | 0.320 | 0.160 | 50.88 | 57.42 | 57.42 | **54.76** | 57.45 |
| `barren` | 0.375 | 0.375 | 0.500 | 0.375 | 35.20 | 36.27 | 36.27 | **35.20** | 36.27 |
| `forest` | 0.410 | 0.410 | 0.500 | 0.400 | 33.42 | 33.86 | 33.86 | **33.42** | 33.93 |
| `agricultural` | 0.565 | 0.565 | 0.500 | 0.505 | 47.18 | 46.93 | 46.93 | **47.18** | 47.20 |

| rule | real-class mIoU | Δ vs published | share of oracle |
|---|---|---|---|
| published τ | 47.46 | +0.00 | — |
| plain argmax *(deployed today)* | 48.70 | +1.23 | 83% |
| inner-CV mean | 48.70 | +1.23 | 83% |
| ⭐ one standard error | 48.22 | +0.76 | 51% |
| oracle *(bound)* | 48.96 | +1.50 | — |

⚠️ The rule is chosen on PRINCIPLE, not on this table. Selecting whichever column scores best on the held-out tiles would be selection on the evaluation set — the same error, one level up. All three are reported so the reader can see the spread.
