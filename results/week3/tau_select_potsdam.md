# Threshold selection: plain argmax vs inner-CV vs one-standard-error

Published τ = 0.1. Fitted on 200 calibration tiles (5 inner folds); every IoU below is on the 1816 held-out tiles.

| class | τ plain | τ cv | τ 1se | τ oracle | IoU pub | plain | cv | 1se | oracle |
|---|---|---|---|---|---|---|---|---|---|
| `road` | 0.085 | 0.085 | 0.100 | 0.055 | 73.87 | 74.55 | 74.55 | **73.87** | 74.88 |
| `building` | 0.115 | 0.115 | 0.100 | 0.130 | 84.96 | 84.96 | 84.96 | **84.96** | 85.03 |
| `grass` | 0.015 | 0.010 | 0.100 | 0.065 | 57.48 | 57.73 | 57.71 | **57.48** | 58.05 |
| `tree` | 0.055 | 0.050 | 0.100 | 0.035 | 37.90 | 38.23 | 38.24 | **37.90** | 38.25 |
| `car` | 0.660 | 0.550 | 0.275 | 0.405 | 77.03 | 80.29 | 80.79 | **79.79** | 81.21 |

| rule | real-class mIoU | Δ vs published | share of oracle |
|---|---|---|---|
| published τ | 66.25 | +0.00 | — |
| plain argmax *(deployed today)* | 67.15 | +0.90 | 73% |
| inner-CV mean | 67.25 | +1.00 | 81% |
| ⭐ one standard error | 66.80 | +0.55 | 45% |
| oracle *(bound)* | 67.48 | +1.23 | — |

⚠️ The rule is chosen on PRINCIPLE, not on this table. Selecting whichever column scores best on the held-out tiles would be selection on the evaluation set — the same error, one level up. All three are reported so the reader can see the spread.
