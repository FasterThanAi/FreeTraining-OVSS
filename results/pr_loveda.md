# Per-class precision / recall / IoU

A = published global tau 0.5.  B = per-class tau, fitted on 200 calibration tiles, table on 1469 held out.

| class | tau A | tau B | prec A | prec B | recall A | recall B | IoU A | IoU B | Δ IoU |
|---|---|---|---|---|---|---|---|---|---|
| `water` | 0.500 | 0.175 | 89.7 | 86.3 | 54.0 | 63.2 | 50.9 | 57.4 | **+6.54** |
| `barren` | 0.500 | 0.375 | 50.9 | 48.5 | 53.3 | 59.0 | 35.2 | 36.3 | **+1.07** |
| `background` *(catch-all)* | 0.500 | — | 56.6 | 57.7 | 69.6 | 70.0 | 45.4 | 46.2 | **+0.88** |
| `forest` | 0.500 | 0.410 | 57.3 | 54.0 | 44.5 | 47.6 | 33.4 | 33.9 | **+0.45** |
| `building` | 0.500 | 0.190 | 77.5 | 72.1 | 78.8 | 85.5 | 64.1 | 64.3 | **+0.13** |
| `agricultural` | 0.500 | 0.565 | 67.0 | 71.2 | 61.5 | 57.9 | 47.2 | 46.9 | **-0.25** |
| `road` | 0.500 | 0.675 | 69.7 | 73.4 | 70.6 | 66.3 | 54.0 | 53.4 | **-0.54** |

mIoU  47.16 → 48.35   ·   excluding the catch-all  47.46 → 48.70

⚠️ The catch-all has no fitted threshold: sub-threshold pixels are assigned *to* it, so no threshold applies to it.

⚠️ This cache stores a histogram, not the score stack, so rung C (per-class scale) cannot be computed here. A scale changes the argmax, and the histogram fixes the argmax at the published one. Re-run the cache with `--cache-full` for the three-rung table.
