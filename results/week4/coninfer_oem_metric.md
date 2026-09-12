# Both metrics: full mIoU and catch-all-excluded mIoU

- cache: `/home/priyanshu/outputs/coninfer_oem/cache` | tiles: **384** | published τ = **0.1** | 5-fold | fit objective **`real`**
- catch-all class: **`background`**, 9 classes total

⚠️ **Full mIoU stays the headline** — it is what the literature reports and what makes this comparable to the baseline. The second column is reported *beside* it, never instead of it.

⭐ **The leverage, stated plainly.** mIoU is an unweighted mean over 9 classes, so the catch-all owns exactly **11.1%** of the metric however meaningful that class is. A 10-point move in `background` alone is **1.11 mIoU** before anything real has changed.

| | full mIoU | catch-all-excluded mIoU | `background` IoU |
|---|---|---|---|
| published τ | 29.90 | 31.51 | 17.00 |
| per-class τ (fitted) | **29.81** | **32.78** | 6.06 |
| **Δ** | **-0.09** | **+1.27** | -10.94 |

## Per class

| class | published τ | fitted | Δ |
|---|---|---|---|
| background *(catch-all)* | 17.00 | 6.06 | **-10.94** |
| bareland | 9.32 | 17.50 | **+8.18** |
| grass | 22.12 | 22.12 | **-0.00** |
| pavement | 16.67 | 16.67 | **+0.00** |
| road | 22.89 | 22.91 | **+0.02** |
| tree | 47.88 | 47.90 | **+0.02** |
| water | 45.38 | 46.92 | **+1.54** |
| cropland | 44.44 | 44.77 | **+0.33** |
| building | 43.41 | 43.45 | **+0.04** |

## Reading

⛔ **The headline is not measuring land cover here.** Full mIoU moves -0.09, and that splits into **+1.13 from the 8 real classes** and **-1.22 from `background` alone** (-10.94 IoU spread over 9 classes). The catch-all **cancels most of a real gain**, so the headline understates the method. Quoting full mIoU without the second column would mislead.

Baseline gap between the two metrics: **+1.61** — the catch-all sits at 17.00 IoU against a real-class mean of 31.51, so it **depresses** the published headline by that much before any method is applied.