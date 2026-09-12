# Both metrics: full mIoU and catch-all-excluded mIoU

- cache: `/home/priyanshu/outputs/oem_tau0.1/cache` | tiles: **384** | published τ = **0.1** | 5-fold | fit objective **`real`**
- catch-all class: **`background`**, 9 classes total

⚠️ **Full mIoU stays the headline** — it is what the literature reports and what makes this comparable to the baseline. The second column is reported *beside* it, never instead of it.

⭐ **The leverage, stated plainly.** mIoU is an unweighted mean over 9 classes, so the catch-all owns exactly **11.1%** of the metric however meaningful that class is. A 10-point move in `background` alone is **1.11 mIoU** before anything real has changed.

| | full mIoU | catch-all-excluded mIoU | `background` IoU |
|---|---|---|---|
| published τ | 44.16 | 47.54 | 17.13 |
| per-class τ (fitted) | **44.47** | **49.30** | 5.83 |
| **Δ** | **+0.30** | **+1.75** | -11.30 |

## Per class

| class | published τ | fitted | Δ |
|---|---|---|---|
| background *(catch-all)* | 17.13 | 5.83 | **-11.30** |
| bareland | 13.77 | 15.68 | **+1.91** |
| grass | 42.92 | 43.27 | **+0.35** |
| pavement | 27.88 | 30.59 | **+2.71** |
| road | 45.88 | 48.82 | **+2.94** |
| tree | 63.91 | 63.95 | **+0.04** |
| water | 66.57 | 65.81 | **-0.77** |
| cropland | 44.08 | 46.58 | **+2.50** |
| building | 75.32 | 79.67 | **+4.34** |

## Reading

⛔ **The headline is not measuring land cover here.** Full mIoU moves +0.30, and that splits into **+1.56 from the 8 real classes** and **-1.26 from `background` alone** (-11.30 IoU spread over 9 classes). The catch-all **cancels most of a real gain**, so the headline understates the method. Quoting full mIoU without the second column would mislead.

Baseline gap between the two metrics: **+3.38** — the catch-all sits at 17.13 IoU against a real-class mean of 47.54, so it **depresses** the published headline by that much before any method is applied.