# Both metrics: full mIoU and catch-all-excluded mIoU

- cache: `/home/priyanshu/outputs/potsdam/cache` | tiles: **2016** | published τ = **0.1** | 5-fold | fit objective **`real`**
- catch-all class: **`clutter`**, 6 classes total

⚠️ **Full mIoU stays the headline** — it is what the literature reports and what makes this comparable to the baseline. The second column is reported *beside* it, never instead of it.

⭐ **The leverage, stated plainly.** mIoU is an unweighted mean over 6 classes, so the catch-all owns exactly **16.7%** of the metric however meaningful that class is. A 10-point move in `clutter` alone is **1.67 mIoU** before anything real has changed.

| | full mIoU | catch-all-excluded mIoU | `clutter` IoU |
|---|---|---|---|
| published τ | 57.87 | 66.05 | 16.96 |
| per-class τ (fitted) | **58.47** | **67.09** | 15.35 |
| **Δ** | **+0.60** | **+1.04** | -1.61 |

## Per class

| class | published τ | fitted | Δ |
|---|---|---|---|
| road | 73.70 | 74.39 | **+0.69** |
| building | 84.84 | 84.67 | **-0.17** |
| grass | 57.27 | 57.92 | **+0.66** |
| tree | 37.62 | 37.95 | **+0.32** |
| car | 76.83 | 80.53 | **+3.70** |
| clutter *(catch-all)* | 16.96 | 15.35 | **-1.61** |

## Reading

✅ **Both metrics agree**: full +0.60, land cover +1.04, catch-all -1.61. The gain is land cover and is not an artefact of the catch-all being repaired.

Baseline gap between the two metrics: **+8.18** — the catch-all sits at 16.96 IoU against a real-class mean of 66.05, so it **depresses** the published headline by that much before any method is applied.