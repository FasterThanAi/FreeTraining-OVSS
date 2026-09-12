# Both metrics: full mIoU and catch-all-excluded mIoU

- cache: `/home/priyanshu/outputs/coninfer_loveda/cache` | tiles: **1669** | published τ = **0.8** | 5-fold | fit objective **`real`**
- catch-all class: **`background`**, 7 classes total

⚠️ **Full mIoU stays the headline** — it is what the literature reports and what makes this comparable to the baseline. The second column is reported *beside* it, never instead of it.

⭐ **The leverage, stated plainly.** mIoU is an unweighted mean over 7 classes, so the catch-all owns exactly **14.3%** of the metric however meaningful that class is. A 10-point move in `background` alone is **1.43 mIoU** before anything real has changed.

| | full mIoU | catch-all-excluded mIoU | `background` IoU |
|---|---|---|---|
| published τ | 37.00 | 39.06 | 24.61 |
| per-class τ (fitted) | **39.52** | **41.00** | 30.61 |
| **Δ** | **+2.52** | **+1.94** | +6.01 |

## Per class

| class | published τ | fitted | Δ |
|---|---|---|---|
| background *(catch-all)* | 24.61 | 30.61 | **+6.01** |
| building | 47.89 | 48.33 | **+0.43** |
| road | 36.10 | 38.62 | **+2.52** |
| water | 53.72 | 57.92 | **+4.20** |
| barren | 18.73 | 19.50 | **+0.78** |
| forest | 29.38 | 31.42 | **+2.04** |
| agricultural | 48.54 | 50.22 | **+1.69** |

## Reading

✅ **Both metrics agree**: full +2.52, land cover +1.94, catch-all +6.01. The gain is land cover and is not an artefact of the catch-all being repaired.

Baseline gap between the two metrics: **+2.07** — the catch-all sits at 24.61 IoU against a real-class mean of 39.06, so it **depresses** the published headline by that much before any method is applied.