# Both metrics: full mIoU and catch-all-excluded mIoU

- cache: `/home/priyanshu/outputs/week3_fused/cache` | tiles: **1669** | published τ = **0.5** | 5-fold | fit objective **`real`**
- catch-all class: **`background`**, 7 classes total

⚠️ **Full mIoU stays the headline** — it is what the literature reports and what makes this comparable to the baseline. The second column is reported *beside* it, never instead of it.

⭐ **The leverage, stated plainly.** mIoU is an unweighted mean over 7 classes, so the catch-all owns exactly **14.3%** of the metric however meaningful that class is. A 10-point move in `background` alone is **1.43 mIoU** before anything real has changed.

| | full mIoU | catch-all-excluded mIoU | `background` IoU |
|---|---|---|---|
| published τ | 47.37 | 47.68 | 45.51 |
| per-class τ (fitted) | **48.53** | **49.04** | 45.48 |
| **Δ** | **+1.16** | **+1.36** | -0.02 |

## Per class

| class | published τ | fitted | Δ |
|---|---|---|---|
| background *(catch-all)* | 45.51 | 45.48 | **-0.02** |
| building | 63.80 | 64.08 | **+0.27** |
| road | 53.88 | 53.99 | **+0.10** |
| water | 51.41 | 58.06 | **+6.65** |
| barren | 35.75 | 36.70 | **+0.94** |
| forest | 33.75 | 34.14 | **+0.40** |
| agricultural | 47.48 | 47.27 | **-0.22** |

## Reading

✅ **Both metrics agree**: full +1.16, land cover +1.36, catch-all -0.02. The gain is land cover and is not an artefact of the catch-all being repaired.

Baseline gap between the two metrics: **+0.31** — the catch-all sits at 45.51 IoU against a real-class mean of 47.68, so it **depresses** the published headline by that much before any method is applied.