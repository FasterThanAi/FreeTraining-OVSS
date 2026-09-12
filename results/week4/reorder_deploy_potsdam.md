# Scaled argmax — deployment fit and prediction

- cache: `/home/priyanshu/outputs/potsdam_full/cache`  |  tiles: **2016**  |  calibration: **200**, held out: **1816** (seed 0)
- published τ: **0.1**  |  objective **`real`**

## Predicted, on the held-out tiles

| rung | scale | thresholds | mIoU |
|---|---|---|---|
| baseline | — | published 0.1 | 57.63 |
| **per-class τ** | — | fitted | **58.40** (+0.78) |
| **+ class scale** | fitted | fitted | **63.31** (+5.68) |

⚠️ Rung 3 is quoted **debiased**: the exact rung 2 plus the subsampled increment. Its raw subsampled absolute is 63.37, which carries the subsample's own offset and is NOT what eval.py prints — the offset cancels in the increment but not in the level.

⭐ **The increment that matters is +4.90** — the scale over the thresholds, both measured on the same subsampled pixels. Beating the baseline proves nothing here; per-class τ already does that.

⚠️ Rungs 1–2 are exact over every pixel; rung 3 is over a 40000-px/tile subsample. Rung 2 recomputed on that same subsample gives 58.47, a drift of **0.017** mIoU — the increment above uses the subsampled rung 2, so the sampling difference is not credited to the scale.

## Fitted parameters

| class | scale `w` | τ **without** scale *(rung 2 cfg)* | τ **under** scale *(rung 3 cfg)* |
|---|---|---|---|
| road | **0.594** | 0.110 | 0.080 |
| building | **0.597** | 0.130 | 0.130 |
| grass | **0.713** | 0.060 | 0.060 |
| tree | **3.731** | 0.020 | 0.055 |
| car | **0.412** | 0.490 | 0.490 |
| clutter *(catch-all)* | **2.573** | 0.110 | 0.060 |

The segmentor echoes whichever vector it loads, so `grep "per-class prob_thd" <log>` should match the third column for the tau-only config and the fourth for the scaled one.


## Predicted per-class IoU — compare directly against eval.py

| class | A published τ | B per-class τ | C + scale | B−A | C−B |
|---|---|---|---|---|---|
| road | 73.21 | 72.67 | 75.30 | -0.54 | **+2.49** |
| building | 84.84 | 84.80 | 84.86 | -0.04 | **+0.00** |
| grass | 56.94 | 57.65 | 61.99 | +0.72 | **+4.40** |
| tree | 37.65 | 37.95 | 59.14 | +0.30 | **+21.11** |
| car | 76.79 | 80.80 | 82.90 | +4.02 | **+2.02** |
| clutter *(catch-all)* | 16.34 | 16.55 | 16.02 | +0.21 | **-0.60** |

⚠️ Every column is a **prediction**. Compare each against the matching `eval.py` table class by class — an aggregate that agrees can hide two classes that disagree in opposite directions, and an aggregate that disagrees says nothing about which class caused it.


## Run it — three passes, same tiles

```bash
cd ~/SegEarth-OV-3
# 1. baseline, restricted to the held-out tiles
python eval.py ./configs/cfg_potsdam.py \
  --cfg-options test_dataloader.dataset.ann_file=/home/priyanshu/splits/potsdam_reorder_heldout.txt
# 2. per-class thresholds only
python eval.py ./configs/cfg_potsdam_tauonly.py
# 3. thresholds + scale
python eval.py ./configs/cfg_potsdam_reorder.py
```

Each printed mIoU should match its predicted row above. Per-class values may differ by a few hundredths: the cache stores `conf` as float16 and rung 3 is subsampled.

`/home/priyanshu/SegEarth-OV-3/configs/cfg_potsdam_tauonly.py` written.
`/home/priyanshu/SegEarth-OV-3/configs/cfg_potsdam_reorder.py` written.

⛔ If pass 3 does not reproduce its prediction, **do not adjust the number** — find out which side is wrong. `verify_argmax_reorder.py` says the two rules are identical, so a disagreement means the config, the split, or the cache is not what this script assumed.
