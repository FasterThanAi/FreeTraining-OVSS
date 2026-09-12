# Scaled argmax — deployment fit and prediction

- cache: `/home/priyanshu/outputs/loveda_full_all/cache`  |  tiles: **1669**  |  calibration: **200**, held out: **1469** (seed 0)
- published τ: **0.5**  |  objective **`real`**

## Predicted, on the held-out tiles

| rung | scale | thresholds | mIoU |
|---|---|---|---|
| baseline | — | published 0.5 | 47.64 |
| **per-class τ** | — | fitted | **47.66** (+0.02) |
| **+ class scale** | fitted | fitted | **49.27** (+1.63) |

⭐ **The increment that matters is +1.32** — the scale over the thresholds, both measured on the same subsampled pixels. Beating the baseline proves nothing here; per-class τ already does that.

⚠️ Rungs 1–2 are exact over every pixel; rung 3 is over a 40000-px/tile subsample. Rung 2 recomputed on that same subsample gives 47.96, a drift of **0.003** mIoU — the increment above uses the subsampled rung 2, so the sampling difference is not credited to the scale.

## Fitted parameters

| class | scale | threshold |
|---|---|---|
| background *(catch-all)* | **0.412** | 0.230 |
| building | **0.594** | 0.430 |
| road | **0.594** | 0.230 |
| water | **2.145** | 0.140 |
| barren | **1.238** | 0.255 |
| forest | **2.145** | 0.395 |
| agricultural | **1.205** | 0.415 |

## Predicted per-class Δ IoU

| class | τ over baseline | scale over τ |
|---|---|---|
| background | -4.92 | **+1.55** |
| building | +0.69 | **+0.09** |
| road | -0.67 | **-0.10** |
| water | +6.43 | **+3.97** |
| barren | -0.26 | **+0.46** |
| forest | +0.10 | **+3.06** |
| agricultural | -1.22 | **+0.17** |

## Run it — three passes, same tiles

```bash
cd ~/SegEarth-OV-3
# 1. baseline, restricted to the held-out tiles
python eval.py ./configs/cfg_loveda.py \
  --cfg-options test_dataloader.dataset.ann_file=/home/priyanshu/splits/loveda_reorder_heldout.txt
# 2. per-class thresholds only
python eval.py ./configs/cfg_loveda_tauonly.py
# 3. thresholds + scale
python eval.py ./configs/cfg_loveda_reorder.py
```

Each printed mIoU should match its predicted row above. Per-class values may differ by a few hundredths: the cache stores `conf` as float16 and rung 3 is subsampled.

`/home/priyanshu/SegEarth-OV-3/configs/cfg_loveda_tauonly.py` written.
`/home/priyanshu/SegEarth-OV-3/configs/cfg_loveda_reorder.py` written.

⛔ If pass 3 does not reproduce its prediction, **do not adjust the number** — find out which side is wrong. `verify_argmax_reorder.py` says the two rules are identical, so a disagreement means the config, the split, or the cache is not what this script assumed.
