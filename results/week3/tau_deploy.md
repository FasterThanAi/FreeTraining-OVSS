# Per-class τ — deployment configuration

- cache: `/home/priyanshu/outputs/week3_fused/cache`  |  seed **0**  |  fit objective **`real`**
- calibration **200** tiles, evaluation **1469** tiles, disjoint
- split file: `/home/priyanshu/splits/loveda_heldout.txt`

## Thresholds

| class | published | fitted |
|---|---|---|
| background *(catch-all)* | 0.5 | — *(cannot change any assignment)* |
| building | 0.5 | **0.190** |
| road | 0.5 | **0.675** |
| water | 0.5 | **0.175** |
| barren | 0.5 | **0.375** |
| forest | 0.5 | **0.410** |
| agricultural | 0.5 | **0.565** |

Spread **0.175–0.675** against a single 0.5. That spread is the argument: one global value is wrong for different classes in opposite directions.

## What the GPU run must reproduce

| rule | predicted mIoU on the held-out tiles |
|---|---|
| published τ = 0.5 | **47.16** |
| per-class τ | **48.35** |
| **Δ** | **+1.18** |

⚠️ Predicted from the cached float16 `conf` binned to 200 levels. Agreement to ~0.01–0.02 confirms the histogram shortcut; a larger gap is a real discrepancy and the method claim waits on it.

## Predicted per-class Δ IoU

| class | Δ |
|---|---|
| background *(catch-all)* | **+0.88** |
| building | **+0.13** |
| road | **-0.54** |
| water | **+6.54** |
| barren | **+1.07** |
| forest | **+0.45** |
| agricultural | **-0.25** |

`background` **+0.88**, the 6 real classes **+7.40** in aggregate.

## Run it

```bash
cd ~/SegEarth-OV-3
# baseline, restricted to the SAME held-out tiles
python eval.py ./configs/cfg_loveda.py \
  --cfg-options test_dataloader.dataset.ann_file=/home/priyanshu/splits/loveda_heldout.txt
# the method
python eval.py ./configs/cfg_loveda_perclass.py
```

`/home/priyanshu/SegEarth-OV-3/configs/cfg_loveda_perclass.py` written.
