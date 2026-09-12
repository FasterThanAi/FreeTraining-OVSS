# Label-free per-class thresholds

- cache: `/home/priyanshu/outputs/oem_tau0.1/cache`  |  tiles: **384**  |  classes: **9**
- published τ: **0.1** → **44.16** mIoU

SegEarth-OV3 tunes one τ per dataset with labels, so a rule that spends **one** label-tuned knob and distributes it across classes by a fixed principle is at parity with the baseline. The oracle row spends **N** independent parameters and is a ceiling, not a competitor.

| rule | knobs | **mIoU** | Δ | share of oracle headroom |
|---|---|---|---|---|
| published τ = 0.1 | 1 (baseline) | **44.16** | — | — |
| best global τ = 0.025 | 1 | **49.31** | +5.15 | 98% |
| per-class Otsu | **0** | **38.36** | -5.80 | -110% |
| equal-commitment, q = 0.01 | 1 | **47.70** | +3.53 | **67%** |
| presence-scaled, level = 0.025 | 1 | **48.94** | +4.77 | **90%** |
| **ORACLE per-class** | 8 | **49.44** | **+5.28** | 100% |

## Thresholds chosen — presence-scaled vs the oracle

| class | label-free τ | oracle τ | IoU published | label-free | Δ |
|---|---|---|---|---|---|
| background | — | — | 17.13 | 61.98 | **+44.85** |
| bareland | 0.008 | 0.010 | 13.77 | 12.95 | **-0.82** |
| grass | 0.036 | 0.030 | 42.92 | 43.27 | **+0.35** |
| pavement | 0.034 | 0.020 | 27.88 | 30.58 | **+2.70** |
| road | 0.035 | 0.025 | 45.88 | 43.78 | **-2.10** |
| tree | 0.036 | 0.025 | 63.91 | 63.70 | **-0.20** |
| water | 0.010 | 0.030 | 66.57 | 66.21 | **-0.36** |
| cropland | 0.005 | 0.055 | 44.08 | 43.80 | **-0.28** |
| building | 0.036 | 0.025 | 75.32 | 74.14 | **-1.18** |

`background` **+44.85**, the 8 real classes **-1.90** in aggregate.

## Verdict

⚠️ **`presence-scaled` gains +4.77 mIoU, but the real classes lose -1.90** — the same background-unwinding pattern as the recovery experiments. A real mIoU gain that is not better land-cover classification. Do not report it without this table.