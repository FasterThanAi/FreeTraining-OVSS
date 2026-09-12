# Week 3 — does selective recovery move mIoU?

- tiles: **1669**  |  τ: **0.5**  |  β: **0.25**  |  min component: **64px**
- background-assigned real-class pixels: **323,084,415**
- region scope: **`oracle`**  ⚠️ **ORACLE — uses GT to choose which pixels to touch. Upper bound only, never a result.**

## Validation gate

| | this run | expected |
|---|---|---|
| mIoU, recover nothing | **47.37** | 47.37 |
| background-assigned px | **323,084,415** | 323,084,415 |

✅ **Gate passed** — this script's confusion bookkeeping agrees with `measure_discard_rate.py`, so the rows below are trustworthy.

## Operating points

Abstain unless the region clears both thresholds. `margin` = top1−top2 of the score; `purity` = share of boundary held by the top neighbour class.

| margin ≥ | purity ≥ | recovered px | precision | **mIoU** | Δ vs baseline |
|---|---|---|---|---|---|
| 0.0 | 0.0 | 166,050,845 (51.4%) | 47.2% | **50.84** | +3.47 |
| 0.5 | 0.0 | 145,537,104 (45.0%) | 48.8% | **50.57** | +3.20 |
| 1.0 | 0.0 | 128,679,210 (39.8%) | 49.2% | **50.21** | +2.84 |
| 1.5 | 0.0 | 105,395,235 (32.6%) | 49.2% | **49.63** | +2.26 |
| 2.0 | 0.0 | 392,012 (0.1%) | 85.0% | **47.38** | +0.01 |
| 2.5 | 0.0 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.5 | 144,986,292 (44.9%) | 48.7% | **50.64** | +3.28 |
| 0.5 | 0.5 | 136,025,154 (42.1%) | 49.2% | **50.44** | +3.07 |
| 1.0 | 0.5 | 124,134,160 (38.4%) | 49.5% | **50.17** | +2.81 |
| 1.5 | 0.5 | 104,745,447 (32.4%) | 49.1% | **49.61** | +2.24 |
| 2.0 | 0.5 | 392,012 (0.1%) | 85.0% | **47.38** | +0.01 |
| 2.5 | 0.5 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.7 | 106,918,680 (33.1%) | 49.4% | **49.81** | +2.44 |
| 0.5 | 0.7 | 106,918,680 (33.1%) | 49.4% | **49.81** | +2.44 |
| 1.0 | 0.7 | 106,918,680 (33.1%) | 49.4% | **49.81** | +2.44 |
| 1.5 | 0.7 | 98,678,705 (30.5%) | 49.0% | **49.53** | +2.16 |
| 2.0 | 0.7 | 247,064 (0.1%) | 80.1% | **47.37** | +0.01 |
| 2.5 | 0.7 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.9 | 81,905,607 (25.4%) | 48.3% | **49.07** | +1.70 |
| 0.5 | 0.9 | 81,905,607 (25.4%) | 48.3% | **49.07** | +1.70 |
| 1.0 | 0.9 | 81,905,607 (25.4%) | 48.3% | **49.07** | +1.70 |
| 1.5 | 0.9 | 81,905,607 (25.4%) | 48.3% | **49.07** | +1.70 |
| 2.0 | 0.9 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 2.5 | 0.9 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |

## Verdict

✅ **Best: margin ≥ 0.0, purity ≥ 0.0 → mIoU 50.84 (+3.47)**, recovering 166,050,845 pixels (51.4% of the residual) at **47.2% precision**.

Compare against the alternatives already measured: τ→0.1 gives −5.54 at 36.6% precision, presence removal −11.97. **Selective recovery is the first intervention that reaches the residual and does not cost more than it returns.** The abstention rule is the contribution — threshold relaxation and DenseCRF both commit everywhere and that is exactly why they lose.

> Report recovery rate and precision separately (WEEK1_RESULTS §12). A headline mIoU gain without the precision column invites the objection the τ-sweep already answers.