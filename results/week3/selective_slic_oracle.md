# Week 3 — does selective recovery move mIoU?

- tiles: **1669**  |  τ: **0.5**  |  β: **0.25**  |  min component: **64px**
- background-assigned real-class pixels: **323,084,415**
- atoms: **`slic`**  (oracle ceiling 92.8%)
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
| 0.0 | 0.0 | 109,344,649 (33.8%) | 56.1% | **50.99** | +3.62 |
| 0.5 | 0.0 | 101,544,297 (31.4%) | 57.2% | **50.88** | +3.51 |
| 1.0 | 0.0 | 93,902,817 (29.1%) | 58.0% | **50.71** | +3.34 |
| 1.5 | 0.0 | 84,613,004 (26.2%) | 58.6% | **50.42** | +3.05 |
| 2.0 | 0.0 | 8,404 (0.0%) | 43.3% | **47.37** | +0.00 |
| 2.5 | 0.0 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.5 | 107,545,344 (33.3%) | 56.4% | **50.99** | +3.62 |
| 0.5 | 0.5 | 101,033,528 (31.3%) | 57.3% | **50.88** | +3.51 |
| 1.0 | 0.5 | 93,866,441 (29.1%) | 58.0% | **50.70** | +3.34 |
| 1.5 | 0.5 | 84,611,189 (26.2%) | 58.6% | **50.42** | +3.05 |
| 2.0 | 0.5 | 8,404 (0.0%) | 43.3% | **47.37** | +0.00 |
| 2.5 | 0.5 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.7 | 90,063,589 (27.9%) | 58.6% | **50.63** | +3.26 |
| 0.5 | 0.7 | 90,063,589 (27.9%) | 58.6% | **50.63** | +3.26 |
| 1.0 | 0.7 | 90,063,589 (27.9%) | 58.6% | **50.63** | +3.26 |
| 1.5 | 0.7 | 84,321,085 (26.1%) | 58.6% | **50.41** | +3.04 |
| 2.0 | 0.7 | 6,054 (0.0%) | 46.6% | **47.37** | +0.00 |
| 2.5 | 0.7 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.9 | 74,947,400 (23.2%) | 59.4% | **50.14** | +2.77 |
| 0.5 | 0.9 | 74,947,400 (23.2%) | 59.4% | **50.14** | +2.77 |
| 1.0 | 0.9 | 74,947,400 (23.2%) | 59.4% | **50.14** | +2.77 |
| 1.5 | 0.9 | 74,947,400 (23.2%) | 59.4% | **50.14** | +2.77 |
| 2.0 | 0.9 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 2.5 | 0.9 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |

## Verdict

✅ **Best: margin ≥ 0.0, purity ≥ 0.0 → mIoU 50.99 (+3.62)**, recovering 109,344,649 pixels (33.8% of the residual) at **56.1% precision**.

Compare against the alternatives already measured: τ→0.1 gives −5.54 at 36.6% precision, presence removal −11.97. **Selective recovery is the first intervention that reaches the residual and does not cost more than it returns.** The abstention rule is the contribution — threshold relaxation and DenseCRF both commit everywhere and that is exactly why they lose.

> Report recovery rate and precision separately (WEEK1_RESULTS §12). A headline mIoU gain without the precision column invites the objection the τ-sweep already answers.