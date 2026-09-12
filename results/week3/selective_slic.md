# Week 3 — does selective recovery move mIoU?

- tiles: **1669**  |  τ: **0.5**  |  β: **0.25**  |  min component: **64px**
- background-assigned real-class pixels: **323,084,415**
- atoms: **`slic`**  (oracle ceiling 92.8%)
- region scope: **`all`**  (every background-assigned pixel, as at inference)

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
| 0.0 | 0.0 | 287,510,669 (89.0%) | 20.7% | **41.18** | -6.19 |
| 0.5 | 0.0 | 262,827,515 (81.3%) | 21.2% | **41.76** | -5.60 |
| 1.0 | 0.0 | 238,134,617 (73.7%) | 21.6% | **42.33** | -5.04 |
| 1.5 | 0.0 | 208,915,772 (64.7%) | 22.0% | **42.95** | -4.42 |
| 2.0 | 0.0 | 16,093 (0.0%) | 15.0% | **47.37** | -0.00 |
| 2.5 | 0.0 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.5 | 280,321,463 (86.8%) | 20.9% | **41.38** | -5.99 |
| 0.5 | 0.5 | 260,772,332 (80.7%) | 21.2% | **41.82** | -5.55 |
| 1.0 | 0.5 | 237,978,896 (73.7%) | 21.6% | **42.34** | -5.03 |
| 1.5 | 0.5 | 208,905,926 (64.7%) | 22.0% | **42.95** | -4.42 |
| 2.0 | 0.5 | 16,093 (0.0%) | 15.0% | **47.37** | -0.00 |
| 2.5 | 0.5 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.7 | 226,062,265 (70.0%) | 21.8% | **42.61** | -4.76 |
| 0.5 | 0.7 | 226,062,265 (70.0%) | 21.8% | **42.61** | -4.76 |
| 1.0 | 0.7 | 226,062,265 (70.0%) | 21.8% | **42.61** | -4.76 |
| 1.5 | 0.7 | 208,070,051 (64.4%) | 22.0% | **42.97** | -4.40 |
| 2.0 | 0.7 | 16,093 (0.0%) | 15.0% | **47.37** | -0.00 |
| 2.5 | 0.7 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.9 | 183,690,116 (56.9%) | 22.1% | **43.42** | -3.95 |
| 0.5 | 0.9 | 183,690,116 (56.9%) | 22.1% | **43.42** | -3.95 |
| 1.0 | 0.9 | 183,690,116 (56.9%) | 22.1% | **43.42** | -3.95 |
| 1.5 | 0.9 | 183,690,116 (56.9%) | 22.1% | **43.42** | -3.95 |
| 2.0 | 0.9 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 2.5 | 0.9 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |

## Verdict

⛔ **No operating point beats the baseline** (47.37). Recovering the residual by neighbour propagation costs more in background IoU than it returns in real-class recall, at every abstention level tested. That is a fourth refuted intervention alongside τ-relaxation and presence removal — a strong motivation section, but not a method. **Re-scope before Week 4.**

> Report recovery rate and precision separately (WEEK1_RESULTS §12). A headline mIoU gain without the precision column invites the objection the τ-sweep already answers.