# Week 3 — does selective recovery move mIoU?

- tiles: **1669**  |  τ: **0.5**  |  β: **0.25**  |  min component: **64px**
- background-assigned real-class pixels: **323,084,415**
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
| 0.0 | 0.0 | 568,900,398 (176.1%) | 11.2% | **32.00** | -15.37 |
| 0.5 | 0.0 | 454,016,355 (140.5%) | 11.8% | **34.90** | -12.47 |
| 1.0 | 0.0 | 342,305,809 (105.9%) | 11.8% | **37.70** | -9.67 |
| 1.5 | 0.0 | 239,308,815 (74.1%) | 11.5% | **40.37** | -7.00 |
| 2.0 | 0.0 | 1,405,462 (0.4%) | 31.6% | **47.36** | -0.01 |
| 2.5 | 0.0 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.5 | 397,897,758 (123.2%) | 11.2% | **36.03** | -11.34 |
| 0.5 | 0.5 | 372,258,224 (115.2%) | 11.4% | **36.73** | -10.64 |
| 1.0 | 0.5 | 316,919,724 (98.1%) | 11.5% | **38.26** | -9.11 |
| 1.5 | 0.5 | 236,303,726 (73.1%) | 11.4% | **40.42** | -6.95 |
| 2.0 | 0.5 | 1,405,462 (0.4%) | 31.6% | **47.36** | -0.01 |
| 2.5 | 0.5 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.7 | 231,802,998 (71.7%) | 10.8% | **40.29** | -7.08 |
| 0.5 | 0.7 | 231,802,998 (71.7%) | 10.8% | **40.29** | -7.08 |
| 1.0 | 0.7 | 231,802,998 (71.7%) | 10.8% | **40.29** | -7.08 |
| 1.5 | 0.7 | 214,879,059 (66.5%) | 10.9% | **40.81** | -6.56 |
| 2.0 | 0.7 | 605,602 (0.2%) | 7.7% | **47.36** | -0.01 |
| 2.5 | 0.7 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 0.0 | 0.9 | 171,235,004 (53.0%) | 10.7% | **41.81** | -5.56 |
| 0.5 | 0.9 | 171,235,004 (53.0%) | 10.7% | **41.81** | -5.56 |
| 1.0 | 0.9 | 171,235,004 (53.0%) | 10.7% | **41.81** | -5.56 |
| 1.5 | 0.9 | 171,235,004 (53.0%) | 10.7% | **41.81** | -5.56 |
| 2.0 | 0.9 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |
| 2.5 | 0.9 | 0 (0.0%) | 0.0% | **47.37** | +0.00 |

## Verdict

⛔ **No operating point beats the baseline** (47.37). Recovering the residual by neighbour propagation costs more in background IoU than it returns in real-class recall, at every abstention level tested. That is a fourth refuted intervention alongside τ-relaxation and presence removal — a strong motivation section, but not a method. **Re-scope before Week 4.**

> Report recovery rate and precision separately (WEEK1_RESULTS §12). A headline mIoU gain without the precision column invites the objection the τ-sweep already answers.