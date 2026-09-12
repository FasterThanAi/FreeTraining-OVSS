# Reachability — is the gain bounded by what a threshold can touch?

- cache: `/home/priyanshu/outputs/oem_tau0.1/cache`  |  tag: **SAM3/OEM**  |  tiles: **384**
- published τ: **0.1**  |  classes: **9**, catch-all `background`
- fit objective: **`real`**, 5-fold

`reachable` = the argmax named a real class and only the threshold blocked it. 
`unreachable` = the argmax went to the catch-all, which **no τ vector can change**.

## Label-free — computed without any ground truth

| quantity | pixels | share |
|---|---|---|
| assigned to catch-all | 17,015,313 | 4.53% of labelled px |
| **reachable** | 14,736,278 | **86.61%** of those |
| unreachable | 2,279,035 | 13.39% |

**This is the deployable statistic** — a practitioner can compute it from a forward pass over unlabelled tiles.

## Labelled — the residual proper

| quantity | pixels | share of the residual |
|---|---|---|
| real-class px assigned to catch-all | 14,063,988 | (3.78% of real-class px) |
| reachable | 13,036,683 | **92.70%** |
| self-reachable (recovered with the *right* label) | 4,139,573 | **29.43%** |
| §7.7 (A) below threshold | 13,041,077 | 92.73% |
| §7.7 (B) argmax at conf ≥ τ | 1,022,911 | 7.27% |

⚠️ `reachable` is a **strict subset** of §7.7's mechanism (A): a pixel below the threshold whose argmax was already the catch-all is (A) and still untouchable. The gap between those two rows is the reason this statistic is sharper than the A/B split.

## The gain, 5-fold on this cache

**Δ = +0.16 mIoU, sd 0.93**, 3/5 folds positive, range -1.27 to +1.14.

Computed here rather than transcribed, so the reachability columns and the gain describe the same tiles and the same threshold.

## Per class

| class | GT px | discarded | reachable | **self-reachable** | P−R gap | **Δ IoU** |
|---|---|---|---|---|---|---|
| bareland | 4,807,092 | 3.9% | 99.9% | **67.2%** | -39.2 | **+0.29** |
| grass | 78,421,807 | 4.0% | 99.8% | **24.2%** | +5.7 | **+0.34** |
| pavement | 73,784,417 | 10.0% | 100.0% | **35.2%** | +36.8 | **+2.71** |
| road | 26,308,585 | 1.5% | 100.0% | **26.3%** | -7.2 | **+2.95** |
| tree | 69,984,887 | 1.4% | 100.0% | **25.5%** | -3.8 | **+0.01** |
| water | 8,792,278 | 6.5% | 18.0% | **3.6%** | +11.4 | **-0.24** |
| cropland | 44,049,292 | 1.5% | 19.0% | **2.3%** | +0.2 | **+2.04** |
| building | 66,327,873 | 1.1% | 100.0% | **36.3%** | -17.9 | **+4.35** |
| *background* *(catch-all)* | 3,163,995 | — | — | — | -75.9 | *-11.04* |

## Which per-class statistic ranks the movers?

Over the 8 real classes, against Δ IoU:

| statistic | ρ | p |
|---|---|---|
| **self-reachable share** | **+0.381** | 0.360 *(exact)* |
| precision−recall gap *(§9g's statistic)* | -0.238 | 0.582 *(exact)* |

⚠️ 8 points. This is a direction, not a law — the cross-dataset test in `--summarize` is what the claim rests on.

