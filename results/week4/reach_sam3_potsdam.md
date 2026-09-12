# Reachability — is the gain bounded by what a threshold can touch?

- cache: `/home/priyanshu/outputs/potsdam/cache`  |  tag: **SAM3/Potsdam**  |  tiles: **2016**
- published τ: **0.1**  |  classes: **6**, catch-all `clutter`
- fit objective: **`real`**, 5-fold

`reachable` = the argmax named a real class and only the threshold blocked it. 
`unreachable` = the argmax went to the catch-all, which **no τ vector can change**.

## Label-free — computed without any ground truth

| quantity | pixels | share |
|---|---|---|
| assigned to catch-all | 29,273,047 | 5.97% of labelled px |
| **reachable** | 27,486,594 | **93.90%** of those |
| unreachable | 1,786,453 | 6.10% |

**This is the deployable statistic** — a practitioner can compute it from a forward pass over unlabelled tiles.

## Labelled — the residual proper

| quantity | pixels | share of the residual |
|---|---|---|
| real-class px assigned to catch-all | 21,977,806 | (4.68% of real-class px) |
| reachable | 21,562,794 | **98.11%** |
| self-reachable (recovered with the *right* label) | 11,570,176 | **52.64%** |
| §7.7 (A) below threshold | 21,703,817 | 98.75% |
| §7.7 (B) argmax at conf ≥ τ | 273,989 | 1.25% |

⚠️ `reachable` is a **strict subset** of §7.7's mechanism (A): a pixel below the threshold whose argmax was already the catch-all is (A) and still untouchable. The gap between those two rows is the reason this statistic is sharper than the A/B split.

## The gain, 5-fold on this cache

**Δ = +0.59 mIoU, sd 0.50**, 4/5 folds positive, range -0.26 to +0.98.

Computed here rather than transcribed, so the reachability columns and the gain describe the same tiles and the same threshold.

## Per class

| class | GT px | discarded | reachable | **self-reachable** | P−R gap | **Δ IoU** |
|---|---|---|---|---|---|---|
| road | 153,976,142 | 6.9% | 98.5% | **65.1%** | -4.1 | **+0.70** |
| building | 122,225,020 | 1.8% | 98.3% | **45.3%** | -1.0 | **-0.17** |
| grass | 101,226,664 | 6.5% | 98.1% | **48.7%** | -19.4 | **+0.68** |
| tree | 84,406,481 | 2.9% | 96.2% | **19.2%** | +54.7 | **+0.32** |
| car | 7,785,415 | 2.1% | 97.3% | **6.9%** | -18.0 | **+3.69** |
| *clutter* *(catch-all)* | 21,036,275 | — | — | — | -9.8 | *-1.65* |

## Which per-class statistic ranks the movers?

Over the 5 real classes, against Δ IoU:

| statistic | ρ | p |
|---|---|---|
| **self-reachable share** | -0.100 | 0.950 *(exact)* |
| precision−recall gap *(§9g's statistic)* | -0.600 | 0.350 *(exact)* |

⚠️ 5 points. This is a direction, not a law — the cross-dataset test in `--summarize` is what the claim rests on.

