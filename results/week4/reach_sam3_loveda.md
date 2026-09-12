# Reachability — is the gain bounded by what a threshold can touch?

- cache: `/home/priyanshu/outputs/week3_fused/cache`  |  tag: **SAM3/LoveDA**  |  tiles: **1669**
- published τ: **0.5**  |  classes: **7**, catch-all `background`
- fit objective: **`real`**, 5-fold

`reachable` = the argmax named a real class and only the threshold blocked it. 
`unreachable` = the argmax went to the catch-all, which **no τ vector can change**.

## Label-free — computed without any ground truth

| quantity | pixels | share |
|---|---|---|
| assigned to catch-all | 750,084,573 | 44.01% of labelled px |
| **reachable** | 665,222,615 | **88.69%** of those |
| unreachable | 84,861,958 | 11.31% |

**This is the deployable statistic** — a practitioner can compute it from a forward pass over unlabelled tiles.

## Labelled — the residual proper

| quantity | pixels | share of the residual |
|---|---|---|
| real-class px assigned to catch-all | 323,084,415 | (29.67% of real-class px) |
| reachable | 271,873,535 | **84.15%** |
| self-reachable (recovered with the *right* label) | 124,464,326 | **38.52%** |
| §7.7 (A) below threshold | 303,706,238 | 94.00% |
| §7.7 (B) argmax at conf ≥ τ | 19,378,177 | 6.00% |

⚠️ `reachable` is a **strict subset** of §7.7's mechanism (A): a pixel below the threshold whose argmax was already the catch-all is (A) and still untouchable. The gap between those two rows is the reason this statistic is sharper than the A/B split.

## The gain, 5-fold on this cache

**Δ = +1.18 mIoU, sd 0.45**, 5/5 folds positive, range +0.80 to +1.89.

Computed here rather than transcribed, so the reachability columns and the gain describe the same tiles and the same threshold.

## Per class

| class | GT px | discarded | reachable | **self-reachable** | P−R gap | **Δ IoU** |
|---|---|---|---|---|---|---|
| building | 122,805,791 | 18.7% | 99.9% | **56.1%** | -1.4 | **+0.28** |
| road | 79,590,500 | 23.1% | 99.6% | **49.8%** | -1.0 | **+0.10** |
| water | 199,567,816 | 32.2% | 64.2% | **34.6%** | +34.8 | **+6.78** |
| barren | 74,383,133 | 25.0% | 99.7% | **53.0%** | -2.4 | **+0.98** |
| forest | 125,615,647 | 34.6% | 94.3% | **30.2%** | +13.1 | **+0.37** |
| agricultural | 487,082,702 | 31.9% | 83.6% | **36.8%** | +4.9 | **-0.21** |
| *background* *(catch-all)* | 615,250,682 | — | — | — | -12.5 | *-0.01* |

## Which per-class statistic ranks the movers?

Over the 6 real classes, against Δ IoU:

| statistic | ρ | p |
|---|---|---|
| **self-reachable share** | **-0.200** | 0.714 *(exact)* |
| precision−recall gap *(§9g's statistic)* | +0.200 | 0.714 *(exact)* |

⚠️ 6 points. This is a direction, not a law — the cross-dataset test in `--summarize` is what the claim rests on.

