# Reachability — is the gain bounded by what a threshold can touch?

- cache: `/home/priyanshu/outputs/coninfer_loveda/cache`  |  tag: **ConInfer/LoveDA**  |  tiles: **1669**
- published τ: **0.8**  |  classes: **7**, catch-all `background`
- fit objective: **`real`**, 5-fold

`reachable` = the argmax named a real class and only the threshold blocked it. 
`unreachable` = the argmax went to the catch-all, which **no τ vector can change**.

## Label-free — computed without any ground truth

| quantity | pixels | share |
|---|---|---|
| assigned to catch-all | 432,170,856 | 25.36% of labelled px |
| **reachable** | 395,157,208 | **91.44%** of those |
| unreachable | 37,013,648 | 8.56% |

**This is the deployable statistic** — a practitioner can compute it from a forward pass over unlabelled tiles.

## Labelled — the residual proper

| quantity | pixels | share of the residual |
|---|---|---|
| real-class px assigned to catch-all | 225,342,578 | (20.69% of real-class px) |
| reachable | 201,504,655 | **89.42%** |
| self-reachable (recovered with the *right* label) | 94,983,679 | **42.15%** |
| §7.7 (A) below threshold | 205,961,984 | 91.40% |
| §7.7 (B) argmax at conf ≥ τ | 19,380,594 | 8.60% |

⚠️ `reachable` is a **strict subset** of §7.7's mechanism (A): a pixel below the threshold whose argmax was already the catch-all is (A) and still untouchable. The gap between those two rows is the reason this statistic is sharper than the A/B split.

## The gain, 5-fold on this cache

**Δ = +2.51 mIoU, sd 0.34**, 5/5 folds positive, range +2.22 to +3.03.

Computed here rather than transcribed, so the reachability columns and the gain describe the same tiles and the same threshold.

## Per class

| class | GT px | discarded | reachable | **self-reachable** | P−R gap | **Δ IoU** |
|---|---|---|---|---|---|---|
| building | 122,805,791 | 14.4% | 75.5% | **48.1%** | -27.4 | **+0.42** |
| road | 79,590,500 | 18.7% | 97.1% | **55.6%** | -31.0 | **+2.47** |
| water | 199,567,816 | 24.6% | 82.1% | **36.0%** | +27.7 | **+4.21** |
| barren | 74,383,133 | 32.9% | 94.3% | **31.3%** | -9.8 | **+0.76** |
| forest | 125,615,647 | 17.9% | 98.4% | **54.7%** | -41.3 | **+2.01** |
| agricultural | 487,082,702 | 19.8% | 91.2% | **41.9%** | +8.6 | **+1.69** |
| *background* *(catch-all)* | 615,250,682 | — | — | — | +14.2 | *+6.03* |

## Which per-class statistic ranks the movers?

Over the 6 real classes, against Δ IoU:

| statistic | ρ | p |
|---|---|---|
| **self-reachable share** | **+0.200** | 0.714 *(exact)* |
| precision−recall gap *(§9g's statistic)* | +0.143 | 0.803 *(exact)* |

⚠️ 6 points. This is a direction, not a law — the cross-dataset test in `--summarize` is what the claim rests on.

