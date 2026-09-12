# Reachability — is the gain bounded by what a threshold can touch?

- cache: `/home/priyanshu/outputs/coninfer_oem/cache`  |  tag: **ConInfer/OEM**  |  tiles: **384**
- published τ: **0.1**  |  classes: **9**, catch-all `background`
- fit objective: **`real`**, 5-fold

`reachable` = the argmax named a real class and only the threshold blocked it. 
`unreachable` = the argmax went to the catch-all, which **no τ vector can change**.

## Label-free — computed without any ground truth

| quantity | pixels | share |
|---|---|---|
| assigned to catch-all | 8,166,096 | 2.17% of labelled px |
| **reachable** | 0 | **0.00%** of those |
| unreachable | 8,166,096 | 100.00% |

**This is the deployable statistic** — a practitioner can compute it from a forward pass over unlabelled tiles.

## Labelled — the residual proper

| quantity | pixels | share of the residual |
|---|---|---|
| real-class px assigned to catch-all | 6,520,227 | (1.75% of real-class px) |
| reachable | 0 | **0.00%** |
| self-reachable (recovered with the *right* label) | 0 | **0.00%** |
| §7.7 (A) below threshold | 0 | 0.00% |
| §7.7 (B) argmax at conf ≥ τ | 6,520,227 | 100.00% |

⚠️ `reachable` is a **strict subset** of §7.7's mechanism (A): a pixel below the threshold whose argmax was already the catch-all is (A) and still untouchable. The gap between those two rows is the reason this statistic is sharper than the A/B split.

## The gain, 5-fold on this cache

**Δ = -0.39 mIoU, sd 1.28**, 1/5 folds positive, range -1.32 to +1.82.

Computed here rather than transcribed, so the reachability columns and the gain describe the same tiles and the same threshold.

## Per class

| class | GT px | discarded | reachable | **self-reachable** | P−R gap | **Δ IoU** |
|---|---|---|---|---|---|---|
| bareland | 4,807,092 | 1.1% | 0.0% | **0.0%** | -45.8 | **+5.88** |
| grass | 78,421,807 | 0.9% | 0.0% | **0.0%** | +24.2 | **-0.00** |
| pavement | 73,784,417 | 3.4% | 0.0% | **0.0%** | +24.4 | **+0.00** |
| road | 26,308,585 | 1.3% | 0.0% | **0.0%** | -0.6 | **+0.02** |
| tree | 69,984,887 | 0.6% | 0.0% | **0.0%** | -6.0 | **+0.01** |
| water | 8,792,278 | 1.1% | 0.0% | **0.0%** | -30.0 | **+1.42** |
| cropland | 44,049,292 | 0.1% | 0.0% | **0.0%** | +11.4 | **+0.26** |
| building | 66,327,873 | 3.4% | 0.0% | **0.0%** | -32.4 | **+0.04** |
| *background* *(catch-all)* | 3,163,995 | — | — | — | -31.9 | *-11.16* |

## Which per-class statistic ranks the movers?

Over the 8 real classes, against Δ IoU:

| statistic | ρ | p |
|---|---|---|
| **self-reachable share** | — *(undefined: constant)* |
| precision−recall gap *(§9g's statistic)* | -0.762 | 0.037 *(exact)* |

⚠️ 8 points. This is a direction, not a law — the cross-dataset test in `--summarize` is what the claim rests on.

