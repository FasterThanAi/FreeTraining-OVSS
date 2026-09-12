# How much of the discard does calibration actually bring back?

- cache `/home/priyanshu/outputs/potsdam_full/cache` | tiles **2016** | published τ **0.1** | 5-fold, held out | objective **`real`**
- catch-all: **`clutter`** | real-class pixels scored: **77,229,879** (subsampled 40000/tile)
- ✅ accounting identity verified at every rung: `discard_A − discard_X == recovered − newly discarded`

⚠️ **A subsample, so these are RATES, not the split’s absolute pixel counts.** The rate is what transfers; `measure_discard_rate.py` owns the absolutes at the published τ.

| rung | rule | **discarded** | recovered vs A | of those, **correct** | newly discarded |
|---|---|---|---|---|---|
| **A** | published τ | **4.69%** | — | — | — |
| **B** | per-class τ | **2.93%** | 2.24% | **61.5%** | 0.48% |
| **C** | + per-class scale | **3.64%** | 2.40% | **64.0%** | 1.35% |

## Per class, rung C against the baseline

| class | real px | discard A | discard C | change | recovered | correctly |
|---|---|---|---|---|---|---|
| road | 25,322,472 | 6.9% | 4.7% | **-2.2** | 3.7% | 85.1% |
| building | 20,035,467 | 1.8% | 2.8% | **+1.0** | 0.5% | 0.0% |
| grass | 16,608,276 | 6.5% | 4.1% | **-2.4** | 3.5% | 44.6% |
| tree | 13,961,407 | 3.0% | 2.5% | **-0.5** | 1.7% | 57.7% |
| car | 1,302,257 | 2.1% | 4.2% | **+2.1** | 1.2% | 0.0% |

## Verdict

**Discard moves 4.69% → 3.64%, a change of -1.05 points.**

The method discards **1.05 points less** than the baseline. Report it beside the newly-discarded column, never alone.

⭐ **Of the pixels it recovers, 64.0% land on the correct class** — 0.56 wrong per right, against **1.73** for the τ→0.1 sweep (WEEK1 §8.2). That comparison is the one worth quoting: the same residual, reached selectively rather than by relaxing a global knob.

⚠️ **Recovered and newly-discarded must always appear together.** Quoting recovery alone is the error §8.2 exists to prevent.