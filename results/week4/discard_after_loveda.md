# How much of the discard does calibration actually bring back?

- cache `/home/priyanshu/outputs/loveda_full_all/cache` | tiles **1669** | published τ **0.5** | 5-fold, held out | objective **`real`**
- catch-all: **`background`** | real-class pixels scored: **42,938,489** (subsampled 40000/tile)
- ✅ accounting identity verified at every rung: `discard_A − discard_X == recovered − newly discarded`

⚠️ **A subsample, so these are RATES, not the split’s absolute pixel counts.** The rate is what transfers; `measure_discard_rate.py` owns the absolutes at the published τ.

| rung | rule | **discarded** | recovered vs A | of those, **correct** | newly discarded |
|---|---|---|---|---|---|
| **A** | published τ | **29.25%** | — | — | — |
| **B** | per-class τ | **26.77%** | 3.67% | **77.7%** | 1.20% |
| **C** | + per-class scale | **25.01%** | 4.65% | **79.2%** | 0.40% |

## Per class, rung C against the baseline

| class | real px | discard A | discard C | change | recovered | correctly |
|---|---|---|---|---|---|---|
| building | 4,853,002 | 18.7% | 14.4% | **-4.3** | 4.5% | 95.2% |
| road | 3,139,329 | 23.1% | 23.2% | **+0.1** | 0.8% | 0.0% |
| water | 7,747,581 | 31.9% | 18.6% | **-13.3** | 13.4% | 95.0% |
| barren | 3,029,035 | 24.5% | 19.0% | **-5.5** | 5.9% | 88.5% |
| forest | 4,964,898 | 34.2% | 29.8% | **-4.4** | 4.9% | 92.4% |
| agricultural | 19,204,644 | 31.3% | 30.3% | **-1.0** | 1.5% | 0.0% |

## Verdict

**Discard moves 29.25% → 25.01%, a change of -4.24 points.**

The method discards **4.24 points less** than the baseline. Report it beside the newly-discarded column, never alone.

⭐ **Of the pixels it recovers, 79.2% land on the correct class** — 0.26 wrong per right, against **1.73** for the τ→0.1 sweep (WEEK1 §8.2). That comparison is the one worth quoting: the same residual, reached selectively rather than by relaxing a global knob.

⚠️ **Recovered and newly-discarded must always appear together.** Quoting recovery alone is the error §8.2 exists to prevent.