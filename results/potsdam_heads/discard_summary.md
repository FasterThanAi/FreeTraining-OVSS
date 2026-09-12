# Week 2 — Discard-Rate Diagnostic

- Images: **2016**  |  τ: **0.1**
- Presence gating: **on (baseline)**
- mIoU recomputed from confusion matrix: **57.87** (no published reference for this config — record it as the new baseline)

## Headline

- Labelled (non-no-data) pixels: **490,655,997**
- Pixels with a real class (excl. background): **469,619,722** (95.7%)
- **Of those, discarded to background: 21,975,006 (4.68%)**

- Per-image discard rate: mean **4.77%**, median 1.19%, max 94.53%

## Loss by class

| Class | GT pixels | Lost to background | % lost |
|---|---|---|---|
| road | 153,976,142 | 10,597,805 | **6.88%** |
| grass | 101,226,664 | 6,548,490 | **6.47%** |
| tree | 84,406,481 | 2,479,904 | **2.94%** |
| car | 7,785,415 | 166,008 | **2.13%** |
| building | 122,225,020 | 2,182,799 | **1.79%** |

## Presence-head collapse — catastrophic vs healthy tiles

`spres_max` = highest presence score over the six real classes (max across sliding-window crops).

| Tile set | n | mean spres_max | median | p90 |
|---|---|---|---|---|
| catastrophic (>=99% discard) | 0 | - | - | - |
| healthy (<1% discard) | 935 | 0.8525 | 0.9219 | 0.9961 |

- Correlation(spres_max, discard%) = **-0.031** over 2016 tiles.

**How to read it.** If the catastrophic set clusters low (~<0.2) while the healthy set sits high, presence collapse is a systematic failure mode and WEEK1_RESULTS 9.2 generalises from n=1 — a figure, and a claim SegEarth-OV3 does not make. If the two distributions overlap, tile 3487 was an anecdote and must be dropped from the writeup.


## Read this

- **> 15% of real-class pixels lost** → premise confirmed, proceed with the co-occurrence prior.
- **5–15%** → real but modest; the gain ceiling is limited, say so explicitly.
- **< 5%** → premise weak. Pivot to the medium-resolution domain gap (GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).

Compare against the τ-sweep before concluding: if τ=0.1 recovers these pixels without hurting precision, the trivial fix suffices and the method needs a sharper justification.
