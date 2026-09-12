# Week 2 — Discard-Rate Diagnostic

- Images: **1669**  |  τ: **0.5**
- mIoU recomputed from confusion matrix: **47.37** (baseline reference: 47.38 — if these disagree, the label alignment is wrong)

## Headline

- Labelled (non-no-data) pixels: **1,704,296,271**
- Pixels with a real class (excl. background): **1,089,045,589** (63.9%)
- **Of those, discarded to background: 323,184,908 (29.68%)**

- Per-image discard rate: mean **33.79%**, median 18.51%, max 100.00%

## Loss by class

| Class | GT pixels | Lost to background | % lost |
|---|---|---|---|
| forest | 125,615,647 | 43,462,196 | **34.6%** |
| water | 199,567,816 | 64,309,668 | **32.22%** |
| agricultural | 487,082,702 | 155,414,274 | **31.91%** |
| barren | 74,383,133 | 18,578,671 | **24.98%** |
| road | 79,590,500 | 18,432,732 | **23.16%** |
| building | 122,805,791 | 22,987,367 | **18.72%** |

## Presence-head collapse — catastrophic vs healthy tiles

`spres_max` = highest presence score over the six real classes (max across sliding-window crops).

| Tile set | n | mean spres_max | median | p90 |
|---|---|---|---|---|
| catastrophic (>=99% discard) | 198 | 0.3125 | 0.2734 | 0.5699 |
| healthy (<1% discard) | 77 | 0.8886 | 0.9180 | 0.9672 |

- Correlation(spres_max, discard%) = **-0.750** over 1669 tiles.

**How to read it.** If the catastrophic set clusters low (~<0.2) while the healthy set sits high, presence collapse is a systematic failure mode and WEEK1_RESULTS 9.2 generalises from n=1 — a figure, and a claim SegEarth-OV3 does not make. If the two distributions overlap, tile 3487 was an anecdote and must be dropped from the writeup.


## Read this

- **> 15% of real-class pixels lost** → premise confirmed, proceed with the co-occurrence prior.
- **5–15%** → real but modest; the gain ceiling is limited, say so explicitly.
- **< 5%** → premise weak. Pivot to the medium-resolution domain gap (GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).

Compare against the τ-sweep before concluding: if τ=0.1 recovers these pixels without hurting precision, the trivial fix suffices and the method needs a sharper justification.
