# Week 2 — Discard-Rate Diagnostic

- Images: **500**  |  τ: **0.5**
- Presence gating: **on (baseline)**
- mIoU recomputed from confusion matrix: **47.43** (baseline reference: 47.38 — if these disagree, the label alignment is wrong)

## Headline

- Labelled (non-no-data) pixels: **512,863,683**
- Pixels with a real class (excl. background): **329,672,009** (64.3%)
- **Of those, discarded to background: 91,713,665 (27.82%)**

- Per-image discard rate: mean **31.89%**, median 16.98%, max 100.00%

## Loss by class

| Class | GT pixels | Lost to background | % lost |
|---|---|---|---|
| forest | 40,305,846 | 12,566,206 | **31.18%** |
| water | 59,048,044 | 18,402,504 | **31.17%** |
| agricultural | 142,957,366 | 42,983,167 | **30.07%** |
| barren | 23,297,420 | 5,405,216 | **23.2%** |
| road | 26,198,956 | 5,644,002 | **21.54%** |
| building | 37,864,377 | 6,712,570 | **17.73%** |

## Presence-head collapse — catastrophic vs healthy tiles

`spres_max` = highest presence score over the six real classes (max across sliding-window crops).

| Tile set | n | mean spres_max | median | p90 |
|---|---|---|---|---|
| catastrophic (>=99% discard) | 52 | 0.2975 | 0.2637 | 0.5078 |
| healthy (<1% discard) | 34 | 0.8877 | 0.9199 | 0.9609 |

- Correlation(spres_max, discard%) = **-0.727** over 500 tiles.

**How to read it.** If the catastrophic set clusters low (~<0.2) while the healthy set sits high, presence collapse is a systematic failure mode and WEEK1_RESULTS 9.2 generalises from n=1 — a figure, and a claim SegEarth-OV3 does not make. If the two distributions overlap, tile 3487 was an anecdote and must be dropped from the writeup.


## Read this

- **> 15% of real-class pixels lost** → premise confirmed, proceed with the co-occurrence prior.
- **5–15%** → real but modest; the gain ceiling is limited, say so explicitly.
- **< 5%** → premise weak. Pivot to the medium-resolution domain gap (GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).

Compare against the τ-sweep before concluding: if τ=0.1 recovers these pixels without hurting precision, the trivial fix suffices and the method needs a sharper justification.
