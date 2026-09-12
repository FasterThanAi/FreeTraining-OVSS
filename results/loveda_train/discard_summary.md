# Week 2 — Discard-Rate Diagnostic

- Images: **2522**  |  τ: **0.5**
- Presence gating: **on (baseline)**
- mIoU recomputed from confusion matrix: **49.56** (baseline reference: 47.38 — if these disagree, the label alignment is wrong)

## Headline

- Labelled (non-no-data) pixels: **2,544,876,613**
- Pixels with a real class (excl. background): **1,634,168,283** (64.2%)
- **Of those, discarded to background: 237,545,456 (14.54%)**

- Per-image discard rate: mean **16.31%**, median 9.20%, max 100.00%

## Loss by class

| Class | GT pixels | Lost to background | % lost |
|---|---|---|---|
| barren | 132,748,857 | 49,626,594 | **37.38%** |
| building | 281,493,525 | 66,094,374 | **23.48%** |
| road | 134,119,839 | 25,484,553 | **19.0%** |
| water | 162,729,743 | 24,468,898 | **15.04%** |
| forest | 409,074,984 | 46,794,364 | **11.44%** |
| agricultural | 514,001,335 | 25,076,673 | **4.88%** |

## Presence-head collapse — catastrophic vs healthy tiles

`spres_max` = highest presence score over the six real classes (max across sliding-window crops).

| Tile set | n | mean spres_max | median | p90 |
|---|---|---|---|---|
| catastrophic (>=99% discard) | 32 | 0.4916 | 0.4736 | 0.7723 |
| healthy (<1% discard) | 421 | 0.9210 | 0.9414 | 0.9883 |

- Correlation(spres_max, discard%) = **-0.291** over 2522 tiles.

**How to read it.** If the catastrophic set clusters low (~<0.2) while the healthy set sits high, presence collapse is a systematic failure mode and WEEK1_RESULTS 9.2 generalises from n=1 — a figure, and a claim SegEarth-OV3 does not make. If the two distributions overlap, tile 3487 was an anecdote and must be dropped from the writeup.


## Read this

- **> 15% of real-class pixels lost** → premise confirmed, proceed with the co-occurrence prior.
- **5–15%** → real but modest; the gain ceiling is limited, say so explicitly.
- **< 5%** → premise weak. Pivot to the medium-resolution domain gap (GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).

Compare against the τ-sweep before concluding: if τ=0.1 recovers these pixels without hurting precision, the trivial fix suffices and the method needs a sharper justification.
