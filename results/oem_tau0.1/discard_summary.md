# Week 2 — Discard-Rate Diagnostic

- Images: **384**  |  τ: **0.1**
- Presence gating: **on (baseline)**
- mIoU recomputed from confusion matrix: **44.16** (no published reference for this config — record it as the new baseline)

## Headline

- Labelled (non-no-data) pixels: **375,640,226**
- Pixels with a real class (excl. background): **372,476,231** (99.2%)
- **Of those, discarded to background: 14,062,186 (3.78%)**

- Per-image discard rate: mean **3.81%**, median 2.40%, max 40.99%

## Loss by class

| Class | GT pixels | Lost to background | % lost |
|---|---|---|---|
| pavement | 73,784,417 | 7,394,189 | **10.02%** |
| water | 8,792,278 | 571,547 | **6.5%** |
| grass | 78,421,807 | 3,149,858 | **4.02%** |
| bareland | 4,807,092 | 186,974 | **3.89%** |
| cropland | 44,049,292 | 680,066 | **1.54%** |
| road | 26,308,585 | 395,523 | **1.5%** |
| tree | 69,984,887 | 969,709 | **1.39%** |
| building | 66,327,873 | 714,320 | **1.08%** |

## Presence-head collapse — catastrophic vs healthy tiles

`spres_max` = highest presence score over the six real classes (max across sliding-window crops).

| Tile set | n | mean spres_max | median | p90 |
|---|---|---|---|---|
| catastrophic (>=99% discard) | 0 | - | - | - |
| healthy (<1% discard) | 128 | 0.9708 | 0.9844 | 0.9961 |

- Correlation(spres_max, discard%) = **+0.094** over 384 tiles.

**How to read it.** If the catastrophic set clusters low (~<0.2) while the healthy set sits high, presence collapse is a systematic failure mode and WEEK1_RESULTS 9.2 generalises from n=1 — a figure, and a claim SegEarth-OV3 does not make. If the two distributions overlap, tile 3487 was an anecdote and must be dropped from the writeup.


## Read this

- **> 15% of real-class pixels lost** → premise confirmed, proceed with the co-occurrence prior.
- **5–15%** → real but modest; the gain ceiling is limited, say so explicitly.
- **< 5%** → premise weak. Pivot to the medium-resolution domain gap (GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).

Compare against the τ-sweep before concluding: if τ=0.1 recovers these pixels without hurting precision, the trivial fix suffices and the method needs a sharper justification.
