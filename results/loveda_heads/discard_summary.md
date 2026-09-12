# Week 2 — Discard-Rate Diagnostic

- Images: **800**  |  τ: **0.5**
- Presence gating: **on (baseline)**
- mIoU recomputed from confusion matrix: **47.33** (baseline reference: 47.38 — if these disagree, the label alignment is wrong)

## Headline

- Labelled (non-no-data) pixels: **816,825,070**
- Pixels with a real class (excl. background): **521,934,659** (63.9%)
- **Of those, discarded to background: 150,809,669 (28.89%)**

- Per-image discard rate: mean **33.05%**, median 18.67%, max 100.00%

## Loss by class

| Class | GT pixels | Lost to background | % lost |
|---|---|---|---|
| forest | 61,759,846 | 20,389,491 | **33.01%** |
| agricultural | 229,950,553 | 73,572,090 | **31.99%** |
| water | 89,830,710 | 25,865,133 | **28.79%** |
| barren | 39,667,812 | 10,035,799 | **25.3%** |
| road | 39,176,810 | 9,302,910 | **23.75%** |
| building | 61,548,928 | 11,644,246 | **18.92%** |

## Presence-head collapse — catastrophic vs healthy tiles

`spres_max` = highest presence score over the six real classes (max across sliding-window crops).

| Tile set | n | mean spres_max | median | p90 |
|---|---|---|---|---|
| catastrophic (>=99% discard) | 91 | 0.2862 | 0.2432 | 0.5078 |
| healthy (<1% discard) | 32 | 0.9070 | 0.9258 | 0.9609 |

- Correlation(spres_max, discard%) = **-0.769** over 800 tiles.

**How to read it.** If the catastrophic set clusters low (~<0.2) while the healthy set sits high, presence collapse is a systematic failure mode and WEEK1_RESULTS 9.2 generalises from n=1 — a figure, and a claim SegEarth-OV3 does not make. If the two distributions overlap, tile 3487 was an anecdote and must be dropped from the writeup.


## Read this

- **> 15% of real-class pixels lost** → premise confirmed, proceed with the co-occurrence prior.
- **5–15%** → real but modest; the gain ceiling is limited, say so explicitly.
- **< 5%** → premise weak. Pivot to the medium-resolution domain gap (GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).

Compare against the τ-sweep before concluding: if τ=0.1 recovers these pixels without hurting precision, the trivial fix suffices and the method needs a sharper justification.
