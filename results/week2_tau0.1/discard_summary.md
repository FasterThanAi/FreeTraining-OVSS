# Week 2 — Discard-Rate Diagnostic

- Images: **1669**  |  τ: **0.1**
- mIoU recomputed from confusion matrix: **41.83** (baseline reference: 47.38 — if these disagree, the label alignment is wrong)

## Headline

- Labelled (non-no-data) pixels: **1,704,296,271**
- Pixels with a real class (excl. background): **1,089,045,589** (63.9%)
- **Of those, discarded to background: 118,477,557 (10.88%)**

- Per-image discard rate: mean **12.06%**, median 0.39%, max 100.00%

## Loss by class

| Class | GT pixels | Lost to background | % lost |
|---|---|---|---|
| water | 199,567,816 | 28,739,040 | **14.4%** |
| agricultural | 487,082,702 | 70,119,725 | **14.4%** |
| forest | 125,615,647 | 9,001,259 | **7.17%** |
| road | 79,590,500 | 3,710,022 | **4.66%** |
| building | 122,805,791 | 5,549,284 | **4.52%** |
| barren | 74,383,133 | 1,358,227 | **1.83%** |

## Read this

- **> 15% of real-class pixels lost** → premise confirmed, proceed with the co-occurrence prior.
- **5–15%** → real but modest; the gain ceiling is limited, say so explicitly.
- **< 5%** → premise weak. Pivot to the medium-resolution domain gap (GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).

Compare against the τ-sweep before concluding: if τ=0.1 recovers these pixels without hurting precision, the trivial fix suffices and the method needs a sharper justification.
