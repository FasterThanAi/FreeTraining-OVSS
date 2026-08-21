# Week 2 — Discard-Rate Diagnostic

- Images: **1669**  |  τ: **0.3**
- mIoU recomputed from confusion matrix: **46.64** (baseline reference: 47.38 — if these disagree, the label alignment is wrong)

## Headline

- Labelled (non-no-data) pixels: **1,704,296,271**
- Pixels with a real class (excl. background): **1,089,045,589** (63.9%)
- **Of those, discarded to background: 223,826,505 (20.55%)**

- Per-image discard rate: mean **23.53%**, median 8.39%, max 100.00%

## Loss by class

| Class | GT pixels | Lost to background | % lost |
|---|---|---|---|
| agricultural | 487,082,702 | 117,814,134 | **24.19%** |
| water | 199,567,816 | 44,809,675 | **22.45%** |
| forest | 125,615,647 | 24,343,686 | **19.38%** |
| road | 79,590,500 | 13,497,374 | **16.96%** |
| building | 122,805,791 | 15,965,267 | **13.0%** |
| barren | 74,383,133 | 7,396,369 | **9.94%** |

## Read this

- **> 15% of real-class pixels lost** → premise confirmed, proceed with the co-occurrence prior.
- **5–15%** → real but modest; the gain ceiling is limited, say so explicitly.
- **< 5%** → premise weak. Pivot to the medium-resolution domain gap (GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).

Compare against the τ-sweep before concluding: if τ=0.1 recovers these pixels without hurting precision, the trivial fix suffices and the method needs a sharper justification.
