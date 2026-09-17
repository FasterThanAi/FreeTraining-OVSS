# Pixel accuracy — potsdam

- 1816 held-out tiles (`~/splits/potsdam_reorder_heldout.txt`), 441,779,185 labelled pixels
- discarded pixels go to index 5 (`clutter`, scored)
- computed from the cache; mIoU is shown only to check against eval.py

| rung | pixel accuracy (aAcc) | eval.py aAcc | mIoU | eval.py mIoU | gate | discarded |
|---|---|---|---|---|---|---|
| **A** baseline (published τ) | **76.99** |  | 57.63 | 57.60 (+0.03) | ✅ | 5.65% |
| **B** + per-class τ | **76.98** |  | 58.40 | 58.35 (+0.05) | ✅ | 5.73% |
| **C** + per-class scale | **80.76** |  | 63.30 | 63.27 (+0.03) | ✅ | 4.89% |

✅ GATE PASSED — every rung within 0.15 of eval.py (largest gap 0.05), so the accuracy numbers can be trusted.

## Per-class pixel accuracy (% of that class's pixels labelled correctly)

| class | share of pixels | A | B | C | C − A |
|---|---|---|---|---|---|
| tree | 17.42% | 38.72 | 39.29 | 66.85 | **+28.13** |
| road | 31.01% | 86.66 | 85.46 | 87.18 | **+0.52** |
| building | 25.03% | 92.37 | 91.51 | 91.10 | **-1.27** |
| clutter | 4.30% | 33.38 | 33.96 | 30.79 | **-2.59** |
| grass | 20.67% | 83.65 | 86.20 | 79.85 | **-3.80** |
| car | 1.58% | 96.91 | 91.97 | 91.76 | **-5.15** |

⚠️ Per-class accuracy is RECALL: it rises whenever a class is predicted more, even wrongly. Read it beside precision (in the JSON) or IoU, never alone.
