# Pixel accuracy — dlrsd

- 1701 held-out tiles (`~/splits/dlrsd_heldout.txt`), 111,476,736 labelled pixels
- discarded pixels go to index 17 (an unscored sink — never correct)
- computed from the cache; mIoU is shown only to check against eval.py

| rung | pixel accuracy (aAcc) | eval.py aAcc | mIoU | eval.py mIoU | gate | discarded |
|---|---|---|---|---|---|---|
| **A** baseline (published τ) | **59.00** | 58.94 (+0.06) | 37.33 | 37.27 (+0.06) | ✅ | 5.88% |
| **B** + per-class τ | **58.51** |  | 39.09 | 39.04 (+0.05) | ✅ | 12.78% |
| **C** + per-class scale | **65.79** | 65.78 (+0.01) | 44.45 | 44.42 (+0.03) | ✅ | 10.56% |

✅ GATE PASSED — every rung within 0.15 of eval.py (largest gap 0.06), so the accuracy numbers can be trusted.

## Per-class pixel accuracy (% of that class's pixels labelled correctly)

| class | share of pixels | A | B | C | C − A |
|---|---|---|---|---|---|
| grass | 14.89% | 32.25 | 28.97 | 58.71 | **+26.46** |
| bare soil | 10.98% | 21.97 | 22.27 | 44.58 | **+22.62** |
| court | 1.30% | 44.15 | 44.15 | 65.24 | **+21.09** |
| ship | 1.46% | 15.20 | 15.97 | 29.08 | **+13.87** |
| trees | 12.60% | 74.58 | 78.57 | 80.67 | **+6.09** |
| pavement | 25.06% | 82.65 | 87.12 | 88.37 | **+5.72** |
| sea | 3.03% | 53.82 | 51.71 | 57.41 | **+3.59** |
| tanks | 0.83% | 45.37 | 25.45 | 47.42 | **+2.04** |
| mobile home | 1.81% | 0.00 | 0.00 | 0.29 | **+0.29** |
| chaparral | 2.19% | 0.00 | 0.00 | 0.00 | **+0.00** |
| sand | 3.21% | 55.86 | 54.18 | 51.53 | **-4.33** |
| cars | 2.69% | 80.76 | 80.44 | 75.75 | **-5.01** |
| buildings | 10.16% | 86.24 | 84.54 | 80.23 | **-6.01** |
| dock | 0.67% | 75.12 | 67.98 | 68.04 | **-7.08** |
| airplane | 0.32% | 97.65 | 84.93 | 84.89 | **-12.76** |
| field | 4.73% | 47.79 | 36.67 | 32.45 | **-15.34** |
| water | 4.08% | 75.56 | 61.18 | 57.29 | **-18.27** |

⚠️ Per-class accuracy is RECALL: it rises whenever a class is predicted more, even wrongly. Read it beside precision (in the JSON) or IoU, never alone.
