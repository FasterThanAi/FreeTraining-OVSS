# Per-tile wins and losses — 2100 tiles

| | tiles | mean Δ | median | total mass |
|---|---|---|---|---|
| improved | **1057** | **+6.89** | +3.29 | **+7,283** |
| worse | **769** | **-4.29** | -1.32 | **-3,302** |
| unchanged | 274 | — | — | — |
| **net** | | **+1.90** | +0.07 | **+3,980** |

⭐ **Win mass ÷ loss mass = 2.21x.** The count ratio is 1.37x, and it is the MASS that decides the dataset number, not the count.

## Which scenes lose — a cause, or noise?

| scene category | tiles | mean Δ | worse |
|---|---|---|---|
| `agricultural` | 100 | **-6.68** | 26/100 |
| `river` | 100 | **-2.84** | 65/100 |
| `parkinglot` | 100 | **-1.51** | 89/100 |
| `buildings` | 100 | **-0.98** | 53/100 |
| `runway` | 100 | **-0.59** | 39/100 |
| `mobilehomepark` | 100 | **-0.37** | 66/100 |
| `airplane` | 100 | **-0.16** | 34/100 |
| `harbor` | 100 | **+0.10** | 49/100 |

*(best few, for contrast)*

| scene category | tiles | mean Δ | worse |
|---|---|---|---|
| `baseballdiamond` | 100 | **+13.53** | 4/100 |
| `golfcourse` | 100 | **+9.91** | 15/100 |
| `forest` | 100 | **+8.97** | 5/100 |

⚠️ **Losses are spread across categories** (worst is `agricultural` at 3.4% of them), so there is no single scene type to blame — the fitted vector is too coarse for part of the dataset.

## Is the damage a WRONG label, or a discarded one?

- pixels newly sent to the sink: **11,131,177**
- pixels that were right and became wrong: **4,704,016**
- ⛔ tiles discarded **entirely**: **22** — Counter({'agricultural': 18, 'beach': 2, 'chaparral': 1, 'golfcourse': 1})

⛔ **A tile that goes to zero with nothing mislabelled is a different failure from a tile that goes to zero wrongly labelled**, and only the second is a segmentation error. Both are real costs; they need different fixes.

## ⚠️ What this report does NOT settle

Nothing here is the dataset metric. **mIoU, `aAcc` and precision/recall are measured by `eval.py` over every pixel** and are what the claim rests on; this is a per-tile view whose averaging differs. Quote it to explain WHERE the method helps and hurts, never to argue whether it does.
