# Pixel accuracy — does the PICTURE get better, not just the class average?

**17–18 Sep 2026.** `scripts/pixel_accuracy.py`, `fig_pixel_accuracy.py`, `fig_pixel_maps.py`.
CPU only, computed from the cached score stack on the **same held-out tiles `eval.py` ran on**,
with the same fitted vectors read out of the deployed configs.

---

## 1. Why this exists

Every headline in this project is **mIoU — a mean over classes**. A reader is entitled to ask
the other question: *of all the pixels in the image, how many more are now right?* That is
**overall pixel accuracy (`aAcc`)**, and it is **weighted by pixel count**, so a rare class
cannot move it. The two metrics can disagree, and where they disagree the disagreement is the
finding.

**Definitions, exactly as mmseg's `IoUMetric` computes them:**

| | formula | reads as |
|---|---|---|
| `aAcc` | Σ_c (correct pixels of c) ÷ Σ_c (pixels labelled c) | *share of all labelled pixels that are right* |
| per-class `Acc` | correct pixels of c ÷ pixels labelled c | **recall** of class c |
| IoU | TP ÷ (TP + FP + FN) | what the headline averages |

⚠️ Pixels with no ground truth (mask value 0) are excluded from every column, as mmseg excludes
them.

---

## 2. The instrument, and its gate

The script streams each held-out tile, applies the three decision rules to the cached scores,
and accumulates one confusion matrix per rung. **It prints its own mIoU next to `eval.py`'s and
refuses to pass if any rung is off by more than 0.15**, because if the mIoU disagrees the
accuracy cannot be trusted either.

| gate | measured | eval.py | Δ |
|---|---|---|---|
| DLRSD mIoU A / B / C | 37.33 / 39.09 / 44.45 | 37.27 / 39.04 / 44.42 | ≤ 0.06 |
| ⭐ **DLRSD aAcc A / C** | **59.00 / 65.79** | **58.94 / 65.78** | ⭐ **0.06 / 0.01** |
| Potsdam mIoU A / B / C | 57.63 / 58.40 / 63.30 | 57.60 / 58.35 / 63.27 | ≤ 0.05 |

✅ Both datasets passed. ⭐ **DLRSD's `aAcc` is verified against the pipeline directly**, so the
instrument is checked on the metric it reports, not only on mIoU.
✅ Correctness of the arithmetic is asserted against a brute-force per-pixel loop on two
synthetic datasets — one with an unscored sink (DLRSD's shape) and one with a scored catch-all
(Potsdam's) — in the tests shipped beside the script.

---

## 3. The headline table

| % of all labelled pixels correct | A baseline | B + per-class τ | C + per-class scale | C − A |
|---|---|---|---|---|
| **DLRSD** (1701 tiles) | 59.00 | **58.51** | **65.79** | ⭐ **+6.79** |
| **Potsdam** (1816 tiles) | 76.99 | **76.98** | **80.76** | ⭐ **+3.77** |

| discarded share of labelled pixels | A | B | C |
|---|---|---|---|
| DLRSD | 5.88% | **12.78%** | 10.56% |
| Potsdam | 5.65% | 5.73% | 4.89% |

> ⭐⭐ **The method improves the picture, not just the class average.** This is the answer to
> *"your mIoU rose because you fixed three tiny classes"* — `aAcc` cannot be moved that way.

---

## 4. ⭐ THE THEORY: lever 1 *cannot* raise pixel accuracy on DLRSD, and this is provable

Lever 1 changes **no argmax**. It only decides whether the class that already won is kept or
sent to `bg_idx`. On DLRSD `bg_idx` is an **unscored sink**, so a discarded pixel is wrong for
its true class, always. For one pixel there are exactly two cases:

| the argmax was | raising τ_c discards it → | effect on aAcc |
|---|---|---|
| **correct** | correct → discarded | ⛔ **−1** |
| **wrong** | wrong → discarded | **0** |

and symmetrically, lowering τ_c can only turn a discard into a correct pixel (+1) or into a
wrong one (0). So:

> **On a dataset whose discard target is unscored, `aAcc` is monotone in the threshold vector:
> it can only fall as any τ_c rises. The aAcc-optimal threshold vector is therefore τ = 0 for
> every class — "never discard anything".**

That is not what mIoU wants, because mIoU counts a false positive against the class that made
it. The fit (objective = real-class mIoU) raised most DLRSD thresholds — discards go
**5.88% → 12.78%** — and pays **−0.49 aAcc** to buy **+1.77 mIoU**. Both numbers are correct;
they price different things.

⭐ **A second exact statement, useful for reading the per-class figure.** Under rung B the
argmax is fixed, so a class's recall depends on **its own threshold alone**:

```
Acc_c(B) = |{pixels: gt = c, argmax = c, raw score ≥ τ_c}| / |{gt = c}|
```

**A class's bar therefore moves down if and only if its own threshold went up.** No
cross-class effect can be involved, and that is visible in the figure:

| DLRSD class | A | B | so its τ went |
|---|---|---|---|
| `water` | 75.56 | **61.18** | **up** |
| `airplane` | 97.65 | **84.93** | **up** (deliberately — see §6) |
| `field` | 47.79 | 36.67 | up |
| `tanks` | 45.37 | 25.45 | up |
| `pavement` | 82.65 | **87.12** | **down** (fitted τ = 0.000) |
| `trees` | 74.58 | 78.57 | down |

⛔ **Potsdam does not obey the monotone rule**, and the reason is its label design: `bg_idx = 5`
**is** `clutter`, a scored class. A discard there is a `clutter` prediction, which is *correct*
wherever the truth is clutter. Lever 1 is consequently near-exactly neutral (−0.01) rather than
negative, and `clutter`'s own recall *rises* (33.38 → 33.96).

---

## 5. ⭐ Why lever 2 CAN raise it

Lever 2 changes **which class wins**, so a wrong pixel can become a right one — the one thing a
threshold can never do. That is where all of the pixel-accuracy gain comes from: **+7.28 on
DLRSD and +3.78 on Potsdam**, measured from rung B.

| biggest per-class accuracy gains, C over A | | |
|---|---|---|
| Potsdam `tree` | 38.72 → **66.85** | **+28.1** |
| DLRSD `grass` | 32.25 → **58.71** | **+26.5** |
| DLRSD `bare soil` | 21.97 → **44.58** | +22.6 |
| DLRSD `court` | 44.15 → **65.24** | +21.1 |
| DLRSD `ship` | 15.20 → **29.08** | +13.9 |

| and the losses | | |
|---|---|---|
| DLRSD `water` | 75.56 → **57.29** | ⛔ **−18.3** |
| DLRSD `field` | 47.79 → 32.45 | −15.3 |
| DLRSD `airplane` | 97.65 → 84.89 | −12.8 *(intended)* |
| Potsdam `car` | 96.91 → 91.76 | −5.1 |
| Potsdam `grass` | 83.65 → 79.85 | −3.8 |

---

## 6. ⚠️ How to read a per-class accuracy drop

**Per-class accuracy is recall.** It rises whenever a class is predicted more — including
wrongly — so a fall is not automatically a loss:

- ⭐ **`airplane` is supposed to fall.** It fired almost everywhere: recall **97.65%** at
  precision **59.03%**. A threshold of 0.955 takes it to **85.05 / 84.87** — its **IoU rises
  +15.6** while its accuracy bar drops 12.8. The bar shows the cost; the IoU shows the point.
- ⛔ **`water`'s fall is a genuine loss.** Its IoU falls too (−2.73 under both levers), which is
  why it is named as this dataset's worst class everywhere it is quoted.

⛔ **Never read the per-class accuracy figure alone.** Read it beside precision and IoU, both of
which the JSON carries.

---

## 7. ⭐⭐ Where the two metrics disagree, and why it matters

Widening lever 2's search range (@ARGMAX_SCALING_RESULTS.md) gives the sharpest case:

| DLRSD, both verified by `eval.py` | mIoU | aAcc |
|---|---|---|
| search range 0.40–2.50 *(the method)* | 44.42 | **65.79** |
| search range 0.10–10 | ⭐ **45.52** | ⛔ **63.66** |

**The wider range gains 1.10 mIoU and loses 2.13 points of pixel accuracy.** Four rare classes
covering 7.3% of the pixels gain 34 IoU between them; `buildings`, `water` and `pavement` —
39.4% of the pixels — pay for it. mIoU counts every class once and rewards the trade; `aAcc`
counts pixels and refuses it.

> ⭐ **This is `WEEK3 §9h`'s leverage argument applied to our own method.** It is the reason the
> headline stays on the pre-registered range, and it is a caution the paper should state
> plainly: **a class-averaged metric can be improved by decisions that make the average picture
> worse.**

⚠️ Potsdam's comparison is cross-source — 80.76 from the cache against 80.48 from `eval.py` on
the wider range — so the −0.28 there is indicative, not a like-for-like measurement, until the
default range's `aAcc` is read off a pipeline run.

---

## 8. The right/wrong maps

`fig_pixel_maps.py` colours **every pixel** by outcome, baseline against ours:
green = correct · red = a different class · dark grey = discarded · white = no ground truth,
plus a fifth panel showing what the method **changed**: green fixed, red broken, yellow one
error swapped for another.

- `docs/pixel_maps_dlrsd/sheet.png`, `docs/pixel_maps_potsdam/sheet.png` — four tiles each,
  chosen to argue rather than to flatter: the biggest gain, the **biggest loss**, a tile the
  method barely touches, and a median tile.
- `--all` writes every held-out tile plus an `index.csv` sorted worst-first.

⭐ Tiles are selected by the script, not by hand: DLRSD's loss panel is `buildings57`
(94.1% → 21.3%) and Potsdam's is `5_13_5488_3072_6000_3584` (88.5% → 45.8%). **A failure panel
stays in the sheet** — the five-fold gains are averages, and a sheet of four wins would
misrepresent them.

---

## 9. Reproducing it

```bash
python scripts/pixel_accuracy.py --preset dlrsd     # writes results/dlrsd/pixel_accuracy.{json,md}
python scripts/pixel_accuracy.py --preset potsdam
python scripts/fig_pixel_accuracy.py \
  --json results/dlrsd/pixel_accuracy.json results/potsdam/pixel_accuracy.json \
  --out docs/fig_pixel_accuracy                     # add --rungs A,C for baseline vs ours only
python scripts/fig_pixel_maps.py --preset dlrsd --auto --out docs/pixel_maps_dlrsd
```

The plot script **refuses to draw** a JSON whose gate failed, so a figure cannot be produced
from numbers that disagree with `eval.py`.

---

## 10. What is not measured yet

⚠️ **LoveDA and UAVid have no `aAcc` row from this instrument.** UAVid's is known from its
`eval.py` run (79.24 → 84.81, @UAVID_RESULTS.md §18) but not the per-class accuracy table;
LoveDA has neither. Each is one CPU pass once its deploy config and held-out split are pointed
at the script.
⚠️ Potsdam's `aAcc` has not been read off a pipeline run on the **default** range (§7).
