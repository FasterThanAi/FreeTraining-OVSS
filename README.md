# FreeTraining-OVSS

**Training-free open-vocabulary semantic segmentation for remote sensing**, built on **SAM 3**.
Final-year project, IIITDM Kurnool. *Last updated 19 Sep 2026 — read this page first.*

You give the model an aerial image and a list of class names typed as words. It returns a
labelled mask. **No model weights are trained and no annotation is used for the segmentation
itself.**

> **Terminology.** This is *training-free, annotation-free, open-vocabulary* segmentation, **not
> unsupervised** — the class names are supplied. `ANALYSIS.md` §3.6.

---

## 1. What we contribute, in one paragraph

The baseline (SegEarth-OV3) scores every class separately, keeps the highest-scoring class per
pixel, and throws the pixel away — to `background` — when that score falls below **one
threshold used for every class**. We show that single threshold is the wrong *shape*, and
replace the decision rule with two small parameter sets fitted on a few hundred labelled tiles,
the same supervision the baseline already spends tuning its own threshold:

| | what it changes | where it acts |
|---|---|---|
| **Lever 1 — per-class threshold** | whether the winning class is kept or discarded | **after** the argmax |
| **Lever 2 — per-class scale** | **which class wins** the pixel | **at** the argmax |

**Nothing else changes** — same model, same weights, same forward pass, no extra inference cost.

---

## 2. Results — five datasets, four verified by the official evaluator

Every number below is measured on **held-out tiles**, with the parameters fitted on a disjoint
calibration split and the baseline recomputed on the *same* tiles.

| dataset | baseline (ours) | + lever 1 | + lever 2 | total | verified by `eval.py` |
|---|---|---|---|---|---|
| **LoveDA** (1469 held out) | 47.65 | 47.68 | **49.02** | **+1.37** | ✅ |
| **Potsdam** (1816) | 57.60 | 58.35 | **63.27** | **+5.67** | ✅ |
| **UAVid** (70 frames) | 56.86 | 57.92 | **63.55** | **+6.69** | ✅ |
| **DLRSD** (1701) | 37.27 | 39.04 | **44.42** | **+7.15** | ✅ |
| OpenEarthMap (384) | 44.16 | 44.47 | — | +0.30 *(+1.75 excl. catch-all)* | ⚠️ underpowered |
| ConInfer (CLIP, not SAM 3) | 36.99 | **39.52** | −0.10 | **+2.51** | 5-fold |

**The five-fold figures, which carry an error bar and are the headline:**

| dataset | lever 1 | lever 2 (on top) |
|---|---|---|
| LoveDA | +1.18 ± 0.45 | +1.16 ± 0.19 |
| Potsdam | +0.59 ± 0.50 | **+4.86 ± 0.35** |
| UAVid | +1.34 ± 0.39 | **+5.89 ± 1.51** |
| DLRSD | **+2.37 ± 0.63** | **+5.84 ± 1.03** |
| ConInfer (CLIP) | **+2.51 ± 0.34** | −0.10 (a clean null) |

⭐ **It is not a SAM 3 quirk.** Lever 1 also works on ConInfer, a CLIP-based competitor: their
36.99 → 39.52. We *improve* the nearest published method rather than beat it.
⭐ **Pixel accuracy rises too**, so this is not an artefact of averaging classes: DLRSD
58.94 → **65.79**, Potsdam 76.99 → **80.76**, UAVid 79.24 → **84.81**.

⚠️ **Always quote the per-class table with any mean.** On DLRSD, six classes covering under 2%
of the pixels own 35% of the metric. Real per-class losses exist and are reported — DLRSD
`water` −2.73, LoveDA `road` −0.53.

---

## 3. The four findings worth presenting

1. **One threshold is worth nothing; N thresholds are worth a lot.** On DLRSD the *best
   possible* single threshold is worth **+0.04** mIoU, and per-class thresholds **+2.99**.
   LoveDA's global row is also +0.04. The level was already right; the shape was wrong.
2. **The two levers do different jobs, and only the second can fix a stolen pixel.** A
   threshold can discard the winner; it can never hand the pixel to the class that should have
   won. Potsdam `tree` (precision 93 / recall 39) moved **+0.32** under thresholds and
   **+21.6** under the scale.
3. ⭐ **The typed vocabulary is the largest single lever in this pipeline, and there is no rule
   for choosing it.** One word on UAVid (`vegetation` → `low vegetation`) is worth **+3.53**;
   the *same* rule on Potsdam costs **−2.71**. Every training-free paper inherits a hand-written
   class list and none report it.
4. **What the model cannot be told, it cannot fix.** DLRSD's `chaparral` scores 0.00 IoU under
   every threshold and scale; renaming it to `shrubs` reaches 13.4. `mobile home` is the
   opposite: no word helps, but a large enough scale does.

**Honest limits, stated everywhere we quote a gain:** the parameters do not transfer across a
domain shift (LoveDA rural +2.77 vs urban +0.10, and the wrong domain's values are worse than
none); calibration must come from the distribution being evaluated; one parameter set is right
on average and wrong on individual scenes (DLRSD: 1057 tiles improve, 769 get worse, 22 are
emptied entirely).

---

## 4. Where everything lives

| file | what is in it |
|---|---|
| **`CLAUDE.md`** | the working state of the project, newest first. **Start here for detail.** |
| **`LOGBOOK.md`** | one entry per working day, in plain language |
| `WEEK1_RESULTS.md` | baseline reproduction, the discard diagnostic, presence gating |
| `WEEK3_RESULTS.md` | the mechanism (catch-all share), the method, the label-free bound |
| `ARGMAX_SCALING_RESULTS.md` | lever 2 on every dataset, and the search-range ablation |
| `POTSDAM_RESULTS.md` · `UAVID_RESULTS.md` · `DLRSD_RESULTS.md` | one file per dataset |
| `VOCABULARY_RESULTS.md` · `PROMPT_ENSEMBLE_RESULTS.md` | the wording experiments |
| **`PIXEL_ACCURACY_RESULTS.md`** | does the *picture* improve — `aAcc`, per-class recall, right/wrong maps, and the proof that lever 1 cannot raise pixel accuracy on DLRSD |
| `TTA_RESULTS.md` · `HEAD_FUSION_RESULTS.md` · `PRESENCE_POWER_RESULTS.md` · `AFFINE_RESULTS.md` | the four things that did **not** work, with their bounds |
| `BASELINE_NUMBERS.md` | the baseline's published table — check before proposing a dataset |
| `prereg/` | predictions committed **before** each run, scored afterwards |
| `paper/` · `slides/` | the write-up and the review decks; `paper/numbers.tex` is the only place a number is typed |
| `docs/` | figures, including the pixel-accuracy plots and the per-tile comparisons |
| `results/` | the machine-readable outputs behind the figures |

---

## 5. Reproducing a result

All measurement runs on the lab workstation (GPU + data); the Mac holds docs and analysis only.

```bash
# the gate: this must print 47.38 before any other number is trusted
cd ~/SegEarth-OV-3 && python eval.py ./configs/cfg_loveda.py

# fit both levers on a calibration split and write the configs + held-out list
python scripts/reorder_deploy.py --cache ~/outputs/dlrsd_full/cache --tau 0.1 \
  --objective real --stratify-re "^([a-z]+)" --calib 400 --seed 0 \
  --base-cfg cfg_dlrsd.py --cls-file cls_dlrsd.txt \
  --split-out ~/splits/dlrsd_heldout.txt \
  --cfg-out ~/SegEarth-OV-3/configs/cfg_dlrsd_perclass.py \
  --md ~/outputs/dlrsd/deploy.md
# then run the three eval.py passes the .md prints, and compare to its predictions

# pixel accuracy and the plots (CPU, from the cached scores)
python scripts/pixel_accuracy.py --preset dlrsd
python scripts/fig_pixel_accuracy.py --json results/*/pixel_accuracy.json \
  --out docs/fig_pixel_accuracy
```

**Rules the project runs on:** every number is predicted from the cache first and then confirmed
by the unmodified evaluator; predictions are committed to `prereg/` before the run and scored
afterwards, including the ones that fail; a calibration tile is never an evaluation tile.

---

## 6. Environment — do not upgrade anything

Three constraints intersect at exactly one workable point (five other combinations failed):

| | |
|---|---|
| Python | 3.11 (conda env `segov3`) |
| torch | 2.4.1+cu121 |
| mmcv | 2.2.0 (prebuilt wheel, torch2.4/cu121 index) |
| mmsegmentation | 1.2.2, `MMCV_MAX` patched to `'2.3.0'` |

Hardware: RTX 2000 Ada, 16 GB, capped at 70 W. Full rationale in `WEEK1_RESULTS.md` §2.

---

## 7. Status and what is left

✅ Baseline reproduced · five datasets · two levers · four end-to-end verifications · the
vocabulary study · four negative levers closed with bounds · a competitor (ConInfer) run and
improved · the search-range ablation.

⏳ **Open:** DLRSD is not yet written into the paper (which still describes four datasets); the
paper needs ~2,000 words moved to supplementary; pixel accuracy is measured on DLRSD and Potsdam
and not yet on LoveDA and UAVid.

**Venue:** IEEE TGRS (rolling). **Content freeze: 1 Jan 2027.**
