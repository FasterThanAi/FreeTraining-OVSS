# SegEarth-OV3's published numbers — Table 2, arXiv:2512.08730

⛔ **Recorded 14 Sep 2026 because their absence cost four wasted exchanges.** Datasets were
proposed and rejected on guesses about whether a baseline number existed and whether the sensor
would work, when the table settles both. **Check here first before proposing any dataset.**

## Training-free row, SegEarth-OV3

| dataset | published mIoU | our reproduction | status |
|---|---|---|---|
| OpenEarthMap | **42.9** | 44.16 *(384 of 500 tiles)* | ⚠️ subset, sanity anchor only |
| **LoveDA** | **47.4** | **47.38** | ✅ Δ 0.02 — the primary gate |
| iSAID | 27.6 | — | ⛔ excluded: 97.11% catch-all, confounded |
| **Potsdam** | **57.8** | **57.83** | ✅ Δ 0.03 |
| ⭐ **Vaihingen** | ⭐ **60.8** | — | ⭐ **their 2nd-best RS result** |
| ⭐ **UAVid** | **54.7** | ⭐ **56.86** | ⚠️ **+2.16 ABOVE** — unexplained, @UAVID_RESULTS.md |
| UDD5 | **71.7** | — | config ✅ vocab ✅ |
| VDD | **64.5** | — | config ✅ vocab ✅ |
| *average* | *53.4* | | |
| ⛔ **DLRSD** | ⛔ **no row — they do not report it** | ⭐ **37.89** | see below |

## ⭐ DLRSD has NO SegEarth-OV3 number, and we ran it anyway

⛔ Their table has eight datasets and DLRSD is not one, nor does the repo ship
`cfg_dlrsd.py` or `cls_dlrsd.txt`. **So DLRSD is the one dataset in this project with
no reproduction gate**, and @DLRSD_RESULTS.md says so in its own §2 rather than
waiting for a reviewer to notice.

⭐ **The substitute anchor comes from OVRSISBench** (arXiv:2604.15652), which does
report DLRSD for training-free methods:

| method | DLRSD mIoU |
|---|---|
| ClearCLIP | 14.80 |
| SCLIP | 20.17 |
| SegEarth-OV | 23.76 |
| ProxyCLIP | 24.03 |
| Trident | 26.31 |
| ⭐ **ours (SAM 3 baseline)** | **37.89** |
| ⭐ **ours (both levers)** | ⭐ **44.42** |

⚠️ **Not a gate** — CLIP at 384² against SAM 3 at 1008², their taxonomy unstated. ⭐ But
**+14.13 over SegEarth-OV is the same order as SAM 3's uplift on LoveDA**, so a broken
preparation would have landed far outside. ⚠️ Most of that margin is the backbone, not
our contribution — the discipline @CONINFER_RESULTS.md applies to its 11.54.

Their `Oracle` row, for reference: LoveDA 50.0, Potsdam 74.3, Vaihingen 61.2, OEM 64.4.
⚠️ Not the same object as our per-class-τ oracle — theirs bounds the masks, ours bounds the
threshold vector. **Do not conflate them.**

## ⛔ Two errors this table corrects

1. ⛔ **"Drop Vaihingen, IRRG will break SAM 3."** Wrong. Vaihingen is IRRG — near-infrared
   replaces blue, vegetation reads scarlet — and SegEarth-OV3 still scores **60.8** there, above
   Potsdam and well above LoveDA. The argument was plausible and the published number refutes it.
   ⭐ **Vaihingen is a strong fourth-dataset candidate**, and ISPRS registration is already done
   for Potsdam.
2. ⚠️ **`measure_discard_rate.py` prints "no published reference for this config" on Potsdam.**
   One exists (57.8). The boilerplate only knows LoveDA's. Harmless, but it reads as an absence of
   evidence when it is an absence of a lookup table.

## ⛔ Datasets with a config but NO shipped vocabulary — unusable

`cfg_gid.py`, `cfg_gf7-building.py`, `cfg_gf-road.py` reference `classname_path` files that are
absent from the repo **and from GitHub** (`cls_gid.txt` → 404). ⭐ Given the vocabulary is worth
**+4.94 mIoU** on LoveDA (`PROMPT_ENSEMBLE_RESULTS.md`), a config without one is not runnable and
its words cannot be safely guessed.

⚠️ GID is the loss that stings: published **42.2** for SegEarth-OV3 against **46.3** for
SegEarth-OV — **the only dataset where SAM 3 loses to CLIP**, at 4 m resolution, and therefore the
most interesting place to ask whether calibration pays more where the baseline is weak.
