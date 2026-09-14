# UAVid — baseline reproduced at 56.86, which is 2.16 ABOVE the published 54.7

**14 Sep 2026.** 70 official val frames, `cfg_uavid.py`'s own vocabulary, thresholds
(`prob_thd=0.3`, `confidence_threshold=0.3`) and 7-class `UAVidDataset` definition.

---

## 1. The numbers

| `max_side` | frame given to SAM 3 | mIoU | vs published **54.7** |
|---|---|---|---|
| 1008 | 1008x567 | **56.86** | **+2.16** |
| 1344 | 1344x756 | **57.29** | **+2.59** |

⭐ **Resolution runs the WRONG way to explain the gap.** More pixels scores *higher*
(+0.43 for 1.33x the linear size), so the full 3840x2160 frame their config implies would
land further above 54.7, not closer to it. The downscale is not inflating us -- it costs
about half a point, and `max_side=1008` is therefore a conservative operating point.

⚠️ **The gap is UNEXPLAINED.** Same code, same config, same `cls_uavid.txt`, same
thresholds, same class definition. The only free variable left is *which frames*: their
`data_prefix` reads `img_dir/test`, and UAVid's official test split has **no public
labels**, so they must have placed labelled frames under a directory named `test`.
Whether those are these 70 is not determinable from the released config.

⭐ **This does not block the work.** Everything the method claims is a **delta** measured
on our own run, which is robust to a constant offset -- the same argument that made
ConInfer usable at a −2.34 reproduction gap (@CONINFER_RESULTS.md). Report both numbers.

## 2. ✅ The labels are verified exactly, not approximately

70 label PNGs, 597,196,800 px: **0 pixels at 255 (ignore), 0 unexpected values.** Nothing
was excluded from the metric, so the mIoU is over the whole split.

⭐ **And the total is self-verifying.** 70 frames at 3840x2160 would be 580,608,000. It is
not -- the split is **40 frames at 3840x2160 plus 30 at 4096x2160**:

    8,294,400 x 40  +  8,847,360 x 30  =  597,196,800   (exact)

UAVid ships two frame widths. Every label is intact at full resolution, and the arithmetic
closing to the pixel is stronger evidence than the histogram alone.
⛔ **Consequence: `tta='d4'` is unusable here** -- it refuses non-square input by design,
and these are 16:9 and 1.896:1. `hflip` is the available TTA.

## 3. ⭐⭐ The per-class profile is the widest spread in the project

| class | GT share | IoU | P | R | **P−R** | what per-class tau should do |
|---|---|---|---|---|---|---|
| **human** | **0.19%** | 17.62 | 78.91 | 18.49 | ⭐ **+60.4** | barely fires, right when it does -> **down** |
| **tree** | 23.55% | 53.15 | 92.13 | 55.68 | **+36.5** | LoveDA `water`'s profile (+34.8) -> **down** |
| car | 1.81% | 63.28 | 86.12 | 70.46 | +15.7 | down |
| building | 33.98% | 90.79 | 95.98 | 94.37 | +1.6 | leave |
| background | 16.16% | 55.23 | 67.88 | 74.77 | −6.9 | up |
| road | 10.17% | 67.79 | 76.67 | 85.41 | −8.7 | up |
| **vegetation** | 14.15% | 50.19 | 55.20 | 84.68 | ⛔ **−29.5** | over-fires -> **up** |

*(IoU/P/R at `max_side=1008`.)*

**+60.4 to −29.5 against LoveDA's +34.8 to −12.5.** Four classes want the threshold lowered
and three raised -- the "one global tau is wrong for different classes in opposite
directions" claim stated more loudly than any dataset held so far. WEEK3 §9g measured that
the **P−R gap** ranks which classes calibration moves (rho +0.713, p 0.013) while precision
alone does not (+0.168, p 0.60), so this table is a direct prediction, not a hope.

⛔ **`human` is 0.19% of pixels and owns 1/7 = 14.3% of mIoU.** One IoU point there moves
the headline 0.14. **Never quote a UAVid mean without this table** -- it is WEEK3 §9h's
leverage argument with a rare class in the catch-all's role.

## 4. ⭐ A new interior point on §7's share axis, for free

`background` (UAVid's clutter) is **16.16%** of ground truth, landing in an empty gap:

| dataset | catch-all share |
|---|---|
| OpenEarthMap | 0.84% |
| Potsdam | 4.29% |
| ⭐ **UAVid** | ⭐ **16.16%** |
| LoveDA urban | 26.0% |
| LoveDA (pooled) | 36.1% |
| LoveDA rural | 42.9% |

WEEK3 §7c found the detectability effect **saturates by ~35%** share, with the action at the
low end and only two points (0.84%, 4.29%) below 26%. UAVid is a third.

## 5. ⛔ The blocker ahead: 70 val frames is below the calibration budget

§9b's learning curve: ~200 calibration tiles for every draw to be positive, and **below 50 it
actively hurts** (−2.14 at n=10). A 5-fold inside 70 frames leaves 56 calibrating and 14
evaluating, which is inside the harmful region, and frames drawn from the same flight
sequence are strongly correlated on top of that.

✅ **UAVid ships 600 labelled TRAIN frames** (verified: `train: 600 images, 600 labels`), so
the budget exists. ⚠️ But WEEK3 §9b's rule is **calibrate on the distribution you evaluate
on** -- LoveDA train->val gives **−0.12**. Decide the protocol before caching, not after.

⚠️ **Cache size.** These frames are ~8.5 M px against LoveDA's 1.05 M. 670 frames is 5.7 G px
against LoveDA's 1.75 G -- **3.3x the LoveDA cache** at the same per-pixel layout. Measure the
estimate before launching, per the fix in `measure_discard_rate.py`.

## 6. Engineering note worth keeping

⛔ **`predict()` ignores mmseg's `inputs` tensor and re-opens the original file from disk**
(`segearthov3_segmentor.py:404-405`, *"Load original image to preserve details for SAM3"*).
**A `Resize` in the test pipeline therefore does nothing at all to what SAM 3 sees.** Adding
one left the failing CUDA allocation at exactly 2.63 GiB with 12.9 GB resident -- an 8x
smaller input that moves no memory is the tell. `max_side` exists because that is the only
point where the frame can be shrunk; it defaults to 0 and its condition is a strict `>`, so
every recorded LoveDA and Potsdam number is untouched (`scripts/test_max_side.py`).
