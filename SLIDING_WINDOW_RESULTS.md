# Sliding-window inference — ⛔ costs 3.85 mIoU, and the reason is the presence gate

**13 Sep 2026, LoveDA val, 1669 tiles, `slide_crop=512`, `slide_stride=341`, τ = 0.5.**
Predictions committed in `prereg/predict_sliding_window.md` (`f5891d1`) before the run.

---

## 1. The result

| | single-view *(published config)* | sliding-window | |
|---|---|---|---|
| **mIoU** | **47.38** | **43.53** | ⛔ **−3.85** |
| catch-all-excluded | 47.69 | 42.93 | −4.76 |
| **mPrecision** | 67.1 | **68.1** | ⭐ **+1.1** |
| **mRecall** | 62.0 | **57.3** | ⭐ **−4.7** |

⛔ **W1 is refuted.** I predicted the baseline would rise by more than +1.0 because SAM 3 resizes
input to 1008² and 512² crops are therefore upsampled ~2×, and because resolution is the known
lever in this lineage. It falls.

---

## 2. ⭐⭐ Why — and it is the paper's own mechanism, a third time

**Precision rises and recall collapses.** That is not what more resolution looks like; sharper
features raise both. It is what a **tighter gate** looks like.

| class | IoU | precision | recall |
|---|---|---|---|
| ⭐ **water** | 51.44 → **37.16** | 89.5 → **90.9** | 54.7 → **38.6** |
| agricultural | 47.47 → 43.80 | 66.9 → **82.0** | 62.0 → **48.5** |
| road | 53.89 → 50.07 | 69.5 → 69.7 | 70.5 → 64.0 |
| barren | 35.73 → 31.02 | 51.5 → 48.8 | 53.9 → 46.0 |
| building | 63.81 → 61.72 | 77.2 → 77.0 | 78.6 → 75.7 |
| forest | 33.78 → 33.80 | 57.9 → 56.2 | 44.8 → 45.9 |
| *background* | *45.50 → 47.15* | *56.9 → 52.5* | ⭐ *69.4 → **82.3*** |

⭐ **`background`'s recall jumps 12.9 points.** The catch-all absorbs far more than before — the
**discard rate rose**, which refutes W2 as well.

> ⭐⭐ **`S_pres` is computed PER VIEW.** A 512² crop of a 1024² tile may contain no water at all,
> so `S_pres(water)` for that crop is near zero and, because `P_final = P_fused · S_pres`, every
> water pixel in that crop is crushed to the floor. Under whole-image inference the presence score
> is computed once over the whole tile, so a class present anywhere survives everywhere.
> **Shrinking the view shrinks what the gate lets through.**

**Water is the clean case**: it is spatially concentrated — rivers and lakes occupy part of a tile,
not all of it — so it is exactly the class that loses its presence score in three crops out of
four. It drops **14.28 IoU**, more than half the total loss.

⛔ **W5 was not merely wrong, it was backwards.** I predicted presence scores would become *more*
informative, since a class in one corner would no longer be averaged against three empty ones. The
opposite is the mechanism: a class in one corner is now **vetoed** in the other three.

---

## 3. Predictions scored

| | prediction | measured | |
|---|---|---|---|
| **W1** | baseline rises > +1.0 | **−3.85** | ⛔ **refuted** |
| **W2** | discard rate falls | rose (bg recall +12.9) | ⛔ **refuted** |
| W3 | levers still clear the gate, gain < +2.32 | not run — pointless on a worse baseline | — |
| **W4** | `water`'s advantage shrinks but survives | **−14.28 IoU**, the worst class | ⛔ |
| **W5** | presence becomes more informative | ⭐ **exactly backwards** | ⛔ **and that is the finding** |

⭐ **The branch table named the mechanism in advance.** Written before the run:

> **W1 ⛔** *(no gain, or a loss)* — *sliding-window costs more than it returns here — plausible,
> since crops lose global context and `S_pres` is computed per view. A one-run negative, reported
> in two sentences.*

---

## 4. ⭐ What it is worth

1. **It forecloses a reviewer question with a measurement.** *"Why not multi-scale inference, as
   the rest of this literature does?"* now has a number instead of a shrug.
2. ⭐⭐ **It is a third independent demonstration that presence gating is a hard per-view ceiling**
   — and the most nearly causal of the three. §9.2b turned the gate **off** and cost 11.97;
   §9.2b's per-class medians showed background at 0.022; this **shrinks the view** and watches
   recall collapse across every spatially concentrated class. Same mechanism, three interventions,
   three directions.
3. ⭐ **SegEarth-OV3's whole-image choice is correct and non-obvious**, and we can now say why
   rather than inheriting it. That is a fair thing to report about a baseline.

✅ **And nothing recorded is affected**, exactly as the pre-registration promised: a separate
config, a separate output, no cache touched. The **47.38** reproduction gate stands because it
reproduces their *published* configuration, and **+2.32** remains a within-configuration
comparison.

⚠️ **One run, one crop size, one dataset.** A larger crop (768) would lose less presence context
and might land differently; we do not claim a curve.

---

## 5. ⚠️ The obvious follow-up, and why I am not recommending it

The diagnosis suggests a fix: compute `S_pres` **once over the whole image** and apply it to
**crop-level** fused scores — global presence, local resolution. It is one more `eval.py` run.

⛔ **I am not recommending it, for three reasons.** It would be a fourth attempt at the same
family after three nulls and my record on "this should work" is now **0 for 4**. It is an
inference-time change, not a methodological one, so it strengthens the baseline rather than the
contribution. And it would move every absolute number in the paper five weeks from a content
freeze. **Record it as the natural next question and leave it to the reviewer or to future work.**
