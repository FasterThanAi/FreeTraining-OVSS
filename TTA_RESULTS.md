# Dihedral TTA — ⭐ the first intervention that improves precision AND recall

**13 Sep 2026, LoveDA val, 1669 tiles, `tta='hflip'`, τ = 0.5.** Two full-image forward passes per
tile, averaged; 1.67 s/image against the baseline's 0.85.

---

## 1. The result

| | baseline | `hflip` | |
|---|---|---|---|
| **mIoU** | 47.38 | **47.67** | ⭐ **+0.29** |
| mPrecision | 67.1 | **67.5** | ⭐ **+0.4** |
| mRecall | 62.0 | **62.1** | ⭐ **+0.1** |
| classes improved | — | **7 of 7** | ⭐ |

| class | IoU | | class | IoU |
|---|---|---|---|---|
| water | 51.44 → **51.97** (+0.53) | | barren | 35.73 → 36.06 (+0.33) |
| building | 63.81 → **64.19** (+0.38) | | background | 45.50 → 45.78 (+0.28) |
| road | 53.89 → 54.12 (+0.23) | | forest | 33.78 → 33.99 (+0.21) |
| agricultural | 47.47 → 47.59 (+0.12) | | | |

⚠️ **+0.29 is modest**, and smaller than either fitted lever. It is reported as what it is.

---

## 2. ⭐⭐ Why it matters more than its size

**Every other intervention measured in this project trades one thing for another:**

| intervention | precision | recall |
|---|---|---|
| τ → 0.1 | falls | rises — net **−5.54** |
| `--no-presence` | — | — net **−11.97** |
| sliding window | **+1.1** | **−4.7** — a tighter gate |
| loosening that gate | **−9.9** | +1.7 — precision collapses |
| per-class τ | per class, in **opposite** directions (`road` +3.7/−4.2, `water` −3.4/+9.2) | |
| ⭐ **hflip TTA** | ⭐ **+0.42** | ⭐ **+0.14** |

> ⭐⭐ **This is the only intervention in the project where both move up. That is the signature of
> genuinely better scores rather than a redistributed decision.** Everything else in this paper
> reallocates a fixed quantity of evidence; this one adds evidence.

⭐ **And it is the version of multi-view inference that sliding window was not.** Crops shrank the
field of view, every head degraded, and the per-crop presence gate was what partly rescued it
(`SLIDING_WINDOW_RESULTS.md` §2a). A flip uses the **whole image**, so field of view, scene
context and presence semantics are all identical between views — only the orientation differs. The
failure mode was removed by construction, and the sign flipped from **−3.85** to **+0.29**.

⭐ Aerial imagery has no canonical orientation, so a mirrored tile is an ordinary image rather than
an unnatural one. The argument is specific to remote sensing and does not carry to natural images.

---

## 3. Predictions scored

Stated before the run: *"hflip is the more likely of the two to pay, and I'd guess under +1.0."*

| | prediction | measured | |
|---|---|---|---|
| hflip pays at all | positive | **+0.29** | ✅ |
| magnitude | under +1.0 | +0.29 | ✅ |

⚠️ After five consecutive wrong calls this one landed, and it is the one I argued for **without** a
mechanism story — on the grounds that it is the single standard technique this pipeline skips.
**That is the lesson, not the +0.29.**

---

## 4. ⛔ What has NOT been established

1. **Whether the fitted levers survive on top.** This is the question sliding window never reached,
   because its baseline got worse. TTA gives a **better** baseline, so it is finally askable: does
   `+2.32` hold when the scores it corrects are already improved? ⭐ **That is the experiment that
   matters, and it needs a `--cache-full` run with TTA enabled, then a CPU refit.**
2. **Whether 8 views beat 2.** `d4` averages all eight dihedral transforms at 8× inference cost.
   TTA gains usually scale sub-linearly in view count, so expect more than +0.29 and well under
   4×~it.
3. **Any dataset but LoveDA.**

⚠️ **Adopting TTA would move every absolute number in the paper**, since it changes the baseline.
The cheap and honest framing is to keep the published single-view configuration as the primary
line and report TTA as an orthogonal, label-free addition that composes — *if* §4.1 confirms it
composes.
