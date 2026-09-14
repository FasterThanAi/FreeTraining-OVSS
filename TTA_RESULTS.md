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

## 2a. ⭐ Eight views: +0.47, and the scaling is sub-linear

| | baseline | `hflip` (2 views) | `d4` (8 views) |
|---|---|---|---|
| **mIoU** | 47.38 | **47.67** (+0.29) | **47.84** (⭐ **+0.47**) |
| mPrecision | 67.1 | 67.5 | **67.7** |
| mRecall | 62.0 | 62.1 | **62.2** |
| classes improved | — | ⭐ **7 / 7** | 5 / 7 |
| s per image | 0.85 | 1.67 | **6.59** (7.8×) |

⭐ **Four times the views buys 1.6× the gain.** Both precision and recall still rise, so the
mechanism holds — but the return per view is falling sharply and there is no reason to go further.

⚠️ **And `d4` is not uniformly better, where `hflip` was.** `background` **−0.70** and
`agricultural` **−0.21** while `road` **+1.17** and `barren` **+1.12**. Rotations help
orientation-sensitive structure (roads, field edges) and hurt the amorphous classes; a mirror
alone helps everything a little. **Report `hflip` as the clean result and `d4` as the ceiling.**

---

## 2b. ⭐⭐ THE FINDING: TTA and lever 2 are substitutes, not complements

Paired fold-by-fold on the **same 800 tiles, same seed, same fold partition** — the only
difference is TTA. `~/outputs/loveda_heads` (no TTA) against `~/outputs/loveda_tta`.

| rung | TTA's gain | folds positive |
|---|---|---|
| **A** baseline | **+0.42 ± 0.32** | ⭐ **5/5** |
| **B** + per-class τ | **+0.58 ± 0.35** | ⭐ **5/5** |
| **C** + per-class scale | **+0.06 ± 0.49** | ⛔ **2/5** |

From the other direction, lever 2's own gain: **+1.26 without TTA → +0.74 with it**, a paired
drop of **−0.52**, shrinking in **4 of 5 folds**.

> ⭐⭐ **TTA survives per-class thresholds and is absorbed by per-class scaling.**

**The reason is mechanical.** Both change *which class wins the argmax*: TTA by averaging two
score vectors, the scale by reweighting one. Lever 1 thresholds the winner **after** the argmax,
so it does not compete — hence +0.58 at rung B and +0.06 at rung C.

⭐ **And the scale is the better deal by a wide margin:**

| | gain | cost |
|---|---|---|
| TTA (`hflip`) | +0.42 | **2× inference, permanently** |
| ⭐ **per-class scale** | **+1.26** | ~200 labelled tiles once, **free at inference** |

**Three times the gain at no runtime cost.** Given the scale, TTA is worth **+0.06**.

### ⭐ What this adds to the five-lever picture

It refines it rather than contradicting it. Levers 3 and 4 **reallocated** evidence per class and
were null; TTA **adds** evidence — a real second measurement — and that does work. But:

> **Adding evidence and fixing the decision are substitutes. Both end up moving the same argmax
> decisions, so whichever is applied second finds the work already done.**

This is the first *measured* overlap between two interventions in the project, and it says
something about why the calibration works: part of what it corrects is score noise that a better
measurement would also remove — far more cheaply, and with no extra inference.

⚠️ **Two caveats.** The TTA arm's search gate is marginal (0.175 against a 0.15 bar; the no-TTA
arm passed at 0.065), and a noisier search finds a slightly worse `w` — **which biases against
TTA's lever 2 specifically, so −0.52 is an upper bound on the overlap.** Both arms also flag
unstable fitted scales (15.9% / 19.4%) at 800 tiles.

⛔ **Not running the `d4` cache-and-refit.** Measured absorption is 14% (+0.42 → +0.06), so `d4`'s
+0.47 projects to roughly **+0.07** past rung C — for 8× inference forever, and 3.5 GPU hours to
establish it.

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
