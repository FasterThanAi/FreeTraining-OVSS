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

## 2a. ⛔ AND §2's DIAGNOSIS IS WRONG — tested 13 Sep, corrected here

§2 above concluded that the per-crop presence gate **caused** the loss: *"a class present in one
corner is vetoed in the other three."* That implies loosening the gate should recover the loss.
**It was tested directly and it does the opposite.**

| gate under sliding window | mIoU | vs per-crop | mPrecision | mRecall |
|---|---|---|---|---|
| **per crop** *(the published behaviour)* | **43.53** | — | **68.1** | 57.3 |
| max over crops — *"present anywhere in the tile"* | **39.86** | ⛔ **−3.67** | 58.2 | 59.0 |
| one whole-image forward | **39.05** | ⛔ **−4.48** | 56.7 | 57.8 |

> ⭐⭐ **Precision collapses as the gate loosens — 68.1 → 58.2 → 56.7 — while recall barely moves.
> The per-crop gate was not destroying valid predictions. It was suppressing false ones.**

`building`'s precision goes **77.0 → 49.4**, `road`'s **69.7 → 52.3**. A class that is genuinely
absent from a crop *should* be vetoed there, and telling the model it is "present somewhere in the
tile" lets it fire across the whole crop grid.

### ⭐ What the loss actually is

If the gate were the cause, removing it would help. It hurts. So sliding-window's **−3.85** is not
*resolution minus gate*: **crops lose scene context, that degrades every head, and the per-crop
gate is what partly rescues it.** The resolution benefit this run was meant to isolate is at best
small and is swamped; it cannot be separated with this design, because every crop configuration
changes context for the semantic and instance heads too, not only for presence.

⭐⭐ **And that is a second job for presence gating that the project had not identified.**
WEEK1 §9.2b established one — it suppresses `background`, whose median `S_pres` is 0.022. This
establishes another: **it suppresses classes that are outside the current field of view**, and
under sliding window that job is worth **3.67–4.48 mIoU**. It is the fourth independent measurement
showing that loosening this gate costs more than it returns (τ→0.1 −5.54; `--no-presence` −11.97;
these two).

### ⚠️ The half of §2 that survives

`water` *is* hurt by per-crop gating, exactly as §2 argued — it is spatially concentrated, so it
loses its presence score in most crops. Under the whole-image gate it recovers **37.16 → 44.33**,
**50.2%** of its loss, and it is the only class that improves. **The mechanism was right for that
class and wrong about the net**: repairing water costs more in `background`, `road` and `building`
than it returns.

⛔ **§2's closing sentence — "shrinking the view shrinks what the gate lets through" — is true and
misleading.** It lets less through because less *should* be let through.

---

## 3. Predictions scored

| | prediction | measured | |
|---|---|---|---|
| **W1** | baseline rises > +1.0 | **−3.85** | ⛔ **refuted** |
| **W2** | discard rate falls | rose (bg recall +12.9) | ⛔ **refuted** |
| W3 | levers still clear the gate, gain < +2.32 | not run — pointless on a worse baseline | — |
| **W4** | `water`'s advantage shrinks but survives | **−14.28 IoU**, the worst class | ⛔ |
| **W5** | presence becomes more informative | ⭐ **exactly backwards** | ⛔ |

⚠️ **W5's scoring is itself corrected by §2a.** Per-crop presence is *less* informative about what
is in the tile and *more* informative about what is in the **crop** — which is the quantity the
decision actually needs. Calling it "backwards" was right; calling the veto a fault was not.

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

## 5. ⚠️ The follow-up: advised against, run anyway, and it refuted my diagnosis

The diagnosis suggests a fix: compute `S_pres` **once over the whole image** and apply it to
**crop-level** fused scores — global presence, local resolution. It is one more `eval.py` run.

⛔ I advised against it: a fourth attempt at the same family after three nulls, an inference-time
change rather than a methodological one, and weeks from a content freeze. **It was run anyway, and
that was the right call** — see §2a. My objections were about cost and novelty and they stand, but
the run did something I had not argued for: it **refuted my own explanation of §2**, which was
about to go into the paper as fact. ⭐ **A wrong mechanism caught by a two-hour run is worth far
more than the run cost.**

⭐ **Scored honestly: my record on "this should work" is now 0 for 5**, and the one time I argued
*against* running something, running it was what caught my error.
