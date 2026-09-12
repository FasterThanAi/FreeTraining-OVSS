# Pre-registration — sliding-window inference, the configuration never run

**Written 13 Sep 2026, BEFORE the run.** `git log` is the timestamp. Not edited afterwards.

---

## The gap, and it is measured rather than argued

`slide_crop` defaults to **0** in the segmentor and **no config in this project sets it**, so
`predict()` has always taken the whole-image branch and `slide_inference()` has never executed.
This is not inferred from the code — instrumentation reports **`views per tile: [1]`** across
800 LoveDA and 2016 Potsdam tiles.

⭐ **Why it could matter.** SAM 3 resizes every input to **1008×1008**. A 1024² LoveDA tile is
therefore seen at roughly 1:1 and never at higher effective resolution. Four 512² crops of the
same tile are each **upsampled ~2×**, so the model sees twice the detail, four times over.

Resolution is *the* known lever in this lineage: SegEarth-OV's entire CVPR 2025 contribution was a
feature upsampler worth **+5.8 mIoU**, and SCLIP / ClearCLIP / ProxyCLIP all run sliding-window
inference as standard. SegEarth-OV3 dropped it. Nobody has measured what that costs.

⚠️ **This is not a contribution.** Multi-scale inference is standard practice and would be
reported as an inference-time observation about the baseline, never as a method.

---

## ⭐ What this CANNOT do: invalidate anything already recorded

Stated explicitly because it is the first question asked of it.

1. **It is a new config and a new output directory.** No existing cache, number, figure or fitted
   vector is touched or recomputed.
2. **The reproduction gate stands.** We reproduce SegEarth-OV3 at **47.38** against their published
   **47.4** in *their* published configuration, which is single-view. That comparison is unchanged
   whatever this run shows.
3. **The method's gain is a comparison within one configuration.** `+2.32` is measured
   single-view against single-view on identical tiles. It remains valid as stated.
4. ⛔ **A different baseline would require a different comparison, not a corrected one.** If
   sliding-window changes the baseline, the levers must be re-fitted and re-evaluated *there*, and
   the two configurations reported side by side. Neither replaces the other.

⚠️ **The real risk is to the STORY, not to the numbers**, and it is worth taking: if the levers pay
less on a stronger baseline, that is true now and will be true when a reviewer runs it. Finding it
ourselves, with the single-view result intact beside it, is strictly better than being told.

---

## Predictions

**W1 — the baseline mIoU will RISE, by more than +1.0.**
*Why:* effective resolution doubles and this dataset is full of small objects. `building` and
`road` — the classes SAM 3 already handles best, and whose boundaries are thin — should gain most.

**W2 — ⭐ the discard rate will FALL.** More resolution means more confident scores, and the
residual is defined by scores below a threshold. **This is the sharp one**, because it is the only
prediction that connects to the paper's own mechanism rather than to general practice.

**W3 — the two levers will still clear the gate on the new baseline**, and their combined gain
will be **smaller** than +2.32.
*Why:* calibration corrects an ill-shaped decision over given scores. Better scores leave less for
it to correct — but per-class asymmetry is a property of the *classes*, not of the resolution, so
it should not vanish. ⚠️ **If the gain vanishes entirely, the paper's framing needs rewriting, and
I would rather know that now.**

**W4 — `water`'s advantage will shrink but survive.** It carries +6.78 of the LoveDA gain on a
precision/recall asymmetry of 89.5/54.7. Resolution should raise recall and narrow the gap.

**W5 — presence scores will become more informative, not less.** `S_pres` is one number per class
per view; with four views per tile a class present in one corner is no longer averaged against
three empty ones. ⚠️ This also means `spres` gains rows, and every script reading it must be
re-checked — `discard_after.py` and `presence_power.py` both assume a single view in places.

---

## The branch table, written now

| outcome | reading |
|---|---|
| **W1 ✅ and W3 ✅** | ⭐ the baseline was leaving multiple points on the table at inference time, **and the correction survives on top of it**. Report both configurations; the method claim strengthens. |
| **W1 ✅, W3 ⛔** | ⛔ the levers were partly compensating for an under-resolved baseline. That is a material finding and the paper must say so. **Do not bury it.** |
| **W1 ⛔** (no gain, or a loss) | sliding-window costs more than it returns here — plausible, since crops lose global context and `S_pres` is computed per view. A one-run negative, reported in two sentences. |
| **W2 ⛔ while W1 ✅** | mIoU rises without the residual shrinking, which would mean resolution fixes boundaries rather than confidence — a different mechanism from the paper's, and worth a sentence. |

⛔ **The single-view numbers are not deprecated by any outcome.** They are the reproduction of a
published configuration and remain the paper's primary line.
