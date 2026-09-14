# Pre-registration — lever 2 (per-class argmax scaling) on UAVid

**Committed 14 Sep 2026, before `~/outputs/uavid_train_full/cache` exists.**
`git log` is the timestamp. Nothing here is edited after the run; the scoring goes in
`UAVID_RESULTS.md`.

## What is being run

`pred = argmax_c(w_c · s_c)`, then threshold the **raw** score with lever 1's per-class τ,
fitting `(w, τ)` together. 200 real UAVid train frames, 20 flights, **group-disjoint folds**
(`--group-re`), augmented copies excluded (`--exclude-re`), τ published = 0.3.

Rung A = published τ · rung B = + per-class τ (lever 1) · rung C = + per-class scale (lever 2).

## Why UAVid might be different from the three datasets already measured

Lever 2 acts on classes that **lose the argmax** — a different error mass from lever 1, which
acts on classes that **fall below τ**. UAVid has an unusually clean instance of that:

| class | precision | recall | reading |
|---|---|---|---|
| **vegetation** | **53.6** | **87.8** | fires far too readily — wins argmaxes it should lose |
| **tree** | **91.8** | **55.9** | fires far too rarely — loses argmaxes it should win |
| building | 95.1 | 96.0 | balanced; nothing to fix |

`tree` and `vegetation` are the two green classes and compete directly for the same pixels.
Their precision/recall profiles are near-mirror images. **That is exactly the shape a
multiplicative reweighting before the argmax can repair, and no threshold can** — lowering
`tree`'s τ cannot take a pixel that `vegetation` already won.

## Predictions

**U1 — the headline.** Rung C − rung B clears the project gate (`mean − 2·sd > 0` **and** 5/5
folds positive). ⚠️ **Moderate confidence only.** Levers 3, 4 and 5 were each predicted as small
positives and each came back null; the honest prior is that I over-predict this family. Precedent
is mixed: LoveDA **+1.16**, Potsdam **+4.92**, ConInfer **−0.10**.

**U2 ⭐ — the sharp one.** `vegetation` is fitted at **w < 1** and `tree` at **w > 1**.
*Reason:* they are the confusable pair and their asymmetry runs in opposite directions. This is
the prediction that makes the run worth pre-registering — it names two classes and two signs
from the precision/recall table alone, before any fit.

**U3.** `human` is fitted at **w > 1**. *Reason:* it recovers only 18.5% of its pixels at 78.9%
precision, the largest precision−recall gap in the project (+60.4).

**U4.** `building` stays within 25% of w = 1 in every fold. *Reason:* 95.1/96.0 is balanced;
there is nothing for a reweighting to buy.

**U5.** Lever 1's own gain (rung B − rung A) stays within **±0.4** of the recorded **+1.34**.
*Reason:* `ARGMAX_SCALING_RESULTS` retired the "levers substitute" reading — on LoveDA's two
domains the scale gain is invariant to what the threshold collected (+1.22 / +1.25 against τ
gains spanning 35×). If lever 1 collapses here instead, that reading needs re-opening.

## Branch table — what each outcome means, written before the number

| outcome | reading |
|---|---|
| **U1 and U2 both hold** | lever 2 works on a fourth dataset, **and** the mechanism is identified: it repairs argmax competition between confusable classes. The strongest available result. |
| **U1 holds, U2 fails** | the lever works and my mechanism story is wrong. Report the fitted `w` and offer no replacement explanation — the same discipline applied when object scale was refuted for lever 3. |
| **U1 fails, U2 holds** | the fit finds the right thing to do and there is too little of it to matter — lever 3's outcome exactly (`road` had the room and one class in seven cannot move a mean). Then report which classes the reweighting reaches and how few pixels that is. |
| **both fail** | lever 2 is a null on UAVid. ⚠️ Then note that UAVid's `tree`/`vegetation` case is the most favourable setup the project has found for it, which makes the null informative rather than merely negative. |

⛔ **Scoring rule.** If **U4 fails** (a balanced class drifting far from w = 1), treat the run as
suspect and check the fit — gauge drift is a known failure mode here: lever 5's `w` drifted to a
uniform 0.40 by tie-break and produced a false null until `normalise(w)` was added.
