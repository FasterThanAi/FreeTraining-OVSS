# Lever 5 — an affine reordering. ⛔ Null, and the decision side is now closed by measurement

**12 Sep 2026, LoveDA, full 1669 tiles, 5-fold held out, τ = 0.5.** Predictions committed in
`prereg/predict_affine.md` (`5fb225b`) before the run.

---

## 1. The result

| | mean | sd | folds+ | mean−2sd | recorded |
|---|---|---|---|---|---|
| lever 1 — per-class τ | **+1.16** | 0.44 | 5/5 | +0.29 | +1.18 ± 0.45 ✅ |
| lever 2 — per-class scale | **+1.22** | 0.65 | 5/5 | −0.09 | +1.16 ± 0.19 ✅ |
| **lever 5 — per-class bias** | **−0.07** | 0.18 | 2/5 | −0.43 | ⛔ **null** |

✅ Both existing levers reproduce their recorded values, so the run is sound.

⚠️ **I predicted "positive but inside the gate" and got a negative — the third time in three
levers.** Levers 3, 4 and 5 were each argued for on a mechanism, each predicted to be a small
positive, and each came back null or negative. **The pattern is worth stating in the paper: the
decision-side family looked larger from the inside than it is.**

---

## 2. ⭐ N3 held exactly, and it explains the null

**Prediction N3, written before the run:** *the fitted `b` will be NEGATIVE for the catch-all and
positive or near zero for real classes.*

| background | agricultural | road | water | barren | forest | building |
|---|---|---|---|---|---|---|
| ⭐ **−0.12** | +0.00 | +0.01 | +0.03 | +0.05 | +0.06 | +0.08 |

✅ **`background` is the only negative bias in the table.** And since only *differences* matter to
an argmax, "every real class about +0.05, background −0.12" is a single statement:

> **suppress the catch-all where scores are small.**

⭐⭐ **Which is why it buys nothing, and the number is already in the paper.** `b` acts on the
**argmax**. WEEK1 §7.7 measured that background assignments split **94.0% the τ rule / 6.0% argmax
wins** — so a bias that suppresses background's argmax addresses **6% of the mechanism**, and
lever 2 already reorders the argmax over the same 6%. On LoveDA `S_pres(background) ≈ 0.022` caps
background's score, so it rarely wins an argmax there is anything to win.

**The fit found the right thing to do and there was almost nothing to do it to.**

---

## 3. Predictions scored

| | prediction | measured | |
|---|---|---|---|
| **N1** | positive, not clearing the gate | **−0.07**, 2/5 | ⚠️ **half** — "does not clear" ✅, "positive" ⛔ |
| N2 | if it clears, LoveDA > Potsdam | did not clear | — void |
| **N3** ⭐ | `b` negative for the catch-all, ≥ 0 for real classes | **exactly** | ✅ **decisive** |
| N4 | learning curve degrades at 2N−1 params | not measured — pointless on a null | — |
| **N5** | ≥ 2 classes keep `b` = 0 in every fold | **1 of 7** | ⛔ |

⚠️ **N5 failed, so the fit was checked before N3 was read**, per the branch table: unit tests pass
including the gauge fix, levers 1 and 2 reproduce, and `barren` (+0.05 ×5), `forest` (+0.05 ×4) and
`agricultural` (0.00 ×5) are stable across folds. The suspicion is discharged; 6 of 7 classes
wanting a non-zero bias is a finding, and its net effect is still zero.

---

## 4. ⭐⭐ What this closes

`prereg/predict_affine.md` named this branch in advance:

> **null** — three evidence-side and one decision-side extension all fail. The decision side is
> then exhausted **empirically as well as structurally**, which is a stronger closing statement
> than the one I over-claimed.

| lever | changes | acts | result |
|---|---|---|---|
| **1** per-class τ | which label a score earns | **after** the argmax | ✅ **+1.18** |
| **2** per-class scale | which class wins | **at** the argmax | ✅ **+1.16** |
| 3 head fusion | the score itself | before the argmax | ⛔ +0.04 / +0.22 |
| 4 presence weight | the score itself | before the argmax | ⛔ +0.16 |
| **5** per-class bias | which class wins at LOW scores | **at** the argmax | ⛔ **−0.07** |

⛔ **The over-claim is corrected.** After lever 4 I wrote *"the decision side is exhausted by
construction; there is no lever 5"*, which skipped vector scaling — a bounded family the
calibration literature puts next. **It was worth running and it was worth being wrong about**: the
family is now closed by a measurement rather than by an assertion, and lever 5 is the row a
reviewer who knows Guo et al. would have asked for.

⚠️ **LoveDA only.** Potsdam's `--cache-full` exists and the run is CPU-only. Given three nulls in a
row, a second dataset is only worth it if a reviewer questions this one.
