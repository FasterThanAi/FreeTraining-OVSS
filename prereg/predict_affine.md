# Pre-registration — lever 5, affine instead of multiplicative

**Written 12 Sep 2026, BEFORE the run.** `git log` is the timestamp. Not edited afterwards.

⛔ **This exists because I closed the family too early.** After lever 4's null I wrote *"the
decision side is exhausted by construction; there is no lever 5"*, reasoning that per-class τ is
provably complete after a fixed argmax and the only thing left at the argmax is a **general**
reordering, unbounded in parameters. **That skipped a bounded family sitting in the middle**, and
it is the one the calibration literature considers the natural next step.

---

## The gap

```
lever 2 (tested):   pred = argmax_c ( w_c · s_c )              N parameters
lever 5 (this):     pred = argmax_c ( w_c · s_c + b_c )        2N parameters
```

This is the standard hierarchy from probability calibration — temperature scaling → **vector
scaling** → matrix scaling (Guo et al. 2017; Kull et al. 2019). We fitted the diagonal case and
skipped the one a reviewer who knows that literature will ask about by name.

⭐ **And the bias acts precisely where the residual lives.** Where scores are high, `w·s` dominates
and `b` is irrelevant. At **low-confidence** pixels — the discarded residual, by definition — `b`
decides who wins, and lever 2 literally cannot reach there: scaling a small number leaves it small.

⚠️ **The counter-argument, stated before the run:** τ already governs the low-confidence regime
per class. `b` may simply re-parameterise what τ does, in which case refitting τ on top absorbs it
— the same fate as levers 3 and 4.

⚠️ `b` shifts the score the **argmax** reads. As with lever 2, the **threshold reads the raw
score**, so `b` is confined to the reordering and cannot masquerade as a threshold change.

---

## Predictions

**N1 — D − C will be positive but will NOT clear the gate** (`mean − 2·sd > 0` and 5/5 folds).
*Why:* it is the same decision-side family as lever 2, which already collects +1.16. The bias adds
reach in the low-confidence regime, but τ is fitted there too. ⚠️ I have now predicted "small
positive" twice (levers 3 and 4) and got a null both times, so this is stated with low confidence.

**N2 — ⭐ if it does clear the gate, the gain will be larger on LoveDA than on Potsdam.**
*Why:* LoveDA's residual is **29.25%** against Potsdam's 4.68%, so there are six times more
low-confidence pixels for `b` to act on. **This is the sharp prediction** — a direction across two
datasets, committed in advance.

**N3 — the fitted `b` will be NEGATIVE for the catch-all** and positive or near zero for real
classes.
*Why:* the catch-all wins the argmax by default in the low-confidence regime, which is the
mechanism the method spends τ correcting. A negative bias suppresses it exactly there.

**N4 — the learning curve will degrade relative to lever 2 at 200 tiles.** 2N−1 free parameters
against N−1; §9b measured ~200 tiles for 6 parameters, and this asks for 13 on LoveDA.
*Why:* it is the standard bias–variance cost, and it is the honest reason not to keep adding
parameters. If the gain is real but needs 400 tiles, the budget claim changes and must be restated.

**N5 — at least 2 classes keep `b` within one grid step of 0 in every fold.**
*Why:* if no class stays at the baseline, suspect the fit rather than the finding — the same guard
that caught a grid-tolerance bug in lever 4.

---

## The branch table, written now

| outcome | reading |
|---|---|
| **clears the gate** | ⭐ a third working lever, and my "exhausted by construction" claim was wrong in a way worth reporting. **The decision-side family is larger than the completeness argument suggested**, and the paper should say so plainly rather than quietly adding a row. |
| **positive, inside the gate** | report as promising-not-established, and **do not** add it to the method. |
| **null** | ⛔ three evidence-side and one decision-side extension all fail. The decision side is then exhausted **empirically as well as structurally**, which is a stronger closing statement than the one I over-claimed. |
| **N5 fails** | treat the run as suspect and check the fit before reading anything else. |

⚠️ **Whatever happens, the over-claim gets corrected in `CLAUDE.md`.** Writing "there is no lever 5"
and then running one without saying so would be the worst available outcome.
