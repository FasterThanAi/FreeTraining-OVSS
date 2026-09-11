# Pre-registration — lever 3, per-class head fusion

**Written 11 Sep 2026, BEFORE any cache is written.** `git log` is the timestamp.
Not edited afterwards.

---

## The gap

SegEarth-OV3 has two heads because the right one is **class-dependent**: the instance decoder
is sharp on countable "things" and fragments amorphous "stuff"; the semantic head is the
reverse. We reproduced that on our own data (`ANALYSIS §4.5`): one LoveDA tile, `building`
returns **14** instance masks, `road` returns **2**.

It then fuses them with a **class-independent** rule
(`segearthov3_segmentor.py:196-210`):

```
s_c = max(P_inst_agg_c, P_sem_c)          # identical for every class
```

That is the third and last global choice the baseline makes, and the only one not yet fitted:

| their global choice | our fix | measured |
|---|---|---|
| one τ for all classes | per-class τ | +1.18 LoveDA, +2.51 ConInfer |
| implicit `w_c = 1` for all classes | per-class scale | +1.16 LoveDA, +4.92 Potsdam |
| `max(P_sem, P_inst)` for all classes | **this experiment** | — |

**The rule.** `s_c = max(a_c·P_sem_c, b_c·P_inst_c)` with `max(a_c, b_c) = 1`, so the free
parameter is the **ratio** `ρ_c = b_c/a_c`. `ρ = 1` is exactly the published rule, so rung D
nests rung C. The overall per-class scale is deliberately left to lever 2, which already fits
it — ρ carries only the part lever 2 **cannot** express.

**Rungs:** A published τ · B +per-class τ · C +per-class scale · **D +per-class fusion**.
⭐ **The result is D − C.**

---

## What is known going in

- Lever 2 already reorders the argmax and collected **+1.16** (LoveDA) and **+4.92** (Potsdam).
  Lever 3 acts on the *same* decision. **Overlap is expected.**
- §9d: `inst_fires` — how often the instance head fires for a class — has ρ **+0.657** against
  the oracle τ, the **largest** value in that column. The one proxy pointing at the target
  rather than at precision is about the instance head.
- ⚠️ Where one head already dominates a class everywhere, `max` is already picking it and ρ
  can do nothing. This is the main reason to expect a small effect.
- ⭐ Substitution between levers 1 and 2 was **refuted** today (`SUBSTITUTION_RESULTS.md`).
  H5 below tests whether it fails at the **class** level too.

---

## Predictions

**H1 — D − C will be POSITIVE but SMALL: above 0 and below +0.50**, and below lever 2's own
gain on the same dataset.
*Why:* lever 2 already owns a free per-class knob on the same argmax. ρ adds a second knob on
a strictly narrower part of the space. **I expect this lever to be the weakest of the three.**

**H2 — more classes will take ρ < 1 than ρ > 1.**
*Why:* SegEarth-OV3's stated failure mode is the instance head fragmenting stuff, and both
datasets are stuff-dominated (LoveDA: 5 stuff classes to 1 thing; Potsdam similar). Discounting
the instance head should be the common correction.

**H3 — ⭐ the countable "thing" class will have the HIGHEST ρ of any class**: `building` on
LoveDA, `car` on Potsdam.
*Why:* `ANALYSIS §4.5` directly — 14 instance masks for `building` against 2 for `road`. This
is a prediction about **which** class moves, not just a sign, so it is much harder to hit by
chance and it is the sharpest available test of SegEarth-OV3's own stated duality.

**H4 — at least 2 classes keep ρ within ±30% of 1 in every fold.**
*Why:* `max` is a sensible default and should already be right for some classes. If **no**
class keeps ρ ≈ 1, suspect the fit is wandering rather than finding.

**H5 — ⭐ Potsdam's `tree` will NOT move (ρ within ±30% of 1).**
*Why:* lever 2 already collected `tree` (+21.17, IoU 37.92 → 59.09). If class-level
substitution held, the class lever 2 already fixed has nothing left. **Substitution was
refuted at dataset level today; this tests it one level down**, and either answer is
informative.

---

## The branch table, written now

| outcome | reading |
|---|---|
| **D − C clears the gate** (`mean − 2·sd > 0` and every fold positive) | ⭐ all three of the baseline's global choices are the wrong shape. Strongest form of the paper's thesis. ⚠️ Still a cached-histogram prediction until `eval.py` reproduces it (§9c). |
| **positive but inside the gate** | report as promising-not-established, exactly as lever 2's control arms were. Do **not** write it up at that width. |
| **null** | ⛔ a clean negative, and a *useful* one: it says `max` is already the right fusion once per-class τ and per-class scaling are in place, which bounds the family and explains why. **Report it.** |
| **H3 holds but D − C is null** | ⭐ the most interesting outcome: SegEarth-OV3's things/stuff duality is real and measurable in the fitted ρ, and correcting it buys nothing because levers 1–2 already collect that mass by other means. |
| **H4 fails (no class stays at ρ ≈ 1)** | ⛔ treat the whole run as suspect and check the fit before reading any other prediction. |

⚠️ **The identity gate is checked before any of this.** The cached stacks are pre-presence, so
at ρ = 1 the reconstruction must reproduce the cached `logits` to float16 precision. If
inference ran sliding-window with overlapping crops it will not, because the pipeline fuses
*inside* the crop loop and a sum of maxima is not the maximum of sums. **A failed gate voids
the run** — it would measure a different model from the one deployed.

⚠️ **Two bugs were found and fixed by unit tests before this was run**
(`scripts/test_head_fusion.py`): τ frozen during the ρ search (capped the objective 50 points
below the reachable answer on a synthetic case), and ties breaking toward the first grid point
instead of toward the published rule (invented confident ρ = 0.25 on classes where fusion is
irrelevant). Both would have produced a clean, plausible, wrong table.
