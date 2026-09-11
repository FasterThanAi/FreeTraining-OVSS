# Pre-registration — lever 4, per-class presence weight

**Written 11 Sep 2026, BEFORE the first run.** `git log` is the timestamp. Not edited afterwards.

---

## The gap, and why it is the best-motivated lever left

SegEarth-OV3 applies a **hard per-class ceiling** — `P_final_c = P_fused_c · S_pres_c` — identically
to every class. But the classes are nowhere near each other on it (WEEK1 §9.2b, median over 1669
LoveDA tiles):

| road | building | water | agricultural | barren | **forest** | *background* |
|---|---|---|---|---|---|---|
| 0.91 | 0.84 | 0.77 | 0.60 | 0.55 | **0.45** | *0.022* |

⭐ **`forest` gets half the ceiling `road` does, and `forest` is the worst class in the dataset**
(33.78 IoU, 34.6% of its pixels discarded).

⭐⭐ **And the known failure predicts this should work.** Turning gating off *globally* costs
**−11.97 mIoU** — but §9.2b measured *why*, and the reason is class-specific: `background`'s
presence is 0.022, so the gate is mostly holding **background** down. Remove it and background
surges (healthy tiles 0.46% → 54.11% discard). **A per-class weight can keep `background` gated
while un-gating `forest`, which is exactly what the global switch could not do.** A knob that fails
globally for a per-class reason is the definition of a knob that should be per class.

**The rule:** `s_c = P_fused_c · S_pres_c^γ_c`. γ = 1 is the published rule; γ = 0 turns gating off
for that class alone. ⭐ **Not absorbed by lever 2**: `w_c` is one constant per dataset, `S_pres`
varies **tile by tile**. It escapes the per-class-τ completeness bound for the same reason.

⭐ **Zero GPU.** `logits` and `spres` are already in every `--cache-full` cache, so this runs on
`loveda_full_all` (all 1669 tiles) and `potsdam_full` (all 2016) — **full splits, no sampling**,
unlike lever 3.

---

## Predictions

**M1 — E − C will be positive and will CLEAR the gate on LoveDA** (`mean − 2·sd > 0` and 5/5 folds),
landing above lever 3's +0.04 and below lever 2's +1.16.
⚠️ *Stated with lower confidence than the mechanism alone suggests: lever 3 also had a good story
and returned a null. If this is a null too, the honest conclusion is that levers 1 and 2 exhaust
what a per-class decision rule can reach, and I will say so rather than look for a lever 5.*

**M2 — ⭐ `forest` will take γ < 1** on LoveDA. The most-gated real class, and the worst-performing.
**This is the sharp one** — a named class and a direction, committed before the run.

**M3 — `road` will keep γ within ±0.15 of 1.** Its presence is already 0.91, so there is almost no
ceiling to lift.

**M4 — `background` will take γ > 1.** Under `--objective real` the catch-all's own IoU is not
scored, so the fit is free to suppress it to buy real-class recall. ⚠️ This is partly a degeneracy
of the objective, not a finding — read its row as "what the fit did to help the real classes".

**M5 — γ will correlate POSITIVELY with each class's median `S_pres`** (Spearman > 0): classes the
gate already barely touches keep γ ≈ 1; heavily gated classes want it reduced.

**M6 — at least 2 classes keep γ = 1 in every fold.** If none does, suspect the fit, not the finding.

---

## The branch table, written now

| outcome | reading |
|---|---|
| **M1 ✅ and M2 ✅** | ⭐ a third working lever, and the one with a *measured* mechanism behind it rather than an architectural argument. The presence gate is the wrong shape in the same way τ was. |
| **M1 ✅, M2 ⛔** | the gain is real but not from the class the mechanism named. Report it and **do not** narrate the forest story. |
| **M1 ⛔ (null)** | ⛔ levers 1 and 2 exhaust the per-class decision rule. Three independent knobs — fusion, presence, and the score itself — reduce to two. **That is a bound worth stating, and it is where this line of work stops.** |
| **M6 ⛔** | treat the run as suspect before reading anything else. |

---

## ⚠️ Two limitations to state with any result

1. **The synonym approximation.** The segmentor applies presence per **query**, before the synonym
   collapse; `spres` is cached per **class** (max over its queries). For a one-prompt class the two
   coincide and the fit is exact. LoveDA's `building,house`, `forest,tree` and
   `barren,bareland,soil` are **approximate**, and `forest` — the class M2 names — is one of them.
   ⛔ **No synonym class's γ may be quoted as deployed until an `eval.py` run reproduces it**, the
   same rule §9c applied to per-class τ.
2. **The identity gate is checked first.** At γ = 1 the ungate/re-gate must reproduce the cached
   `logits` exactly, or the reconstruction is not the pipeline and E is refused.

⚠️ **Two bugs were fixed by unit tests before this was run** (`scripts/test_presence_power.py`),
both inherited from lever 3's design: τ must be refitted **inside** the γ search, and ties must
break toward **γ = 1** rather than toward a grid endpoint. The test's positive case is a 3-cycle
where a per-class scale is **provably** unable to help and γ fixes it outright (+27.77), and its
negative case has presence carrying no information at all (+0.00, γ = 1 everywhere).
