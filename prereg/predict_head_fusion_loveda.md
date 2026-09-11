# Pre-registration — lever 3 on LoveDA, and the scale hypothesis

**Written 11 Sep 2026, AFTER the Potsdam result, BEFORE the LoveDA cache exists.**
`git log` is the timestamp. Not edited afterwards.

⚠️ **Stated plainly: this is a post-hoc hypothesis given a pre-registered test.** It was formed by
looking at Potsdam's fitted ρ. It earns its status only from being committed before the LoveDA
numbers exist, and it must be reported that way.

---

## What Potsdam gave

| class | ρ | |
|---|---|---|
| **car** | **2.33** | wants the instance head |
| road / tree / grass / clutter | 1.11 / 1.00 / 0.97 / 0.88 | `max` already right |
| **building** | **0.53** | wants the **semantic** head |

`D − C = +0.04 ± 0.11` — a null. But `building` going **below** 1 contradicts the reasoning behind
H3, which was *countable thing → instance head*, taken from `ANALYSIS §4.5`: one **LoveDA** tile
where `building` returns **14** instance masks and `road` returns **2**.

---

## The hypothesis

> **The head a class wants is set by its object size in PIXELS, not by whether the class is
> semantically countable.**

Potsdam is **5 cm** GSD: a building is a large contiguous roof, which is *stuff*. LoveDA is
**30 cm**: the same building is ~36× fewer pixels, small and discrete, which is *thing*. `car` is
small at either resolution and takes ρ = 2.33.

---

## Predictions

**L1 — ⭐ LoveDA's `building` will take ρ > 1**, the opposite sign to Potsdam's 0.53.
*Why:* the hypothesis, and §4.5's 14 instance masks were measured on LoveDA itself. **This is the
whole test.** A single number, a sign, committed in advance, on a dataset that has not been cached.

**L2 — `building` will have the highest ρ of any LoveDA class.**
*Why:* it is the only countable "thing" in LoveDA's vocabulary; the other five are all stuff.
Stricter than L1 and it can fail while L1 holds.

**L3 — `road`, `water`, `forest`, `agricultural` will each keep ρ within ±30% of 1, or fall below.**
*Why:* all four are amorphous stuff at 30 cm; none should want the instance head.

**L4 — D − C will be a null on LoveDA too**: mean − 2·sd ≤ 0, or fewer than 5/5 folds positive.
*Why:* levers 1 and 2 already collect this mass. LoveDA's residual is 6× Potsdam's (29.68% vs
4.68%), so there is more room — but the same two levers act on it first. ⚠️ I am predicting my own
lever fails twice.

**L5 — at least 3 of 7 classes keep ρ within ±30% of 1 in every fold.**
*Why:* Potsdam gave 4 of 6. If **no** class stays at 1, suspect the fit rather than the finding.

---

## The branch table, written now

| outcome | reading |
|---|---|
| **L1 ✅** | ⭐ object scale, not semantic category, decides which head a class wants. A crisp, transferable statement about SAM 3's two heads, confirmed across a 6× GSD difference on a sign committed in advance. Worth a paragraph even though the mIoU is a null. |
| **L1 ⛔** (`building` ≤ 1 again) | the scale story is **wrong**. `building` simply prefers the semantic head, and §4.5's 14 masks say nothing about which head wins a pixel. Report the hypothesis and its refutation; do not look for a third explanation. |
| **L4 ⛔** (LoveDA gains) | lever 3 is dataset-dependent, not closed. Then it needs a third dataset before any claim, and Potsdam's null becomes the interesting half. |
| **L5 ⛔** | treat the run as suspect and check the fit before reading L1. |

⚠️ **The identity gate is checked first, as on Potsdam.** LoveDA has **11 queries for 7 classes**
(`building,house`, `forest,tree`, `barren,bareland,soil`), so this is also the first real test of
the query→class collapse in `--cache-heads`. A failed gate voids the run.

⚠️ **The LoveDA cache will be a `--sample 800` draw**, not the full 1669 tiles, for disk. So rung A
will land near 47.4 but not on it, and the 29.68% discard figure does not apply to the subset.
