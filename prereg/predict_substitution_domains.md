# Pre-registration — do the two levers SUBSTITUTE? LoveDA rural vs urban

**Written 11 Sep 2026, BEFORE the run.** `git log` is the timestamp. Not edited afterwards.

---

## The claim under test

`ARGMAX_SCALING_RESULTS.md` closes with a post-hoc observation over three datasets:

| dataset | published τ | τ gain (B − A) | **scale gain (C − B)** | total |
|---|---|---|---|---|
| Potsdam | 0.1 | +0.75 | **+4.92** | +5.67 |
| LoveDA (SAM 3) | 0.5 | +1.16 | **+1.16** | +2.32 |
| ConInfer (CLIP) | 0.8 | **+2.51** | **−0.10** | +2.41 |

> *Where the threshold collects a lot, the scale collects nothing; where the threshold
> collects little, the scale collects a great deal.*

The document flags it correctly and gates it: **post-hoc, not pre-registered, and the
mechanism that WAS pre-registered (argmax-lost share) was falsified by these same three
points.** It may not enter the paper as more than a description of three datasets until it
is tested on a fourth.

⚠️ **This run is NOT that fourth dataset.** Rural and urban are strata of a dataset already
in the table, so they cannot be independent confirmation — the same limitation §7a carries.
What they *can* do is test the claim's **direction** on the widest τ-gain contrast available
anywhere in this project, at zero GPU cost, off a cache that already exists. A directional
failure here would retire the claim outright; a directional success raises it from "three
points" to "three points and a within-dataset dissociation", and a fourth dataset is still
required afterwards.

---

## What is known going in

From `WEEK3_RESULTS.md` §9e — 5-fold **within** each domain, per-class τ only:

| | tiles | τ gain (B − A) | folds positive |
|---|---|---|---|
| **rural** | 992 | **+2.77 ± 0.92** | 5/5 |
| **urban** | 677 | **+0.10 ± 0.39** | 2/5 |

A **27× spread in the τ gain**, wider than the three-dataset table's own 3.3× spread
(+0.75 → +2.51). If substitution is real anywhere, it is visible here.

⚠️ Note the totals in the three-dataset table are **not** constant (+5.67, +2.32, +2.41), so
substitution is explicitly **not** "the two levers split a fixed pool". The prediction below
is about **ordering only**, and predicting the totals is not attempted.

---

## Predictions

**S1 — the scale gain will be LARGER on urban than on rural.**
*Why:* substitution. Urban's threshold collects +0.10, i.e. nothing, so if the two levers
substitute there is a great deal left for the scale; rural's threshold already collects
+2.77. **This is the whole prediction — everything else is secondary.**

**S2 — urban's scale gain will clear +1.00**, the size at which it is not noise.
*Why:* the substitution ordering is only interesting if the "leftover" is actually
collectable. If urban lands at +0.2 the ordering may hold while meaning nothing.

**S3 — rural's scale gain will be below its own τ gain of +2.77.**
*Why:* the direct reading of substitution on the stratum where the threshold already won.

**S4 — both strata will exceed their size-matched random control's spread.**
*Why:* urban is 677 tiles and OpenEarthMap's 384 could not resolve a 1 mIoU effect
(baseline swinging 10 points across folds). 677 is between OEM and LoveDA's 1669, so it is
**not safe to assume** it has the power. The control is run at n = 677 and n = 992 and its
fold spread is the floor any claimed effect must clear.

**S5 — the fitted scale vectors will DIFFER between the domains**, mean |Δw| well above zero.
*Why:* §9e measured mean |Δτ| = 0.227 with `road` at 0.725 rural vs 0.225 urban, and §4.4
found 10 of 15 class-adjacency pairs flip sign between the domains. If `w` transferred where
`τ` does not, that would itself be a finding, and a useful one.

---

## The branch table, written now

| outcome | reading |
|---|---|
| **S1 ✅ and S2 ✅** | substitution survives its sharpest available directional test. Report as three datasets **plus** a within-dataset dissociation, still gated on a fourth dataset before it becomes a claim about pipelines. |
| **S1 ✅, S2 ⛔** | the ordering holds but the leftover is not collectable. Substitution is then a statement about *what the levers cannot do*, not a route to more mIoU. **Report and stop** — do not fit a third lever expecting to find the leftover. |
| **S1 ⛔** | ⛔ **substitution is retired.** It becomes a description of three datasets that does not survive the first test with a committed prediction — exactly what happened to the argmax-lost mechanism it replaced. Delete it from `ARGMAX_SCALING_RESULTS.md`'s closing section rather than weakening it. |
| **S4 ⛔ on urban** | the run is underpowered and answers nothing; report that and do not read S1 either way. This is OpenEarthMap's outcome and it must not be quietly forgotten a second time. |

⚠️ **S4 is checked FIRST.** If the control's spread swallows the effect, no other prediction
is scored — a table read off an underpowered run is how the OEM rows got written twice.
