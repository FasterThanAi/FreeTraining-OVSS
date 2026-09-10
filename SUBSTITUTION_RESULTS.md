# Do the two levers substitute? ⛔ No — they are INDEPENDENT, and that repairs urban

**11 Sep 2026.** LoveDA rural/urban strata + two size-matched random controls, off the
existing 1669-tile `--cache-full`. **No GPU.** Predictions committed in
`prereg/predict_substitution_domains.md` (`1b5ad24`) before the run.

---

## 1. The four arms

| arm | tiles | τ gain (B−A) | **scale gain (C−B)** | mean−2sd | folds+ | project gate |
|---|---|---|---|---|---|---|
| **rural** | 992 | **+2.77 ± 0.92** | **+1.22 ± 0.53** | +0.16 | 5/5 | ✅ **PASS** |
| **urban** | 677 | **+0.08 ± 0.22** | **+1.25 ± 0.56** | +0.13 | 5/5 | ✅ **PASS** |
| control | 677 | +1.02 ± 1.11 | +0.66 ± 1.21 | −1.77 | 3/5 | ⛔ fail |
| control | 992 | +1.11 ± 0.59 | +0.89 ± 0.93 | −0.96 | 5/5 | ⛔ fail |

The gate is §9e's: `mean − 2·sd > 0` **and** every fold positive.

✅ **A free reproduction.** Rural's τ gain lands at **+2.77 ± 0.92**, identical to §9e's
+2.77 ± 0.92, and urban's +0.08 ± 0.22 against §9e's +0.10 ± 0.39 — through a different
script, a different cache and a different fold partition. §9e is independently confirmed.

---

## 2. ⛔ S1 fails: substitution predicted a difference, and there is none

| | τ gain | **scale gain** |
|---|---|---|
| rural | **+2.77** | +1.22 |
| urban | **+0.08** | +1.25 |
| **ratio** | **35×** | **1.02×** |

**The threshold gain varies 35× between the strata. The scale gain does not move at all** —
+0.03, against a standard error on that difference of **0.35**.

⭐ **And the test had the power to see substitution if it existed.** The three-dataset table
substitution was read off spans **5 points** of scale gain (+4.92 Potsdam to −0.10 ConInfer).
An effect of that size against se 0.35 is a 14σ separation. **This is a refutation from
power, not a null from weakness** — the single most important sentence in this file.

⛔ **Substitution is retired**, exactly as the branch table said it must be. It goes the way
of the argmax-lost mechanism it replaced: a description of three points that did not survive
its first committed prediction. **Delete it from `ARGMAX_SCALING_RESULTS.md`'s closing
section rather than weakening it.**

⚠️ Nominally urban > rural by 0.03, so S1's *letter* is satisfied. That is not a pass and is
not reported as one. Substitution predicted a **difference**; the measurement shows **none**.

### ⭐ What replaces it: the levers are independent

The scale gain is invariant to what the threshold collected. On this evidence the two levers
act on **disjoint** error mass — which is what the completeness argument said all along, and
which nobody had measured:

- per-class τ reaches the **94.0%** of the residual that fell below threshold
- the scale reaches the **6.0%** the catch-all won at the argmax, plus the **9.9%** confused
  with another real class

⚠️ Two strata of one dataset. This is a *within-dataset dissociation*, not a law, and the
three-dataset totals are still not constant (+5.67, +2.32, +2.41). Independence is what the
data show here; it inherits substitution's gate and needs a fourth dataset too.

---

## 3. ⭐⭐ The paper's largest scope limitation is REPAIRED

`CLAUDE.md` currently carries this in bold:

> ⛔ ⭐ **The +1.18 is a RURAL result** — rural +2.77 ± 0.92 (5/5 folds), **urban +0.10 ± 0.39
> (2/5 folds) — urban is not distinguishable from zero.** ⚠️ Never quote +1.18 without this
> breakdown.

On the identical folds, on the identical tiles:

| LoveDA urban, 677 tiles | mean | sd | folds+ | gate |
|---|---|---|---|---|
| lever 1 — per-class τ | +0.08 | 0.22 | 2/5 | ⛔ fail |
| ⭐ **lever 2 — per-class scale** | **+1.25** | **0.56** | **5/5** | ✅ **PASS** |

> ⭐⭐ **Lever 2 works precisely where lever 1 fails.** The domain that made the method a
> "rural result" gains **+1.25 ± 0.56** from the second lever, every fold positive.

**Per-domain totals**: rural **+3.99**, urban **+1.33**. The method is now positive on both
halves of LoveDA under the project's own gate, and the caveat narrows from *"it does not work
on urban"* to *"on urban the gain comes from the second lever, not the first."*

⚠️ **This is a comparison of two levers on the same folds, and the random controls are
irrelevant to it.** The controls answer a different question (§4).

---

## 4. ⚠️ Both controls fail the gate — and the control's design has a real limit

| | mean | sd | folds+ |
|---|---|---|---|
| control 677 | +0.66 | **1.21** | 3/5 |
| control 992 | +0.89 | **0.93** | 5/5 |
| *rural / urban* | *+1.22 / +1.25* | ***0.53 / 0.56*** | *5/5* |

**A random draw of the same size has roughly twice the fold-to-fold spread of a stratum**, and
neither control clears the gate the strata clear. Stratifying reduces variance, which is the
opposite of the OpenEarthMap situation where 384 tiles could not resolve the effect at all.
✅ **S4 passes**, and 677 tiles is enough here.

⚠️ **But own what this control is not.** A random subset of LoveDA is **not a null** — LoveDA
genuinely has a +1.16 scale gain, so the control's mean should be positive and is. It bounds
**variance**, not **effect**. Comparing urban's +1.25 against control-677's +0.66 is a ~1σ
difference and licenses nothing; the urban claim in §3 rests on the lever-vs-lever comparison
on shared folds, not on this table.

---

## 5. Per class — and the fit rediscovers §9g unprompted

Δ IoU, C over B:

| | catch-all | building | road | water | barren | forest | agri | **full** | **excl.** |
|---|---|---|---|---|---|---|---|---|---|
| rural | **−3.20** | +0.14 | +0.10 | **+4.34** | +1.41 | **+4.98** | +0.78 | **+1.22** | **+1.96** |
| urban | +0.55 | +0.32 | +0.23 | **+4.71** | +0.10 | **+2.19** | +0.68 | **+1.25** | **+1.37** |

**Positive in both metrics in both domains**, so no catch-all artefact in either direction.
⚠️ Rural's catch-all pays 3.20, so rural's excluded figure (+1.96) is the larger one — quote
both, per §9h.

⭐ **`water` and `forest` carry it in both domains** — the same two classes §9g measured as
**85%** of the rural/urban τ gap. The second lever moves the same classes as the first.

⭐ **And the fitted scales differ between domains on exactly the class §9g explained.**

| | bg | building | road | water | barren | **forest** | agri |
|---|---|---|---|---|---|---|---|
| rural `w` | 0.36 | 0.63 | 0.83 | 2.48 | 1.08 | **2.43** | 0.91 |
| urban `w` | 0.40 | 0.56 | 1.02 | 2.60 | 1.10 | **1.33** | 1.14 |
| ratio | 1.11× | 1.12× | 1.23× | 1.05× | 1.02× | **1.83×** | 1.25× |

Six of seven agree within 25%. **`forest` is 1.83×** — and §9g found rural forest at recall
**9.9** against urban's **68.9**, "opposite regimes". The fit was given no labels about
domain and recovered the one class that genuinely differs. ✅ **S5 passes.**

⛔ **`w` therefore does not transfer across domains either**, same as τ (§9e). Calibrate on
the distribution you evaluate on — the rule is unchanged, and now holds for both levers.

---

## 6. Predictions scored

| | prediction | measured | |
|---|---|---|---|
| **S1** | scale gain **larger on urban** than rural | +1.25 vs +1.22, **diff +0.03 ± 0.35** | ⛔ **substitution refuted** — no difference, with power to see one |
| **S2** | urban's scale gain clears +1.00 | **+1.25** | ✅ |
| **S3** | rural's scale gain below its τ gain | +1.22 < +2.77 | ✅ *(weak prediction)* |
| **S4** | strata beat their size-matched controls | strata sd 0.53/0.56 vs 1.21/0.93; strata pass the gate, controls fail | ✅ |
| **S5** | fitted `w` differs between domains | mean \|Δ\| 0.288, **forest 1.83×** | ✅ |

---

## 7. ⚠️ A verdict-text defect, the fourth of its kind

`argmax_reorder.py` printed on the **control** arms:

> ⛔ *Not readable: the fitted scales move 17% between folds. Two causes to separate:
> (1) a partial or mixed cache … (2) a search subsample too thin.*

Neither applies. It is a **random 677-tile draw by construction**, so unstable fits are the
expected result and are the measurement, not a fault. The script has no way to know it is
running a control, and its diagnostic prose asserts causes it did not test.

⭐ **The tables were correct in all four arms; only the prose was wrong.** That is the fourth
occurrence of this exact failure in this codebase (`WEEK3 §11`). `--control` now needs its own
verdict branch, and until it has one the control arms' verdict sections are not to be quoted.

---

## 8. What this changes

| | |
|---|---|
| ⛔ **substitution** | **retired** — refuted with power on its first committed prediction |
| ⭐ **independence** | the levers act on disjoint error mass; gated on a fourth dataset, like substitution was |
| ⭐⭐ **urban** | **repaired** — +1.25 ± 0.56, 5/5 folds, where lever 1 gave +0.08 and failed the gate |
| ⭐ **§9e** | independently reproduced through a different script and cache |
| ⛔ **`w` transfer** | fails across domains, as τ does; `forest` 1.83× is the mechanism |
| ⚠️ **next** | urban's +1.25 is a cached-histogram prediction. §9c's rule applies: **verify end-to-end in the segmentor before it is written up.** |
