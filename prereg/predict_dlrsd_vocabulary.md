# Pre-registration — the DLRSD vocabulary arm

**Committed before the new vocabulary is cached or evaluated.** `git log` is the
timestamp. Nothing here is edited afterwards; the scoring goes in
`DLRSD_RESULTS.md`.

## Why this is a separate experiment and not a fix

`prereg/predict_dlrsd.md` fixed the vocabulary as DLRSD's 17 published class names,
verbatim, *"not negotiable after the fact"*. ⛔ **That still holds.** The measured
result — 37.27 → 44.42 — keeps the published vocabulary, and nothing below changes
it. Editing a prompt because a result disappointed is exactly what a
pre-registration exists to prevent.

⭐ What has changed is that three prompts are now suspect on **evidence gathered
under the committed protocol**, not on a hunch:

| prompt | evidence |
|---|---|
| ⛔ `chaparral` | **0.00 IoU** at every rung — baseline, lever 1, lever 2. Never wins a pixel |
| ⛔ `mobile home` | **0.00 IoU** at every rung, and fitted at **w = 2.488**, the grid ceiling. The fit pushes as hard as the search allows and it still reaches 0.29 |
| ⚠️ `field` | worst precision of any class (**16.39%**), worst discard rate (**30.84%**), **18 of the 22 tiles annihilated** are agricultural, and lever 1 *hurts* it (13.91 → 11.47) |

**Two classes contributing nothing is 11.8% of a 17-class metric.**

## The intervention, and the rule it follows

⛔ **One line changed per class. Arity unchanged — one prompt per line, no synonyms
added**, so this cannot be confused with prompt ensembling (`WEEK3 §7b`).

⭐ **Two rules, both stated from the WORD alone, not from the data beyond the fact
that the class fails:**

**Rule A — an unfamiliar or regional term is replaced by the common English word for
the same visual thing.**

| | from | to |
|---|---|---|
| A1 | `chaparral` | **`shrubs`** — chaparral is a regional Californian biome name; the pixels are low woody scrub |
| A2 | `mobile home` | **`trailer`** — the ordinary term for these units |

**Rule B — an ambiguous word is disambiguated by one modifier.**

| | from | to |
|---|---|---|
| B1 | `field` | **`crop field`** — "field" alone also means a sports field or any open ground, and DLRSD has `court`, `grass` and `bare soil` competing for exactly that reading |

⭐ **The three effects are separable by construction.** Each class is an independent
forward pass with its own text prompt, so changing `chaparral`'s word leaves the other
sixteen channels **bit-identical**. The only coupling is the argmax. That is checkable
and is prediction W6.

⚠️ **This rule is not "use the dataset's own class name"** — DLRSD already does, which
is why that rule has nothing left to give here. `VOCABULARY_RESULTS` measured that rule
at **+3.53 on UAVid and −2.71 on Potsdam**, so no vocabulary rule in this project has
survived two datasets, and this one is not claimed to either.

---

## Predictions

**W1 — `chaparral` rises above 0.00 IoU.** It cannot fall. ⚠️ Low information on its
own, which is why the bar is **above 5.0**, not above zero.

**W2 — `trailer` rises above 0.00 IoU**, bar **above 5.0**.
⚠️ Weaker than W1. `mobile home` may be failing because the units look like
`buildings` rather than because of the word, and a rename cannot fix a visual
confusion. `VOCABULARY_RESULTS` recorded exactly that on Potsdam: renaming `tree`'s
competitor left its recall **38.63 → 38.63, unchanged to a hundredth**.

**W3 — `crop field` raises `field`'s precision above 16.39%**, and its discard rate
falls below **30.84%**.

**W4 ⭐ — the baseline rises by at least +2.0 mIoU** (37.89 → ≥ 39.89 on the full
split). *Reason:* two dead classes each own 5.88% of the metric, so lifting them to a
merely average IoU is worth ~+3.5 alone. ⚠️ Moderate confidence — it assumes the
words, not the pixels, are the problem, and W2's caveat says that may be false for at
least one of them.

**W5 ⭐⭐ — THE ONE THAT MATTERS. The discard rate falls below 6.16%.**
*Reason:* `D1` failed — DLRSD has **0.00% catch-all share** and should have had the
smallest residual in the project (predicted < 3.78%), yet measured **6.16%**, the
second-largest. Two suspects were named and neither established: **17 competing
classes** and **3.9× upsampling**. ⭐ **A vocabulary this arm can fix is a third, and
it is the only one that is cheap to test.**

| outcome | what it means for D1 |
|---|---|
| discard falls **below 3.78%** | ⭐⭐ the bad prompts caused D1's failure. §7's rule survives on a corrected vocabulary, and the miss becomes a statement about *prompt quality*, not about the mechanism |
| falls, but stays **above 3.78%** | prompts are part of it; 17 classes or the upsampling carry the rest. Report both |
| ⛔ **does not fall** | the vocabulary is exonerated. **D1's failure is structural** — a 0% catch-all does not give the smallest residual — and §7's extrapolation is dead on its own terms |

**W6 — the other 14 classes move by less than 0.5 IoU each at the baseline.**
*Reason:* independent forward passes per prompt; the only coupling is the argmax.
⭐ **This is the control.** If untouched classes move a lot, something other than the
three words changed and nothing else here can be read.

**W7 — the lever gains SHRINK.** Lever 1 + lever 2 total falls below the measured
**+7.15**. *Reason:* on UAVid, correcting one word removed **64%** of lever 2's gain,
because the levers were repairing what the prompt broke. ⚠️ If the total instead
holds, the levers and the vocabulary are fixing **different** error mass on this
dataset — which would be new, since UAVid found them to be substitutes.

---

## Branch table — decided now

| outcome | reading |
|---|---|
| **W6 fails** | ⛔ **stop.** Untouched classes moved, so the run is not a controlled comparison. Find out why before reading anything else. |
| W1 and W2 both hold | ⭐ two dead classes were a naming problem, and the paper gains a concrete instance of the vocabulary lever on a fifth dataset |
| W1 holds, **W2 fails** | ⭐ more informative than both holding: `chaparral` was a word the encoder did not know, `mobile home` is a **visual** confusion a rename cannot reach. That is the Potsdam `tree` result replicating. |
| **W5 holds strongly** | ⭐⭐ D1's failure is explained and the explanation is testable; the strongest available outcome |
| ⛔ **W5 fails** | the vocabulary is exonerated and D1's failure stands as structural. **Report it as such** — that is a finding about our own mechanism, not a disappointment |
| W7 fails (gains hold) | ⭐ the levers and the vocabulary address different error mass here, unlike UAVid. Worth a paragraph. |
| everything fails | ✅ publishable: three prompts chosen by a stated rule, on a dataset where two classes score zero, and none of it helps. **That bounds the vocabulary lever**, which no paper in this area has done. |

---

## What is reported either way

⛔ **The method's DLRSD result stays 37.27 → 44.42 on the published vocabulary.**
Any gain here is reported as **its own result**, exactly as UAVid's +3.53 was — never
folded into the method's number. ⚠️ And if the corrected vocabulary scores higher, the
method's contribution is re-measured **on top of it**, because keeping a number
obtained with a prompt we know to be wrong, on the grounds that it flatters the method,
is the thing this file exists to prevent.
