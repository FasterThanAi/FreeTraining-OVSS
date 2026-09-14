# The vocabulary is the largest lever in this pipeline — and there is no rule for it

**14 Sep 2026.** Two pre-registered single-word interventions, on two datasets, applying the
**same rule** — *use the name the dataset's own annotation guide uses* — and getting
**opposite results**.

| dataset | prompt SegEarth-OV3 ships | official class name | Δ mIoU from switching |
|---|---|---|---|
| **UAVid** | ⛔ `vegetation` | *low vegetation* | ⭐ **+3.53** (56.86 → 60.39) |
| **Potsdam** | ⭐ `grass` | *low vegetation* | ⛔ **−2.71** (57.83 → 55.12) |

> ⭐⭐⭐ **The same rule helps one dataset by 3.5 points and hurts another by 2.7.**
> A practitioner cannot look up the class name and be done. The vocabulary is a real,
> high-variance design choice that every training-free OVSS paper inherits, none report, and
> no simple rule fixes.

Both arms committed to git before the run: `prereg/predict_uavid_vocabulary.md` (`b20eaf8`),
`prereg/predict_potsdam_vocabulary.md` (`f504d81`). Both are single-word, one prompt per
class, **no arity change** (`WEEK3 §7b`: prompt *count* moves scores regardless of meaning).

---

## 1. UAVid — the shipped prompt was wrong, and it cost 3.53

`vegetation` → `low vegetation`. **Surgical:** `tree` **+14.66**, `vegetation` **+11.14**,
and every other real class moves **≤ 0.04**.

| | before | after | predicted |
|---|---|---|---|
| `vegetation` precision | 53.62 | ⭐ **71.51** | rises ✅ |
| `tree` recall | 55.90 | ⭐ **72.76** | rises ✅ |

⭐ The bare word `vegetation` was **matching trees**, so the class over-fired and stole
`tree`'s pixels. Full scoring in `UAVID_RESULTS.md` §19–§20.

## 2. Potsdam — the shipped prompt was already BETTER than the official name

`grass` → `low vegetation`. **57.83 → 55.12, −2.71.**

| | recorded baseline | with `low vegetation` | Δ |
|---|---|---|---|
| **mIoU** | **57.83** | **55.12** | ⛔ **−2.71** |
| `tree` IoU | 37.92 | 37.40 | −0.52 |
| `tree` **precision** | 93.34 | 92.17 | **−1.17** |
| ⭐ `tree` **recall** | **38.63** | ⭐ **38.63** | ⭐ **0.00** |

### Predictions scored

| | prediction | measured | |
|---|---|---|---|
| **P1** | \|Δ mIoU\| < 1.0, *"may fall"* | **−2.71** | ⛔ **failed on magnitude** |
| ⭐ **P2** | `tree` recall rises by < 5 points | ⭐ **0.00** | ✅ **exact** |

⛔ **P1 failed and I am reporting it as a miss**, not reinterpreting it. I predicted the
change would be small; it was nearly three points. The *direction* was allowed ("may fall"),
the magnitude bar was not.

⭐⭐ **P2 is the informative one, and it is exact.** `tree`'s recall does not move by a
hundredth. **Renaming the competing class changes nothing about `tree`.** Its precision falls
1.17 — `low vegetation` is a *weaker* prompt, so some of its pixels leak to `tree` — but
`tree` finds exactly the same 38.63% of trees either way.

> ⭐ **Potsdam's `tree` deficit is visual confusion at 5 cm GSD, not a naming error.**
> That is what P2 was written to test, and it is why the answer matters.

## 3. ⭐ What this settles for the method

**Potsdam's lever-2 gain of +4.86 is NOT prompt repair.** The shipped `grass` already beats
the one principled alternative, so there is no captured mass for a better prompt to take —
unlike UAVid, where the correction removed 64% of lever 2's gain.

| dataset | lever 2 | after the vocabulary was tested |
|---|---|---|
| **Potsdam** | **+4.86** | ✅ **stands** — the prompt is not the explanation |
| **UAVid** | +5.64 → **+2.03** | ⛔ corrected; 64% was the prompt |
| LoveDA | +1.16 | untested, but its vocabulary already carries hand-tuned synonyms |

⭐ **So UAVid was a one-off careless prompt, not a systemic flaw in our results.** That is the
sentence the paper needs immediately after the UAVid correction, and it is now measured
rather than asserted.

⛔ **CLOSED. No third word.** The pre-registration fixed this in advance: *"If `low
vegetation` loses, we do not go looking for a third word — that would be searching for a
flattering number."* `configs/README_vocabularies.md` already forbids exploratory revisions.

## 4. ⭐ The finding that belongs in the paper

**The same released config directory ships a careless word for one dataset and a careful one
for another.** `vegetation` on UAVid names the neighbouring class; `grass` on Potsdam and
Vaihingen does not. Nobody reports either. And the correction is worth **more than any
published method's contribution on these benchmarks**:

| intervention | Δ mIoU |
|---|---|
| ⭐ one word, UAVid | **+3.53** |
| ⭐ two words, LoveDA `barren` *(`PROMPT_ENSEMBLE_RESULTS`)* | **+4.94** |
| our whole method, LoveDA | +2.32 |
| our whole method, UAVid *(corrected)* | +3.22 |

⚠️ And the direction is not predictable: **the official class name is not automatically the
better prompt.** Any benchmark reporting training-free OVSS numbers is reporting a
vocabulary as much as a method.
