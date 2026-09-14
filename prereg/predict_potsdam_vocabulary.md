# Pre-registration — does the vocabulary explain Potsdam's lever-2 gain too?

**Committed 14 Sep 2026, before any Potsdam run with a changed prompt.** `git log` is the
timestamp.

## Why ask

On UAVid, correcting one word took **64%** of lever 2's gain (+5.64 → +2.03). Potsdam's
lever-2 gain is **+4.86**, the project's second largest, and **72% of it is `tree`**
(w = 4.04, +21.56). A reviewer who reads the UAVid section will ask about Potsdam
immediately. We ask first.

## ⭐ But the prompts are NOT the same, and that is the interesting part

| dataset | official class name | prompt SegEarth-OV3 ships | |
|---|---|---|---|
| UAVid | *low vegetation* | ⛔ **`vegetation`** | too BROAD — also names trees |
| **Potsdam / Vaihingen** | *Low vegetation* | ⭐ **`grass`** | specific; does **not** name trees |

**The same authors chose a careless word for one dataset and a careful one for another.**
That inconsistency is itself the "inherited, hand-written, unreported" point, whichever way
this run lands.

## The intervention — one variable, chosen by rule, not by search

`cls_potsdam.txt` line 3: **`grass` → `low vegetation`**, the name the ISPRS annotation guide
uses. ⛔ **One arm only, and the word is chosen by a rule applied uniformly** (*use the
dataset's own class name*), not by trying words until one helps. `configs/
README_vocabularies.md` forbids exploratory revisions and that stands.
✅ Arity unchanged: one prompt per class, no commas, before and after.

## Predictions

**P1 ⭐ — I expect this to FAIL to help.** Baseline mIoU changes by **less than ±1.0**, and
may fall. *Reason:* `grass` already names the thing and does not name trees, so UAVid's defect
— a prompt matching the neighbouring class — is absent. *"Low"* is also an awkward modifier
for a vision-language model, where *grass* is a concrete visual concept.
⚠️ **Note this is the opposite direction from UAVid, and it is predicted before the run.**

**P2.** `tree`'s recall (38.63) rises by **less than 5 points**. *Reason:* if `tree`'s
residual is going to `grass`, that is visual confusion at 5 cm GSD, not a naming error, and
renaming the competitor cannot fix it.

**P3 — conditional, needs the full run.** If P1 and P2 hold, Potsdam's **+4.86 survives** and
is a genuine model limitation rather than prompt repair.

## Branch table — written before the number

| outcome | what it means |
|---|---|
| **P1 and P2 hold** *(expected)* | ⭐ Potsdam's vocabulary is adequate, its +4.86 stands, and **UAVid was a one-off careless prompt rather than a systemic flaw in our results.** The paper reports the contrast between `grass` and `vegetation` as evidence that vocabularies are hand-made and uneven. |
| **baseline rises > +1.0** | ⛔ same defect, different word. Run the full lever test on Potsdam and **correct that headline too**, exactly as UAVid was corrected. |
| **baseline falls > 1.0** | ⭐ also informative: it shows the *official* class name is not automatically the best prompt, which tempers the UAVid finding rather than extending it. Report it — it is the harder result to publish and the more honest one. |

⛔ **Whatever happens, this is the only Potsdam vocabulary arm.** If `low vegetation` loses,
we do **not** go looking for a third word. That would be searching for a flattering number,
which is what the UAVid pre-registration was written to prevent.
