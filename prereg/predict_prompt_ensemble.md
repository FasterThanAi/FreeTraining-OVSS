# Pre-registration — prompt ensembling, with the arity confound controlled

**Written 14 Sep 2026, BEFORE the runs.** `git log` is the timestamp. Not edited afterwards.

---

## Why this and not another lever

Five per-class corrections have now been fitted and only the two that reshape the **decision**
work. The one thing that raised **precision and recall together** was test-time augmentation —
because it *adds evidence* rather than reallocating it. But TTA's gain is absorbed by per-class
scaling (`TTA_RESULTS.md` §2b): both act on the argmax, so whichever comes second finds the work
done.

⭐ **Prompts are the other way to add evidence, and they may not be absorbed for a reason TTA was.**
TTA asks the *same question* of a *different view* and averages out noise — which is what scaling
also smooths. A different prompt asks a *different question*: `water` and `river` and `lake` may
light up genuinely different pixels. That is **coverage**, not noise reduction, and per-class
scaling has no route to it.

⭐ **And prompts are structurally cheap.** `set_image` is called **once per image**, outside the
prompt loop, so the vision encoder runs once however many prompts are supplied; only the text
encoder, fusion encoder and heads repeat. A second view costs a full extra forward pass; a second
prompt does not.

---

## ⛔ The confound this design exists to control

The published `cls_loveda.txt` gives classes **1, 2, 1, 1, 3, 2, 1** prompts and collapses each
class's queries by **`max`**. The maximum of three samples is systematically larger than the
maximum of one, so `barren` carries a score bonus over `road` that is pure arity.

**Any comparison that changes prompt count and prompt content together measures arity and calls it
semantics.** So three vocabularies, two of them equal-arity:

| arm | prompts per class | isolates |
|---|---|---|
| `base` | 1–3, unequal | the published configuration |
| **`k1`** | **exactly 1** | the baseline minus the arity bonus |
| **`k3`** | **exactly 3** | more prompts, arity equal |

---

## Predictions

**R1 — `k1` will land within ±0.5 mIoU of the baseline**, sign not predicted.
*Why:* the hand-picked synonyms add coverage, the unequal arity distorts the argmax, and the two
pull opposite ways. ⭐ **Whatever the sign, this is the first measurement of what those synonyms
are worth**, because the baseline never varied them with arity held fixed.

**R2 — `k3` will beat `k1`** by more than +0.3.
*Why:* three questions per class cover more of each concept than one, with arity identical so the
comparison is clean. ⚠️ My record on "this should work" is **1 for 6**; stated with low confidence.

**R3 — ⭐ `background`'s IoU will FALL under `k3`.** Its extra prompts (`other`, `unlabeled`) are
not visual concepts and SAM 3's median presence for the catch-all is 0.022, so they add arity
without evidence and `max` inflates its score. **This is the sharp prediction** — a named class, a
direction, and it makes the arity confound visible rather than hidden.

**R4 — any `k3` gain will be at least half absorbed by per-class scaling**, as TTA's was.
*Why:* more prompts raise scores, and the scale rescales scores. ⚠️ **If it is NOT absorbed, that
is the interesting outcome** — it would mean prompts add coverage rather than magnitude, which is
a different kind of evidence from anything else tested here.

**R5 — `k1` will be the FASTEST arm** (7 queries against the baseline's 11), and `k3` (21 queries)
will cost well under 3× the baseline, because the encoder is shared.
*Why:* it tests the shared-encoder claim above, which is the argument for preferring prompts to
views. A near-3× cost would refute it.

---

## The branch table, written now

| outcome | reading |
|---|---|
| **R2 ✅ and R4 ⛔** | ⭐⭐ the best available result: prompts add evidence the decision levers cannot reach, and the gains stack. Worth a section. |
| **R2 ✅ and R4 ✅** | another substitute for the scale, like TTA. Report it beside TTA as a second instance of the same rule — *adding evidence and fixing the decision are substitutes* — which strengthens that claim from one case to two. |
| **R2 ⛔** | prompt count is not the lever; the baseline's hand-picked vocabulary is already adequate. ⭐ Then **R1 still stands on its own** as the first controlled measurement of what those synonyms are worth. |
| **R3 ✅** | the arity confound is real and measurable in a published configuration. Worth a sentence regardless of R2. |

⛔ **No vocabulary here was selected on evaluation data**, and no prompt search was run. All three
lists were written by hand before any of them was evaluated, exactly as the baseline's was.
