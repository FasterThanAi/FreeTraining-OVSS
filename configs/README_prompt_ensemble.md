# Prompt-ensemble vocabularies — and the arity confound they control for

⛔ **The published `cls_loveda.txt` gives different classes different numbers of prompts:**

| class | prompts | |
|---|---|---|
| background, road, water, agricultural | **1** | |
| building, forest | **2** | `building,house` · `forest,tree` |
| barren | **3** | `barren,bareland,soil` |

A class's queries are collapsed by **`max`** (`segearthov3_segmentor.py`, the `num_cls !=
num_queries` branch). **The maximum of three samples is systematically larger than the maximum of
one**, so `barren` carries a score bonus over `road` that has nothing to do with whether the model
sees barren land better. That is an uncontrolled variable in the baseline, and any prompt
experiment that ignores it measures arity and calls it semantics.

## The three vocabularies

| file | prompts per class | what it isolates |
|---|---|---|
| `reference/cls_loveda.txt` | **1–3, unequal** | the published baseline |
| `cls_loveda_k1.txt` | **exactly 1** | the baseline with the arity bonus removed |
| `cls_loveda_k3.txt` | **exactly 3** | more prompts, arity held equal |

⭐ **Two comparisons, each controlled:**

- **k1 vs baseline** — what are the hand-picked synonyms actually worth? This is the only way to
  find out, because the baseline changes arity and vocabulary together.
- **k1 vs k3** — does adding prompts help, with arity identical across classes?

⚠️ **`background`'s three prompts are `background, other, unlabeled`, and they are not visual
concepts.** SAM 3's median presence score for `background` is 0.022 — it has nothing to detect —
so these add arity without adding evidence. That is deliberate: equal arity is the point of the
design, and if the catch-all's score inflates while its IoU falls, the arity confound is visible
rather than hidden. Reported, not hidden.

⚠️ **These prompts were written by hand, as the baseline's were.** No prompt search was run, and
none of the three vocabularies was selected on evaluation data.
