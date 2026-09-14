# Prompt ensembling — ⛔ both arms lose, and one word is worth more than every lever combined

**14 Sep 2026, LoveDA val, 1669 tiles, τ = 0.5.** Predictions in
`prereg/predict_prompt_ensemble.md` (`1600ceb`), committed before the runs.

| arm | prompts per class | mIoU | vs baseline | s/image |
|---|---|---|---|---|
| baseline | 1, 2, 1, 1, 3, 2, 1 | **47.38** | — | 0.85 |
| **`k1`** | exactly 1 | **40.90** | ⛔ **−6.48** | 0.67 |
| **`k3`** | exactly 3 | **45.13** | ⛔ **−2.25** | 1.37 |

---

## 1. ✅ The sanity check first

The four classes that already had **one** prompt are unchanged in `k1`, to within ±0.07:

| background | road | water | agricultural |
|---|---|---|---|
| 45.50 → 45.47 | 53.89 → 53.82 | 51.44 → 51.48 | 47.47 → 47.49 |

Identical prompts, identical scores. Whatever moved, moved for a reason.

---

## 2. ⭐⭐⭐ The finding: one word is worth 34 IoU

Strip the synonyms and only the classes that lost words collapse:

| class | words removed | IoU |
|---|---|---|
| ⭐ **barren** | `bareland`, `soil` | **35.73 → 1.17** |
| forest | `tree` | 33.78 → 24.31 |
| building | `house` | 63.81 → 62.54 |

> ⭐⭐⭐ **The word `barren` alone finds essentially no barren land — IoU 1.17, recall 1.17% at
> 75.6% precision. It is right when it fires and it almost never fires. Adding `bareland` and
> `soil` takes that class from 1.17 to 35.73.**

**Two words are worth +34.56 IoU on one class — +4.94 mIoU on the benchmark.** For comparison,
everything else measured in this project:

| | Δ mIoU |
|---|---|
| ⭐ **two words in the vocabulary** | ⭐ **+4.94** |
| per-class τ | +1.18 |
| per-class scale | +1.16 |
| dihedral TTA | +0.29 |
| head fusion / presence weight / affine bias | null |

> **The single largest lever in this pipeline is the choice of words, and it had already been
> pulled by hand before we arrived.**

⚠️ **And that is a fragility of the field, not a feature.** A published benchmark number depends on
an unreported, hand-chosen vocabulary to the tune of ~5 mIoU. Nobody reports prompt sensitivity,
and on this evidence nobody should be reporting a benchmark number without it.

---

## 3. ⭐ R3 confirmed: the arity confound is real and visible

**Prediction R3, committed before the run:** *`background`'s IoU will fall under `k3`, because its
extra prompts (`other`, `unlabeled`) are not visual concepts and `max` inflates its score without
adding evidence.*

**Measured: 45.50 → 41.66, −3.84.** ✅

`agricultural` shows the same signature from the other side — recall **62.0 → 68.8** while
precision **66.9 → 57.9**. More prompts means a higher `max`, which means the class fires more
often, which trades precision for recall exactly as a lower threshold would. ⭐ **Prompt count
behaves like a threshold, not like better evidence.**

---

## 4. ⛔ And my `k3` design was flawed — it is not a clean count test

`k3` was meant to hold prompt *content* roughly fixed and vary *count*. It does not: I wrote
`barren, bareland, bare soil` where the baseline has `barren, bareland, soil`. **Changing `soil` to
`bare soil` costs that class about 10 IoU** (35.73 → 25.81), which is most of `k3`'s deficit.

So `k3` confounds count with content, and the `k1`-vs-`k3` comparison **cannot** be read as "does
adding prompts help". ⭐ `k1` vs baseline is clean and answers R1 decisively; `k3` answers R3 and
nothing else. **R2 is not testable from these runs.**

---

## 5. ⭐⭐⭐ R5 confirmed, quantitatively — the encoder is shared

Fitting `time = a + b × queries` across the three arms (7, 11, 21 queries):

| | |
|---|---|
| fixed cost (the vision encoder) | **0.309 s** per image |
| marginal cost per prompt | **0.050 s** |
| predicted vs measured | 0.660/0.667, 0.860/0.850, 1.362/1.365 |

> ⭐ **Tripling the prompts costs 2.06× the time, not 3×. Doubling the views cost exactly 2.0×.**
> Prompts are structurally cheaper than views, and the margin grows with prompt count.

---

## 6. ⭐⭐ What this opens: a fitted vocabulary

The baseline hand-tunes its prompts, exactly as it hand-tunes its single threshold. **We replaced
the hand-tuned threshold with a fitted per-class one on ~200 labelled tiles. The same move is
available here, and the evidence says the prize is far larger.**

⛔ **Screen before building, per the rule.** Cache K candidate prompts per class once, then compute
*"best prompt subset per class, chosen on the evaluation fold"* — the peeking ceiling. If that is
small, the idea dies for one caching run.

⚠️ Storage is the constraint: 6 prompts × 7 classes = 42 channels ≈ 88 MB/tile, so a 300-tile
sample (~26 GB) is the screen and the full split only follows a large ceiling.
