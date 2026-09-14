# Pre-registration — DLRSD as a fifth dataset

**Committed 14 Sep 2026, before any DLRSD inference has been run, before a config or a
vocabulary exists, and before a single cached score.** `git log` is the timestamp. Nothing
here is edited afterwards; the scoring goes in `DLRSD_RESULTS.md`.

⚠️ **This pre-registration has to do more than the others.** For LoveDA, Potsdam, UAVid and
OpenEarthMap the baseline shipped a config, a vocabulary and a tuned `prob_thd`, so the only
free choices were ours to make *after* the baseline. **DLRSD ships none of those** — `ls
configs/ | grep -i dlrsd` returns nothing, and SegEarth-OV3's published table has no DLRSD
row. So the vocabulary, the threshold, the fold protocol and the discard sink are all
**ours**, and every one of them could be tuned to flatter the method. They are therefore
fixed here, in advance, with reasons.

---

## 0. What is already measured, with no GPU

From ground truth alone (`scripts/inspect_dataset.py`, `scripts/dlrsd_class_map.py`):

| | |
|---|---|
| images / labels | **2100 / 2100**, stems match exactly |
| tile size | **256 x 256**, every file |
| total labelled pixels | **137,625,600 = 2100 x 256 x 256, difference zero** |
| label values | **1..17, no gaps, and `0` never appears in any of the 2100 files** |
| classes | **17**, alphabetical, ladder confirmed against the data by category fingerprint |
| ⭐ catch-all share | ⭐ **0.00%** — no background, clutter or unlabelled class exists |
| groups | 21 UC Merced scene categories x exactly 100 images |

⭐ **Every pixel in DLRSD carries a real class.** That is the first time in this project, and
it is the single most important fact about the dataset.

---

## 1. Design decisions, fixed now

### 1a. The vocabulary — DLRSD's own 17 class names, verbatim

    airplane · bare soil · buildings · cars · chaparral · court · dock · field · grass
    mobile home · pavement · sand · sea · ship · tanks · trees · water

⛔ **No synonyms, no rewording, no "prompt engineering", and this is not negotiable after
the fact.** `VOCABULARY_RESULTS.md` measured that the vocabulary is the largest lever in this
pipeline — one word was **+3.53** on UAVid, two words **+4.94** on LoveDA, both larger than
our entire method — and that **no rule exists** for choosing it: the same principled rule
(*use the dataset's own class name*) gained 3.53 on UAVid and lost 2.71 on Potsdam.

With no published vocabulary to inherit, we would otherwise be tuning the biggest lever in
the pipeline with nothing to anchor it. So we commit to the one rule the project has already
pre-registered and scored, and we accept whatever it gives. ⚠️ **If a better DLRSD
vocabulary exists, our number is an underestimate, and that is the correct direction for the
error to run.**

### 1b. No `background` prompt, and the discard sink is unscored

The vocabulary is exactly the 17 classes, so it **covers the label space exactly**. The
segmentor still needs somewhere to put sub-threshold pixels:

    seg_pred[max_vals < prob_thd[seg_pred]] = bg_idx

`bg_idx` points at an **18th index that has no prompt, no GT pixels, and is excluded from
the metric.** A discarded pixel therefore becomes a false negative for its true class and a
false positive for nothing.

⭐ **Consequence, and it is a genuine structural difference from every other dataset here:
on DLRSD, discarding a WRONG pixel is free.** Elsewhere a thresholded pixel lands in a
scored catch-all and costs that class's precision; here it costs nothing. **So the fitted
thresholds should come out systematically higher than on any other dataset** (see D4).

### 1c. Folds are STRATIFIED within category, not category-disjoint

⛔ **This reverses my own advice from earlier today, and the reversal is the point of having
measured it.** Group-disjoint folds are mandatory wherever groups exist — UAVid's frame-level
split leaked **+0.54 mIoU and two folds**. But `dlrsd_class_map.py` shows **5 classes live in
at most 2 of the 21 categories** (`airplane` 1, `dock` 1, `tanks` 1, `sea` 2, `mobile home`
2), so holding a category out leaves those classes with **no calibration pixels at all**. The
fit would be asked to set a threshold it cannot see and the failure would read as the
method's.

**Primary protocol: split each category's 100 images 20/20/20/20/20 across the five folds**,
so every fold contains every category and every class. ⭐ **Secondary arm: category-disjoint
folds, reported beside it**, exactly as UAVid reported both — there the gap between protocols
*was* the finding.

### 1d. The baseline is the best global τ, FITTED — not an inherited constant

DLRSD has no published `prob_thd`. Two rows are reported, and the second is the one the
method must beat:

| row | τ | why |
|---|---|---|
| reference | **0.1** | the lowest published τ in the project (Potsdam and OEM both use it), for cross-dataset comparability |
| ⭐ **honest baseline** | **best single global τ, fitted on the same calibration tiles** | same data, same budget, one parameter instead of 17 |

⭐⭐ **This is a STRONGER protocol than any other dataset in this project.** Everywhere else
the comparison is against a τ someone else chose, which invites *"you just inherited a bad
threshold"*. Here per-class τ has to beat a global τ tuned on the identical calibration set.
The objection cannot be made.

---

## 2. Predictions

### D1 — the residual is the smallest yet ⭐

At τ = 0.1, DLRSD's real-class discard rate is **below OpenEarthMap's 3.78%**.
Point estimate **2.5%**, bracket **0.5 – 4.0%**.

*Reason.* `WEEK3 §7`'s surviving half — catch-all share predicts residual size — is monotone
over three datasets at τ = 0.1 and was confirmed by a committed prediction on Potsdam:

| dataset | catch-all share | discard @ τ=0.1 |
|---|---|---|
| ⭐ **DLRSD** | ⭐ **0.00%** | ⭐ **predicted < 3.78%** |
| OpenEarthMap | 0.84% | 3.78% |
| Potsdam | 4.29% | 4.69% |
| LoveDA | 36.1% | 10.88% |

⚠️ **Moderate confidence, and two things could break it.** DLRSD has **17 classes** against
OEM's 9, so the discrimination problem is harder and the presence gate is being asked about
far more concepts — which is the regime SegEarth-OV3 built presence gating *for*. And every
tile is **upsampled ~3.9x** to reach SAM 3's 1008², a regime nothing in this project has
run. Either could raise the discard rate.

### D2 — the mechanism split is ~100% threshold

At least **99%** of sink-assigned pixels are mechanism (A), `conf < τ`, and mechanism (B),
an argmax loss to the catch-all, is **exactly 0**.

⚠️ **This is a structural consequence of 1b, not a discovery** — with no background prompt
there is no channel to lose an argmax to. It is listed because it is a **cheap check that
the config is right**: if (B) is non-zero, a background prompt has crept in. LoveDA splits
94.0% / 6.0%.

### D3 — the objective separates exactly ⭐

`--objective real` and `--objective all` produce **identical** threshold vectors,
`max |Δτ| = 0.000`.

*Reason.* `SEPARABILITY_RESULTS.md` proved the real-class objective separates exactly, and
that `all` does **not**, because the scored catch-all couples the vector — Potsdam shows
`max |Δτ| = 0.0050` under `all` against `0.0000` under `real`, and LoveDA's 0.0000 under
`all` is called a coincidence of that dataset. **With no scored catch-all the coupling term
does not exist**, so the two objectives should coincide by construction. ⭐ **High
confidence, and it is a clean test of that file's stated mechanism rather than of its
result.**

### D4 — the fitted thresholds run higher than on any other dataset

The **median** fitted per-class τ exceeds the best fitted **global** τ, and the fitted
vector's median is higher than LoveDA's (0.375) and Potsdam's.

*Reason.* Per 1b, discarding a wrong pixel is free here. Raising τ_c strips both true and
false positives from class c, and elsewhere the false ones reappear as catch-all errors;
here they vanish from the metric entirely. ⚠️ **Low-to-moderate confidence** — this is a
first-principles argument about an objective, and three such arguments (levers 3, 4, 5)
each predicted a small positive and returned a null.

### D5 — the per-class *shape* carries it, not the level

Per-class τ beats the **best fitted global τ** by **≥ +0.5 mIoU** on held-out tiles.

*Reason.* UAVid measured exactly this decomposition under transfer: global τ fitted the same
way gave **+0.12**, per-class gave **+1.03** — the shape was **89%** of the gain. ⚠️ Moderate
confidence; this is the first dataset where the global arm is the *headline* comparison
rather than a diagnostic.

### D6 — the power gate ⭐ GO / NO-GO

Under **stratified** folds, rung A's fold-to-fold mIoU spread is **under 3.0 points**.

*Reason, and this is the prediction that decides whether DLRSD is usable at all.*
OpenEarthMap was killed by exactly this: rung A swung **39.64 → 49.79**, a **10-point**
spread against an effect of 0.4, and `ARGMAX_SCALING_RESULTS.md` concluded its 384 tiles
"cannot resolve a ~1 mIoU effect". DLRSD has 2100 tiles — 420 per evaluation fold against
OEM's 77 — and stratification puts **the same category mix in every fold**, which is the
specific protection OEM lacked.

⛔ **If the spread exceeds 3.0, DLRSD is reported as a reproduced baseline and a catch-all
data point only, with no lever column**, and that is written up as a limit of the
measurement rather than a finding about the method. Same verdict OEM received.

### D7 — the gain concentrates in the rare classes

Of the two classes contributing most to lever 1's gain, **at least one** comes from
{`airplane`, `dock`, `tanks`, `court`, `ship`, `mobile home`}.

*Reason.* mIoU is an unweighted mean over 17 classes, so each owns **5.88%** of the metric
however small it is. Those six classes are **6.44% of the pixels and 35.3% of the metric** —
a 5.5x leverage. UAVid's `human` is the precedent: **0.19%** of pixels, 14.3% of the metric,
and it carried lever 1 single-handedly at **+6.47**.

⚠️⛔ **Whatever the outcome, the per-class table is reported with the headline.** `WEEK3 §9h`
is at its most extreme on this dataset, and a mean over 17 classes where six of them are
under 2% of the pixels is exactly the arithmetic that let OEM's +2.28 and UAVid's +5.89 be
misread. **No DLRSD mean is ever quoted alone.**

---

## 3. Branch table — what each outcome means, decided now

| outcome | reading |
|---|---|
| **D6 fails** | ⛔ DLRSD joins OpenEarthMap: underpowered, baseline row only. **Not a finding about the method.** Stop before the levers. |
| D6 holds, D1 holds | ✅ §7's surviving half extends to a fifth dataset and a **0% extreme**, with a committed prediction. The strongest available outcome for the mechanism. |
| D6 holds, **D1 fails high** | ⭐ More interesting than a pass. Either the 17-class vocabulary or the 3.9x upsampling drives the residual independently of catch-all share — **report it as a limit on §7 discovered by extrapolation**, and say which of the two is the suspect without claiming to know. |
| D3 fails | ⛔ `SEPARABILITY_RESULTS.md`'s stated mechanism is wrong — the coupling is not the catch-all. That is a correction to a load-bearing file and must be written up as one. |
| levers positive, D7 holds | ⚠️ Correct, and **must** be reported as "carried by rare classes", never as a clean mean. |
| levers positive, D7 fails | ⭐ Better news than D7 holding: the gain is spread across large classes, which is the first time in the project. |
| levers null | ✅ Publishable. A dataset with **no catch-all and a perfectly covering vocabulary** is where the method has least to repair, and a null there **supports** the mechanism rather than denting it. Say so plainly. |

---

## 4. What this dataset is worth, stated before the result

⭐ **The reason to run DLRSD is not a fifth row.** It is that three standing objections to
this project's numbers cannot be raised here, by construction:

1. ⛔ *"The gain is a repaired catch-all."* Qualified OEM's +2.28 (110% `background`),
   ConInfer's +2.51 (a third), and Potsdam's baseline (depressed 8.18 points by `clutter`).
   **DLRSD has no catch-all, so full mIoU equals catch-all-excluded mIoU by construction.**
2. ⛔ *"You inherited a bad global threshold."* Per 1d the comparison is against a global τ
   **fitted on the same calibration tiles**.
3. ⛔ *"The vocabulary was tuned."* Per 1a it is the published class list, verbatim,
   committed here before any run.

⚠️ And one objection this dataset **invites** instead, which must be stated in the paper and
not waited for: **there is no published baseline to reproduce.** Every other dataset here was
gated against a released number (LoveDA −0.02, Potsdam +0.03, UAVid +2.16 and reported as
unexplained). DLRSD has none, so a preparation error cannot be caught by a gate. The
mitigations are the pixel-exact label accounting in §0 and the category fingerprint that
pinned the class ladder — **both are weaker than a reproduction gate, and the paper says so.**
