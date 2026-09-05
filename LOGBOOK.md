# Logbook

One entry per working day: **what was tried, what broke, what the number was**
(`ROADMAP.md`, "Weekly rhythm"). Newest first.

The point of this file is week 12. When the results section needs "which config gave 47.2", this
should answer it with a `grep` instead of an archaeology session.

> ⚠️ **Entries before 28 Aug were reconstructed from git history and the results files on
> 28 Aug 2026, not written contemporaneously.** They are accurate as to dates, commits and
> numbers, but they record what landed rather than what the day felt like. Every entry from
> 28 Aug onward is written the same day.

---

## 2026-09-04 (Fri) — a statistic for what a threshold can actually reach

**No measurement today — a script and a hypothesis, both written on the Mac.** The workstation
runs it.

**The observation.** Ordering every (pipeline, dataset) pair by the operating threshold it ships
with: ConInfer/LoveDA τ=0.8 **+2.51**, SAM3/LoveDA τ=0.5 **+1.18**, SAM3/Potsdam τ=0.1 **+0.60**,
SAM3/OEM τ=0.1 **+0.30**, ConInfer/OEM τ=0.1 **−0.09**. That is monotone across two architectures
and it has never been written down anywhere in this project.

**The mechanism it points at was measured in week one and then forgotten.** WEEK1 §7.7 split
catch-all assignment into (A) below the threshold and (B) argmax lost. Lowering a threshold cannot
change an argmax, so anything in (B) is outside what per-class τ can *ever* fix. At τ=0.5 the
threshold does most of the work; at τ=0.1 it barely does any, so most of what remains must be
argmax loss — which the method cannot touch by construction.

⭐ **The reason it matters: the statistic is label-free.** "Of the pixels the model sent to the
catch-all, what fraction were below the threshold" needs no ground truth. §9f went looking for
exactly this kind of rule, tested the **discard rate**, and closed it as a negative (U-shaped on
LoveDA, opposite sign on OEM). Reachable share is *not* the discard rate — so §9f killing one does
not kill the other, and the script checks that collinearity **first**, before anything else it
prints can be read.

⚠️ **The tautology risk, stated in the docstring rather than discovered later.** A higher τ
mechanically discards more *and* makes more of that discard threshold-driven, so part of the
ordering is arithmetic. The report prints ρ(published τ, gain) alongside, and prefers the
per-class evidence, which sits within one operating point and is free of it.

**Sharpened past §7.7 while writing it.** `reachable` is a *strict subset* of mechanism (A): a
pixel below the threshold whose argmax was already the catch-all is (A) and still untouchable. And
the per-class version has to be **self**-reachability — `pred == c`, not just `pred != bg` —
because a pixel of class c recovered as some other real class comes back wearing the wrong label
and does not help c's IoU.

**The prediction it exists to test:** self-reachability separates Potsdam's `tree` (P−R gap
**+54.7**, Δ IoU **+0.32** — nothing) from LoveDA's `water` (gap **+34.8**, Δ IoU **+6.78**) where
§9g's precision–recall gap does not. `tree` is currently an unexplained fact that *weakens* §9g's
ρ = +0.713.

**Verified before shipping**, since none of the five caches are on this machine: a synthetic cache
with seven hand-worked pixels, checking all seven per-class counts, five partition invariants
(`reach + unreach == to_bg`, `mech_A + mech_B == to_bg`, `reach ⊆ mech_A`, `self_reach ⊆ reach`),
and — the one that matters — that `to_bg` equals the background column of `confusion_at` **exactly**,
so the reachability columns and the gain describe the same pixels. All four verdict branches and
the <3-row guard exercised with shape-only fixtures. The toy also reproduces the target pattern in
miniature: its `water` is 100% reachable, **0% self-reachable**, has a +50 P−R gap and gains
**+0.00**.

⛔ **Not a result yet, and the verdict text says so in the winning branch.** Five points chosen
after the gains were known. The step that converts it into evidence is the Potsdam protocol:
commit the predicted Δ for Vaihingen from its reachable share alone, *then* run it.

**RAN IT — four rows, and two problems the run exposed.**

⛔ **A tie bug in my own Spearman.** `argsort(argsort(x))` hands tied values distinct ranks in
array order, so a **constant** vector gets ranks 0..n-1 and correlates with whatever it is paired
against. ConInfer/OEM has self-reachable = 0 for every class, and the first run reported
**ρ = +0.214** for it — that was the class ordering, nothing else. It also inflated the
`published τ` row, where two datasets tie at 0.1: **+1.000 became +0.949** under averaged ranks.
Fixed with tie-averaged ranks and a guard returning `nan` when either side is constant, so it
prints *undefined* rather than a number. Fifth time in this project that the tables were right and
the statistic printed beside them was not.

⛔ **ConInfer/OpenEarthMap's published threshold is INERT, and that is a real finding.**
Mechanism (A) is **exactly 0** — not one pixel scores below τ. `CONINFER_RESULTS.md` records that
cache's `conf` range as **[0.1042, 0.9195]** against a published `prob_thd` of **0.1**, so the
threshold sits below the score floor and never fires. Every catch-all assignment there is an
argmax loss. The row cannot carry a correlation (reachable share is 0 by construction, not by
measurement), and it was carrying the entire thing. The script now detects this, names such rows,
and reports every ρ twice — with and without them.

**Where that leaves the hypothesis, honestly:**

⛔ **The per-class half is REFUTED.** SAM3/LoveDA gives self-reachable ρ **−0.200** — `water`
gains **+6.78**, by far the most, on the *second-lowest* self-reachability (34.6%). Converting to
absolute headroom kills it too: discard × self-reachable is **10.4–13.3% of GT for all six
classes**, essentially flat. Self-reachability does not explain which classes move, and §9g's
precision–recall gap is not displaced.

⚠️ **The dataset-level half is UNIDENTIFIABLE at three live rows.** Reachable share orders the
gain (ρ +1.000) and beats the discard rate (+0.500), but the live rows are 91.4 / 88.7 / 86.6 —
a **five-point spread** standing in for a 2.35 mIoU range — and **published τ orders them
identically (+1.000)**. The tautology flagged in the docstring is total here, not partial.

⭐ **Potsdam is the discriminating row, and it did not run** (wrong cache path). It sits at
**τ = 0.1, the same as SAM3/OEM**, so τ cannot separate the two — while their gains differ
(+0.60 vs +0.16). If reachable share separates them, it separates from τ. One CPU pass decides it.

**RESULT — refuted, both halves. @REACHABILITY_RESULTS.md.** Potsdam ran (2016 tiles,
`~/outputs/potsdam/cache`) and killed it. Over four live rows, reachable share scores ρ
**+0.400** against the discard rate's **+0.800** and the published τ's **+0.949** — the
*worst* of the three, and worse than the statistic §9f already closed. Potsdam has the
highest reachable share (93.9%) and the third gain.

⭐ **The cleanest refutation is Potsdam alone:** its residual is **98.11% reachable**,
mechanism (B) is 1.25%, so a threshold can touch nearly all of it — and the gain is
**+0.59**. Maximum reachability, small gain, exactly backwards.

Per class it is backwards too: Potsdam `car` at **6.9%** self-reachable gains **+3.69**
while `tree` at 19.2% gains **+0.32**. `tree` was the case this was built to explain, and
it stays unexplained. §9g is not displaced.

⚠️ **The go/no-go criterion I wrote before the run was too loose, and the code was
stricter.** I said a Potsdam reachable share above 86.6% would break the τ confound — that
only checked the pairwise τ=0.1 comparison and ignored the *global* ordering. Potsdam came
in at 93.9%, above the stated bar, and the correlation **fell from +1.000 to +0.400**. The
script's verdict, which reads the whole ordering, called it right. Trust the coded
criterion over the prose one.

⭐ **One finding survives and it is worth having: ConInfer's published OEM threshold cannot
fire.** Mechanism (A) is exactly zero over 384 tiles — `conf` floor **0.1042** against
`prob_thd` **0.1**. Their OEM baseline is an un-thresholded argmax, so our §7.1a −0.39 row
was never a like-for-like transfer test, and the write-up now says so.

**Twelve label-free attempts, all bounded.** Stopping here — the Potsdam discipline applies:
no thirteenth statistic fitted to five points. Back to the paper.

---

## 2026-09-04, later — scaling before the argmax looks REAL, and my verdict said the opposite

**The experiment.** `argmax_reorder.py`: `pred = argmax_c(w_c · s_c)` then `keep if s_pred ≥
τ_pred`, fitting `(w, τ)` together on the same budget. Attacks the family the completeness
argument explicitly does **not** cover — the 6.0% catch-all argmax wins and the 9.9% real-class
confusion, neither reachable by any threshold. LoveDA, 500-tile `--cache-full`, subsample gate
passed (largest disagreement **0.061** against a 0.15 bar).

**C − B = +1.16 ± 0.97, 5/5 folds positive**, range +0.07 to +2.48. Roughly the size of the whole
per-class τ contribution again.

⛔ **My verdict function called this "buys nothing".** The positive branch required
`mean − 2·sd > 0`; the case *positive, every fold positive, spread too wide* had **no branch** and
fell through to the negative one. Sixth time in this project the table was right and the generated
prose was wrong, third time it was mine. Branch added, plus a fitted-scale stability diagnostic —
the first version could not distinguish "the fit is unstable" from "the fit is stable and the
evaluation folds are small", which need opposite responses.

⭐ **Four independent lines say it is probably real, not overfitting:**

1. **5/5 folds positive** — 1 in 32 under a sign test.
2. ⭐ **The fitted scales are stable across folds**: `water` **2.45 ± 0.08**, `background`
   **0.39 ± 0.01**, largest relative spread 13.8% (`road`). Stable fit, scattered gains — so the
   noise is in the evaluation folds (100 tiles each), not in the parameters.
3. ⭐⭐ **The fit rediscovered §7.7 without being told.** It scales `background` **down to 0.39**
   and `water` **up to 2.45** — and §7.7 measured, in week one, that *every* mechanism-(B) pixel
   (catch-all winning the argmax at `conf ≥ τ`) is **water**, 19.4M px in 24 tiles. Two
   independent measurements, the same fault.
4. **The gains land where every prior analysis pointed**: `water` **+4.06**, `forest` **+4.52** —
   §7.3's two deep-discard classes and §9g's 85% of the rural/urban gap. `forest` up to 1.45
   against `agricultural` at 1.03 is exactly §8.1's top confusion (forest→agricultural, 23.8M px).

⚠️ **Not established.** `mean − 2·sd = −0.78`, so it fails the bar this project uses elsewhere;
six extra parameters; and the 500-tile subset makes rung B itself noisy (**+0.85 ± 0.94** here,
against the published +1.18 ± 0.45, with fold 1 at −0.32). Next: the full 1669-tile `--cache-full`
(~24 GB, 128 GB free) for real error bars, then an end-to-end segmentor run as §9c did for τ.

---

## 2026-09-05 (Sat) — the scale is real, verified in the pipeline, and free

**@ARGMAX_SCALING_RESULTS.md.** Full 1669-tile `--cache-full` (41.7 GB), gate passed exactly
(47.37 / 29.68%). **C − B = +1.16 ± 0.19, 5/5 folds, mean−2sd = +0.78** — clears the bar. Total
over the published baseline **+2.32**.

✅ **End-to-end.** Three `eval.py` passes on 1469 held-out tiles. Predicted increment **+1.31**,
measured **+1.34**; per-class `background` +1.55/+1.55, `building` +0.09/+0.09, `agricultural`
+0.17/+0.17. The segmentor printed its scale vector, so the config demonstrably reached the model.

⚠️ **A subsampling offset I had to chase.** Rung 3's raw absolute predicted 49.27 against a
measured 49.02 — a 0.25 miss that looked like a failure. It is not: the subsample carries a
**+0.28** absolute offset (rung 2 exact 47.68 vs subsampled 47.96) which cancels in the
increment. `reorder_deploy.py` now reports the debiased absolute, which for this run gives 48.97
against 49.02. **The increment was always the trustworthy quantity and the script said so; the
level was not.**

⭐⭐ **The calibration budget does not rise.** 13 parameters instead of 6, and 200 tiles is still
where every draw turns positive (increment +0.94, worst +0.66; at 100 the worst is −0.21). And
the combined rule at **200** tiles (+1.60) beats thresholds alone at **800** (+1.26) — adding the
scale is worth more than quadrupling the labels.

⚠️ The deployment draw was unlucky for τ alone (+0.03 against the 5-fold's +1.16), which is why
the 5-fold is the headline and this run is the verification.

⛔ **LoveDA only.** OEM, Potsdam and ConInfer are untested, and §9e gives no reason to assume `w`
transfers across domains any better than τ did.

**OEM settling run — the subsample was NOT the cause, and OEM is the wrong test.**
`--subsample 150000` made the scales *less* stable (15.7% → **28.6%**) and the increment
weaker (+0.95 → **+0.40 ± 1.41**). Rung A — the published baseline, independent of the
subsample — swings **39.64 → 49.79** across folds, and per-class τ itself comes out
**+0.40 ± 1.34** with a fold at −1.87. **384 tiles over 5 folds is 77 evaluation tiles; the
baseline's fold variance is 10 points against an effect of 0.4.** OpenEarthMap cannot measure
either lever. A measurement limit, not a result.

⚠️ **And the two runs were never comparable.** The fold partition came from the same RNG
stream as the pixel subsampling, so raising `--subsample` reshuffled the folds. Fixed: the
partition is drawn first from its own stream and depends on `--seed` alone. Verified by two
subsample settings giving identical rungs A and B. **Two settings of one knob must differ in
that knob alone.**

⭐ **Potsdam is the transfer test, not OEM** — 2016 tiles (more than LoveDA), 403 per fold,
512² so `--cache-full` is ~6 GB and ~35 GPU-minutes. Bigger and cheaper.

---

## 2026-09-01 (Tue) — Phase 6 closes; ConInfer run, and the method transfers across backbones

**§6.1 vocabulary intervention — the mechanism is now causal.** Catch-all share is set by the
vocabulary, so it was manipulated directly instead of stratified. CPU-only, because every class is
an independent forward pass and the only cross-class op is the `argmax`, so dropping a class from
the vocabulary is *exactly* dropping its channel — 13 arms off one `--cache-full` stack, all
reading identical model outputs. Faithful arm (prompts dropped, pixels relabelled), share
0.84%→58.20%: `conf` **0.794→0.582** vs control 0.710 (2.5×), `conf2` **0.913→0.590** vs 0.855
(5.6×). §7a's "stratification, not a randomised intervention" concession is retired.

⚠️ **Two things I got wrong and had to fix.** The first verdict called it causal off a raw AUC drop
to 0.208 — but AUC is symmetric, so that is a 0.792 detector *inverted*, not a destroyed one.
Everything is scored on `det = max(AUC, 1−AUC)` now. And the pre-registered `A` arms merged
channels by max, which makes the catch-all a *union of well-detected prompts* — unnaturally
competent, the opposite of a real catch-all. That is why `conf` inverted. The `B` arms (drop the
prompt, relabel the pixels) are the faithful analogue and were added with their own predictions
committed first.

**§7c reverse arm — P8 fails, and the failure is worth more.** Removing `background` from LoveDA's
vocabulary moved detectability 0.592→**0.540**, not up. But it changes the *vocabulary* while
leaving the *label space* alone. Every arm across both datasets that does that moves ≤0.052; every
arm that raises GT share moves 0.213–0.323. **The lever is the label space, not the prompt list.**
Also: the effect saturates by ~35% share, which is why LoveDA's own dose arm is underpowered — and
means share cannot explain the urban/rural gap.

**§9f — no label-free rule for when calibration pays.** U-shaped on LoveDA (ρ +0.400), opposite
sign on OEM (−0.500), controls moving nearly as far. Unifies with §9d: one bound covers choosing
the thresholds *and* deciding whether to choose them.

**§9g — the domain gap is 85% `water`+`forest`.** P−R asymmetry ranks the movers (ρ +0.713,
p 0.013); precision alone does not (+0.168, p 0.60). ⛔ But `water` — 48% of the gap — has the same
share and same asymmetry in both domains and an 8× different gain. Half the gap still unexplained,
and the script now says so rather than claiming otherwise.

**⭐ §9h — OpenEarthMap turns positive.** Reported in both metrics: full **+0.30** decomposes into
**+1.56** from the eight real classes and **−1.26** from `background` alone. It was written up as
flat and filed as an artefact; on the catch-all-excluded metric it gains **+1.75**, more than
LoveDA's +1.36. **The paper gains a second positive dataset.** OEM's published baseline is also
depressed 3.38 points by one class at 17.13 IoU.

**Paper.** 7,101 words, 26 references, 1 `\todo`. Abstract/contributions/conclusion rewritten for
the causal framing; bibliography rebuilt (it had malformed author fields that would have rendered
as garbage); em-dashes 75→47 and 2-dash sentences 25→11. `make_overleaf_bundle.sh` flattens it for
upload. Two blocks marked SUPPLEMENTARY CANDIDATE — ~2,000 words must *move*, not shrink.

**ConInfer, start to finish — the afternoon.** The comparison row the paper cites without a
number. Environment first: their `requirements.txt` wants torch 2.7.1, for which no prebuilt mmcv
wheel exists, so the source build dies on `pkg_resources` and would then die again on nvcc 13.3 vs
torch 12.x — both recorded in WEEK1 §2 back in week one. Ran it on our own stack instead (torch
2.4.1 + mmcv 2.2.0 prebuilt), which is a **deviation that must be reported**.

Then a chain of things their repo does not ship: `segearth_segmentor.py` is **absent** though
`eval.py` imports it at line 4 (copied from upstream); `ftfy`/`regex`, `psutil` and `openpyxl` are
imported by packages that never declare them; `ConInfer_segmentor.py` **hardcodes the first
author's filesystem** (`/data/users/cwy/...`) for the DINOv3 repo and weights; and
`configs_baseline/` sets `gmm_temp=0` while `gmm_fitting` computes `1/temp` unconditionally →
`ZeroDivisionError`. DINOv3 SAT-493M is licence-gated (403 on fbaipublicfiles) until the form is
accepted. **Every one of these is a reason most people cite the paper without a number.**

⛔ **The reproduction gate FAILED, and that is the day's most important process outcome.** Their
published figures are **39.33 LoveDA / 41.95 OEM**; we get **36.99 / 29.90**. The split that
reproduces closely (−2.34) is the one we hold in **full**; the one that fails badly (−12.05) is the
one we hold **384 of 500** of. So: report LoveDA with *both* numbers, drop OEM entirely. Eliminated
as causes — label encoding (uint8 0–8, correct), prompt order (matches exactly), image format (RGB
8-bit 1000²), `feature_up` (setting it `True` changes **nothing**, identical to the digit), and
alphabetical subset bias (73 cities, aachen→zanzibar). Stopped there deliberately rather than
spend the timebox on what cannot be verified without the missing 116 tiles.

**⭐⭐ Then the experiment that mattered more than the comparison.** ConInfer thresholds per-class
scores at a global `prob_thd=0.8` — exactly what our method replaces. `coninfer_cache.py` wraps
their `predict()` **read-only** (mmseg leaves `seg_logits`/`gt_sem_seg` on each `SegDataSample`, so
**no edit to their source at all**) and writes our `.npz` format. Both gates passed: instrumented
run **36.99 exactly**, `conf` in **[0.1601, 0.9611]** so the [0,1] threshold grid is valid.

| | SAM 3 | **ConInfer (CLIP)** |
|---|---|---|
| five-fold Δ | +1.18 ± 0.45 | ⭐ **+2.51 ± 0.34** |
| worst fold | +0.84 | **+2.22** |
| Δ excl. catch-all | +1.36 | **+1.94** |
| calibration tiles | ~200 | ⭐ **~25** |

**Every one of the seven classes improves** — which our own SAM 3 result does not manage. Their
pipeline goes **37.00 → 39.52**.

⭐ **The claim stops being about SAM 3.** It is a property of any pipeline that thresholds
per-class scores: two architectures, two methods, two operating points, same correction — and
*larger* on the pipeline it was not developed for. ⭐ **And we compose with the nearest competitor
rather than beating it**, which is a better answer to "how is this different from ConInfer?" than
any margin would have been. Paper abstract, contribution 2, related work and a new
Table 6 rewritten accordingly.

⚠️ Honest notes kept beside the result: the deltas are a **delta on our reproduction**, their
catch-all gains **+6.01** (about a third of the full movement, and the excluded column is
independently positive), and `tau_cv.py`'s closing "train→val −0.12" line is **SAM 3 boilerplate**
that was never tested here.

**Bugs of mine, caught and fixed today:** the `MMCV_MAX` patch located mmseg *by importing mmseg*,
which is the very import it repairs, so it silently patched nothing — and the failure was hidden
behind `2>/dev/null || true`; the import check tested a narrower import than `eval.py` performs, so
it passed while `openpyxl` was still missing; and the path patcher rewrote *both* `WEEK_DIR` lines,
which would have pointed the `vith16plus` branch at `vitl16` weights — wrong only on a dataset we
never run, i.e. silently. **Two lessons worth keeping: never locate a package by importing it when
the import is what you are repairing, and a verification narrower than the thing it verifies will
pass and mean nothing.**

**Timeline reset.** EarthVision 2027 is ~March 2027 (**not officially published — verify**), so the
horizon is ~6 months, not weeks. ROADMAP Phase 7 written: ConInfer, datasets 3–4, the coupled
label-free objective. **Content freeze 1 Jan 2027.**

**ConInfer started.** `setup_coninfer.sh` refuses to run while `segov3` is active, builds a separate
env, and snapshots the package list. ⭐ **Behavioural gate re-run today, before any ConInfer
install: `eval.py ./configs/cfg_loveda.py` → mIoU 47.3700**, every per-class value within 0.02 of
record. That is the instrumented segmentor's documented figure (47.37, not the pristine 47.38), and
it is a dated proof that `segov3` was healthy at the moment the risky work began.

## 2026-08-28 (Fri) — the project gets a method: calibrated per-class τ, +1.18

**21 commits, 08:32–16:41.** The day per-class thresholds went from an untested idea to an
oracle bound to a cross-validated, end-to-end-confirmed method — and the day the argument for
why no label-free rule can match it was itself refuted and replaced with a better one.

### Meeting with supervisor

Explained the whole arc: 29.68% residual, the co-occurrence prior's failure, the detection wall.
Two suggestions came back:

| suggestion | disposition |
|---|---|
| "find which label is discarded most and improve that" | ✅ **Validated in a form not previously tested** — not by *recovering* the worst class, but by giving each class its **own threshold**. Worth **+1.46 mIoU**. See below. |
| "do some clustering" | ⛔ **Closed.** Detection already works on OEM (AUC 0.913) and recovery still fails there, so better features buy nothing. DINOv3 plan dropped. |

Also confirmed to him that the co-occurrence matrix is dead: **+0.2 mined, +0.3 with a perfect
ground-truth matrix.**

### Tried — figures 2–5 (`ca64d15`, `5ab62cb`)

`fig_mechanism.py` and `fig_results.py`. Six-panel mechanism figure across both datasets, plus
per-class decomposition, detection AUC, atom purity. Small multiples rather than a scatter,
deliberately — a scatter of "background share vs outcome" at n=2 asserts a trend two collinear
points cannot support.

**Broke:** two errors the figures surfaced that the tables had hidden.

- Fig 3 recomputed deltas from IoU rounded to 2dp and printed **+2.06** where the run reported
  **+2.05**. A figure that disagrees with its own table by 0.01 is uncheckable. Reported deltas
  are now carried explicitly.
- Fig 4's caption claimed "nine signals sit at or below the floor" on LoveDA. **Five do.**
  Corrected to *"every signal lands between 0.434 and 0.622, at most 0.09 above the floor"* —
  true, and the stronger statement.

### Tried — threshold tuning, oracle then rules (`191ff42`, `27778fe`, `ce6bed3`) ⭐

`tau_oracle.py`. Three rungs, each strictly more powerful. Both swept rows are **oracle bounds** —
they select thresholds on the evaluation labels.

| rung | free params | LoveDA | OEM |
|---|---|---|---|
| published τ | 0 | 47.37 | 44.16 |
| best global τ | 1 | 47.41 (+0.04) | 49.31 (+5.15) |
| **best per-class τ** | N−1 | **48.83 (+1.46)** | 49.44 (+5.28) |

**LoveDA's +1.46 is the only gain in this project where land cover genuinely improves.** Six real
classes **+8.63 IoU** in aggregate against background's +1.59. Driven almost entirely by `water` at
**τ = 0.170 → +6.70 IoU** (baseline precision/recall 89.5 / 54.7). Chosen thresholds span
**0.170–0.595**: one global value is wrong for different classes in *opposite* directions.

**OEM's +5.28 is the familiar artefact** — `background +49.22`, real classes **−1.71**, and 98% of
it comes from a single global change (0.1 → 0.025).

Then `tau_rules.py` — three label-free rules, each spending one knob or none:

| rule | knobs | LoveDA Δ | OEM Δ |
|---|---|---|---|
| per-class Otsu | 0 | **−0.17** | −5.80 |
| equal-commitment (q-th percentile) | 1 | **−2.98** | +3.53 |
| presence-scaled (τ_c ∝ `S_pres_c`) | 1 | **−0.74** | +4.77 |
| *oracle per-class* | N−1 | *+1.46* | *+5.28* |

**On LoveDA every label-free rule scores below the published τ.** On OEM the two that look good
capture the background artefact (real classes −1.90).

**Why — and this is the actual finding.** The oracle exploits per-class **precision**. `water` can
afford 0.170 because it is right 89.5% of the time it fires; `agricultural` needs 0.595 because it
is not. Across the six LoveDA classes the oracle threshold tracks the precision–recall gap at
**r = −0.618** (n=6, a direction rather than a law). Precision is label-derived by definition, and
confidence percentiles, presence scores and Otsu splits all describe how the *model is
distributed*, not how *often it is right*.

> Per-class thresholding is worth +1.46, and the quantity needed to set it is precisely the
> quantity a training-free method cannot have. **That bounds the whole family** and generalises to
> any training-free pipeline thresholding a per-class score.

> ⛔ **Superseded the same day, 16:32.** The first half of that reason is **false** — precision
> *is* predictable without labels (`mean_conf` ranks it at ρ +0.943). The bound survives for a
> better reason. See the 16:26–16:41 block below.

**Broke:** Otsu, silently. Between-class variance is **flat** across every split separating two
well-spaced modes, so `argmax` returns the leftmost tie — a threshold *below both modes*, which
keeps everything and defeats the point. Now takes the plateau midpoint; verified to split a
0.10/0.90 bimodal at 0.50.

**Also corrected the project's own record.** `CLAUDE.md` had said "threshold tuning is closed" —
asserted after sweeping only **one global τ**. The family was declared exhausted without testing
the variant that turned out to matter.

`GLOSSARY.md` added (τ, residual, background share, atom, purity, PMI, boundary vs area marginals,
base rate, oracle bound, supervision leak, validation gate).

### Tried — third dataset, pre-registered before inference (`c65d309` → `e115639`)

`predict_dataset.py` reads GT masks only — no model, no inference — and states what must hold
before the pipeline runs. A dataset that is merely measured adds a row; one whose behaviour is
**predicted then measured** tests whether the mechanism is a rule or a coincidence.

iSAID locked in: **97.11% background**, 458 masks, 3.22B labelled pixels. Prediction filed in
`docs/isaid_PREDICTION.md`: catch-all regime like LoveDA, so discard **higher than** LoveDA's
10.88% @ τ=0.1 and detection AUC **lower than** 0.622, near the ~0.53 floor. `S_pres(background)`
must **not** move — that is a property of SAM 3, not of the annotation scheme.

**Then found the confound, and it is load-bearing (`df2111b`).** Everything has been attributed to
the catch-all's **share** of GT. But across the only two datasets measured, share moves together
with **confusability**:

| | share | does the catch-all *look like* the real classes? |
|---|---|---|
| LoveDA | 36.1% | **high** — unlabelled roads, pavement, built structures |
| OpenEarthMap | 0.84% | **low** — rare genuinely-unlabelable leftovers |

n=2 cannot separate them. **iSAID discriminates**: 97.11% background, but that background is
visually *distinct* from ships and planes. **Share predicts detection fails; confusability predicts
it works.** Either outcome eliminates one explanation. My earlier "detection near the floor"
prediction was reasoning from the weaker hypothesis, and is now recorded as such.

**Broke — a real baseline issue (`1940d71`).** `labels.py` looked for the literal string
`background`. Potsdam and Vaihingen call the catch-all `clutter` and list it **last**, so on those
datasets it would have fallen back to mask value 1 and treated **`road`** as the catch-all —
wrong, and silent. Now checks an alias list. The check also surfaced that the segmentor takes
`bg_idx=0` as a constructor default and `cfg_potsdam.py` never overrides it, so on Potsdam the
**discard target is `road`** while the **catch-all is `clutter` at index 5**. On LoveDA and OEM the
two coincide at index 0, which is why nothing surfaced this until now. Not asserted as a defect in
the baseline without checking their published Potsdam number first.

**Broke — iSAID mask format (`9fd7745`).** iSAID ships semantic masks as
`*_instance_color_RGB.png`, where the class is a **colour**. Every script here reads integer class
maps, and PIL would hand them a 3-channel array numpy indexes without complaint — the same silent
corruption as `reduce_zero_label`, the class ladder and the hardcoded `.png`. `isaid_prepare.py`
now **aborts on one pixel of an unmapped colour**; unmapped colours would otherwise fall through to
class 0, inflating `background`, the single quantity the mechanism turns on. Palette read off the
data, not recalled.

### Tried — turning the bound into a method (`4c4c7a9`)

`tau_fit.py`: fit per-class τ on a disjoint **train** split, evaluate on **val**. The fairness
argument — SegEarth-OV3 tunes τ per dataset **with labels** (0.5 LoveDA, 0.1 OEM), so 6 parameters
on a held-out split is the same protocol with more parameters, and no weights are trained. Refuses
to run if any tile id appears in both caches, given how the earlier supervision leak happened.
Train mIoU printed beside val for every fitted row so overfitting is visible rather than assumed.

---

### Afternoon — 12:09–12:40: cross-validated, and the objective corrected (`820d8d0`, `87b053b`, `dcda688`)

**Tried:** not trusting one split. A single 50/50 partition gave **+1.44** with real classes +8.06,
and the fitted thresholds landed almost on the oracle's (water 0.195 vs 0.170, barren 0.375 vs
0.370, forest 0.445 vs 0.440). The generalisation gap was **positive (+0.52)** — six parameters on
835 tiles are not overfitting. But one split is one draw, so: k-fold for the spread, and a learning
curve for the label cost.

Cheap because confusion matrices are functions of a `(gt, pred, conf-bin)` histogram and
**histograms add** — each tile's own 7×7×200 histogram is built once (65 MB for LoveDA val) and
every split afterwards is a subset sum. Verified exact against histograms built directly from the
same subsets. The published-τ baseline is recomputed on the *same held-out tiles*, so both sides
are measured on identical pixels.

**Broke — the objective itself, on OpenEarthMap.** Five folds gave OEM **+5.80 mIoU** but real
classes **−1.77**, background **+53.93**. Background there sits at 17.13 IoU with 17.33% precision,
so ~54 points are available from one class, and a coordinate ascent maximising *full* mIoU trades
away road (−2.14) and building (−1.20) to collect them. **It optimised exactly what it was asked
for; the ask was wrong.** `--objective real` now excludes the catch-all from what the fit
maximises while still *reporting* full mIoU.

**The result (`WEEK3_RESULTS.md` §9b):**

| | LoveDA, 5-fold |
|---|---|
| **Δ mIoU** | **+1.18 ± 0.45**, every fold positive, worst **+0.84** |
| land cover | **+8.30 IoU** aggregate |
| catch-all | **−0.01** — untouched |
| `water` | **+6.78** at fitted τ **0.195** vs global 0.5 |
| share of the +1.46 oracle | **81%** |

Under the corrected objective OEM shows **land cover +12.45 with full mIoU flat** — background
pays for all of it. A caveat about the benchmark, not the method.

**Calibration cost: ~200 labelled tiles** for a reliably positive draw; **below 50 it actively
hurts.**

**Scope limit, and it cuts into the paper's mechanism.** Fitting on LoveDA *train* and evaluating
on *val* gives **−0.12**, because those splits differ **2.04× in discard rate (14.54% vs 29.68%)
at an essentially identical background share (35.8% vs 36.1%)**. Two consequences, both recorded:
calibration data must come from the evaluation distribution, and — more importantly —
**background share cannot be the sole driver of the residual, since holding it constant doubles
the residual.** That supports the *confusability* reading over the *share* reading, and it means
**29.68% is a LoveDA-val number, not a LoveDA number.**

### Afternoon — 14:45–16:26: end-to-end through the real pipeline (`0818fdc`, `e0ac9f9`) ⭐

**Tried:** proving the +1.18 is not just arithmetic on a cached histogram. The segmentor's
`prob_thd` now accepts a scalar (bit-identical to the published line, so the 47.37 gate is
preserved *by construction*) or a per-class vector indexed by the argmax class, with length checked
against `num_cls` — a misaligned vector would apply water's τ to forest and still print a plausible
mIoU. `verify_perclass_tau.py` checks equivalence on CPU over 20 random threshold vectors before
spending 25 GPU-minutes. `tau_deploy.py` fits, writes the held-out split, emits the config, and
prints the mIoU the GPU run must reproduce.

**Number:** fitted on **200 calibration tiles**, predicted **47.16 → 48.35** on the 1469 disjoint
tiles. `eval.py` measured **47.16 → 48.35**. Every per-class delta agrees to **≤ 0.04**, inside the
float16 cache noise called in advance. **The histogram is now validated as an instrument, not just
as arithmetic** — every τ sweep, oracle bound and cross-validation in §9a/§9b rests on it.

**Broke — a bin-edge bug the test found.** `confusion_at` built its bin edge with
`(tau * nbins).astype(int)`, which **truncates**: `29/200*200` is `28.999999999999996`, so it
scored a threshold one bin below the one it reported. Now `np.rint`. **No recorded number
changes** — 7 of 201 grid values are affected and none was ever chosen — but a *deployed* threshold
could have gone wrong, which is exactly what this step exists to catch.

**Two things recorded rather than smoothed over:** background gains **+0.85** here against −0.01 in
the 5-fold (10% of the total, not 110% as in the OEM artefact) — so §9b's "background untouched" is
a property of that protocol, not of the method. And **`road` LOSES 0.53**: the fit raised its
threshold on the calibration tiles and it did not transfer. **A per-class rule can hurt a per-class
result.** This run is one fit at n=200 on the learning curve (+0.79 ± 0.35), not the 5-fold
protocol; that it also lands at +1.18 is a coincidence of two protocols, and they are quoted
separately.

Also dropped the stale committed `measure_discard_rate.py.bak`, and routed `--tau` through
`set_prob_thd` so a vector from a config cannot survive a scalar override.

### Afternoon — 16:26–16:41: the impossibility argument refuted and rebuilt (`2e10b8f`, `c022e28`, `8e54fcf`) ⭐

**Tried:** eliminating the rule a reviewer would propose. §9a's argument rested on three rules that
all describe how the model's **scores are distributed** (Otsu, confidence percentile, presence).
None asks how often the model is **right**. Cross-head agreement does.

The signal was not in the cache — `fused = max(P_sem, P_inst_agg)` destroys the distinction — so
this needed instrumentation: the segmentor now carries `P_inst_agg` and `P_sem` separately
(`-inf` as the identity for `max`, so a head that never fires stays distinguishable from one
scoring 0.0), and the cache keeps each head's own top-1. `precision_proxy.py` tests eight
label-free proxies against three label-derived targets, screened by Spearman with an **exact
permutation p over all n! relabelings**, because at 6 classes |ρ| = 0.6 is not evidence.

**Broke — my own stated reason (`WEEK3_RESULTS.md` §9d).** `mean_conf` ranks the six LoveDA classes
by precision at **ρ +0.943, p 0.017** by exact enumeration of all 720 relabelings. **So precision
IS predictable without labels** — SAM 3 is rank-calibrated across classes, which the baseline does
not report. And it **buys nothing**: −0.18 mIoU, with no proxy beating the random control's p95 of
+0.58.

**The restated bound — and it is a real argument now, not a restatement:**

> The right per-class threshold solves a **coupled multi-class IoU objective.** Raising one class's
> τ moves pixels into background and changes every other class's optimum. **No per-class scalar
> can express that, measurable or not.**

The relation is not even monotone — water 88.5 → 0.175, road 68.5 → 0.675, barren 55.6 → 0.375 —
and **ρ(precision, oracle τ) = −0.429, p = 0.419**. This explains *why* the 6-parameter fit works
and every 1-parameter rule fails, rather than merely recording it.

Cross-head agreement is still pending its GPU re-run, but **the coupling argument predicts it fails
regardless of how good a precision estimate it is — written down before the number.**

### Numbers to remember from today

- ⭐ **the method: LoveDA +1.18 ± 0.45 mIoU, 5-fold, every fold positive, worst +0.84**
- land cover **+8.30**, catch-all **−0.01**, `water` **+6.78** at fitted τ **0.195**
- **81%** of the +1.46 oracle bound
- end-to-end: predicted **47.16 → 48.35**, measured **47.16 → 48.35**, per-class ≤ 0.04
- calibration cost **~200 tiles**; below 50 it hurts; train→val **−0.12**
- `mean_conf` ranks precision **ρ +0.943, p 0.017**; ρ(precision, oracle τ) **−0.429, p 0.419**
- OEM under the corrected objective: land cover **+12.45**, full mIoU flat
- iSAID background share: **97.11%**

### Open at end of day

1. **Cross-head agreement awaits its GPU re-run.** The coupling argument predicts it fails; the
   prediction is on record before the number.
2. **iSAID cannot run yet.** `ValidationData/val` holds 458 masks and **no images**; the DOTA-v1.0
   val tiles are a separate download. The pre-registration is locked in regardless.
3. **Share vs confusability** — still the paper's central risk, but no longer symmetric: today's
   train/val comparison (2.04× discard at identical background share) already **favours
   confusability**. iSAID settles it.
4. **`road` −0.53 in the deployed run** — a per-class rule can damage a per-class result. Needs a
   sentence in limitations, or a fix.
5. Potsdam's `bg_idx` / catch-all mismatch — flagged, not resolved.
6. `WEEK3_RESULTS.md` gives OEM's baseline as **44.19** (§7, §12) and **44.16** (§9a). Reconcile —
   44.19 is what is compared against the published 42.9.

---

## 2026-08-27 (Thu) — OpenEarthMap; the claim becomes a two-dataset mechanism

**11 commits.** The second dataset landed and changed the paper from a negative result into a
mechanism.

**Tried:** de-LoveDA-ing the entire codebase, then running OEM end to end.

`labels.py` is now the single source of class identity — names resolved from mmseg's
`dataset_meta` or the prompt file (a **line** is one class; commas separate synonyms, so
`building,house` is one class, not two), written into the `.npz` cache, read back by every
downstream script. **`background` is located by name, not by position**, because the whole project
is about pixels assigned to it and an off-by-one there would invalidate every number silently.
Regression-checked against pre-refactor values: `atom_quality` purity 0.807 / ceiling 80.6%,
`prior_ceiling` 3.5% reachable, `build_m_global` matrices `np.allclose`.

**Broke — five LoveDA assumptions, four of them silently:**

| assumption | what would have happened |
|---|---|
| `reduce_zero_label=True` | OEM's raw 0 **is** `background`; `valid = gt > 0` would have **deleted every background pixel** — the exact class this project is about. No crash. |
| class ladder hardcoded LoveDA **mask values** | `[2]` labelled "building" names **`grass`** on OEM. Clean table, wrong row labels. No crash. |
| `.png` hardcoded | crashed against a directory of `.tif` — the only loud one |
| `NCLS = 7` | "cannot reshape array of size 65 into shape (7,7)" |
| f-string followed by a parenthesised conditional | `TypeError: 'str' object is not callable`. Python had printed a `SyntaxWarning` and **I ignored it.** File now compiles clean under `-W error::SyntaxWarning`. |

Also: a run reported **"0 images"** and continued all the way to a report. An empty input should
never produce a report; it now `SystemExit`s with cwd, path existence and the layout mismatch.

**Broke — the filesystem.** `ln -s target dest` creates the link *inside* `dest` when `dest`
already exists, producing `images/val/val → images/val`. Use `ln -sfn`. The tell was a file count
of **385** where 384 was expected.

**Numbers:**

| | LoveDA | OpenEarthMap |
|---|---|---|
| tiles / τ | 1669 / 0.5 | 384 / 0.1 |
| baseline mIoU | 47.38 *(pub. 47.4)* | **44.19** *(pub. 42.9, +1.29)* |
| `background` share of GT | **36.1%** | **0.84%** |
| real-class pixels discarded | 29.68% | **3.78%** |
| catastrophic tiles (≥99%) | 198 | **0** |
| corr(`spres_max`, discard) | −0.750 | **+0.094** |
| best detection AUC | 0.622 (texture) | **0.913** (`conf2`) |
| region `mean_conf` AUC | 0.576 | 0.798 |
| detection base rate | 43.1% | 82.7% |
| SLIC oracle ceiling | 92.8% | 82.9% |
| honest recovery Δ mIoU | +0.04 | **+2.28** ⚠️ |

> **Presence-gated pipelines discard heavily into a catch-all class when the dataset provides one,
> and the residual is detectable exactly when it does not.**

**And the day's most important decision: refusing OEM's +2.28.** The per-class decomposition
(`113b3b4`) showed `background` **+22.67**, real classes **−2.11**, `building` **−3.75** —
damaging the baseline's best class. A background-assigned real-class pixel is *already wrong*, so
relabelling it either fixes it (27.5%) or leaves it wrong elsewhere, while `background` sheds 13.8M
false positives either way. Averaged over nine classes that reads as +2.28. **The method makes
land-cover classification worse.** Recorded as "never quote +2.28 without the per-class table".

**Two LoveDA claims relabelled as dataset-specific:** the presence-collapse correlation (−0.750,
198 catastrophic tiles) does **not** replicate on OEM, and **29.68% is not a general figure**.

---

## 2026-08-25 (Mon) — `M_global` built, validated, and found not to earn its place

**11 commits.** Week 3's method was built and killed in one day.

**Tried:** `build_m_global.py`, `validate_m_global.py`, `sweep_mining_tau.py`, `prior_ceiling.py`,
`selective_recovery_miou.py`, `recoverability_signal.py`, `atom_quality.py`, `atoms.py`.

**Numbers:**

- **Gate 1 initially failed** with background included — Spearman **−0.110**, 18/42 sign flips.
  Dropping background from both sides → **+0.311**; mining at τ=0.70 → **+0.757**, 6/30 flips, all
  on pairs GT calls ≈ chance. Circularity retired at **−0.257**.
- Mining τ sweep: **fidelity rises as coverage collapses** — ρ +0.418 at τ=0 (165% coverage) to
  **+0.757 at τ=0.70 (5% coverage)**. The prediction that threshold starvation was the problem was
  wrong: low-confidence pixels are noise, not sparse signal.
- **Gate 2 fails against ground truth too.** A *perfect* matrix would **reinforce** the baseline's
  top confusions (forest→agriculture +0.32, water→agriculture +0.25). Adjacency and confusability
  are the same signal.
- ⛔ **Ceiling test:** neighbour vote alone **48.4%**, best mined mixture 48.6%, **oracle GT matrix
  48.7%**. A perfect co-occurrence matrix adds **0.3 points**. Works only where 4+ classes border
  the region (+2.01 mined / +4.22 oracle) — **10% of the residual**.
- ⭐ **Atomisation settled:** connected components ceiling **72.8%**, SLIC **92.8%**. Pixels in
  unlabelable atoms 12.5% → 1.1%. The largest connected component was **an entire tile**.
- ⛔ **Detection:** nine signals, AUC **0.434–0.622** against a 43.1% base rate and a ~0.53 floor.
  `P_fused` before gating scores **0.559** — *worse* than the gated score, so presence gating is
  not hiding the signal.
- Selective recovery: honest **−6.19**, oracle **+3.62**, honest + best filters **+0.04**.
  Decomposition: oracle 61.3M correct / 48.0M wrong vs honest 59.5M / 228.0M — **same correct
  count, 4.7× fewer wrong.**

**Broke — a supervision leak in my own scoring script (`a0b39cb`).** The first version scoped
regions to `(gt >= 2) & (base == 1)`, only pixels GT says are real classes — handing the method
advance knowledge of where to look and immunity from damaging true background. It produced a
plausible, quotable **+3.47 that was not a result**. `--regions oracle` now reproduces it
deliberately, labelled as an upper bound.

**Broke — a hypothesis, instructively.** Per-class vote reliability on the oracle scope (building
86.8%, water 69.8%, forest 21.9%) **did not transfer** to the honest scope, where every class sits
at 35–43%. It was never a property of the class; it was a property of the oracle.

Added a hard **0.50 mIoU bar** to `selective_recovery_miou.py` so +0.04 cannot be reported as a win.

---

## 2026-08-23 (Sat) — housekeeping

One commit: trailing newlines in `measure_discard_rate.py`. No measurements.

---

## 2026-08-21 (Thu) — the correction day

**18 commits — the highest-value day in the project.** Three of my own claims died.

**Tried:** committing the empirical core, instrumenting it, then auditing every load-bearing claim.

- `measure_discard_rate.py` committed at last, with `reference/` pinning the exact baseline code
  behind 47.38. Until this, the project's empirical core existed on one untracked filesystem.
- Instrumented run **passed the validation gate exactly**: mIoU **47.37**, discard **323,184,908
  (29.68%)**, per-image mean/median/max **33.79 / 18.51 / 100.00** — every figure identical to the
  pre-instrumentation run. Patch is observation-only. `.npz` cache written for all 1669 tiles, so
  every future τ and ablation became a numpy pass instead of a 25-min GPU run.

**Broke — my own null model (`faf4f8e`, `c5791b9`).** `ANALYSIS §4`'s PMI compared a **boundary**
observation against an **area** marginal, systematically inflating high-perimeter classes.
Correcting it moves per-pair values ~0.9 bits and flips 5 signs.

- ⛔ **"Road is a hub" REFUTED** — it was measuring road's perimeter. `road–barren` +2.32 → **+0.17**,
  `road–forest` +1.96 → **+0.06**. The real hub is **agriculture** (row variance **0.04**).
- ✅ Premise **survives, restated**: mean |PMI_bnd| **0.574 bits vs a 0.003 permutation floor —
  216×** (not the published 1.3–1.7).
- ✅ **§4.4 survives** — hierarchical M still required: 10/15 sign flips urban vs rural, mean |diff|
  0.664 bits.
- ✅ `building–water` **−2.83** survives as the strongest constraint.

**Broke — my own causal claim (`e89bf24`).** The `--no-presence` counterfactual **refuted**
presence-head collapse as a *cause*: mIoU 47.37 → **35.39 (−11.97)**, and decisively
**corr(`spres_max`, recovery) = +0.018** over the 198 catastrophic tiles. If gating were
suppressing recoverable evidence, the lowest-presence tiles would recover most. Healthy tiles went
0.46% → **54.11%** discard. Presence gating is a **correlate, not a cause** — do not put a causal
claim in the paper.

**Found the real mechanism instead.** Median `S_pres(background)` = **0.022** against 0.45–0.91 for
every real class. SAM 3 essentially never detects `background` — it is LoveDA's catch-all, not a
visual concept. **This inverts SegEarth-OV3's stated motivation for gating**, and unlike the causal
claim it is measured.

**Broke — a terminology assumption (`81d8d73`, `b4ff65c`).** "Assigned to background" is **not** the
same set as "discarded by τ": **94.0%** fell below τ, **6.0% (19,378,177 px)** had `conf ≥ τ` and
background won the **argmax** — **all of it water, in 24 tiles**, and **unreachable by any τ**. 20
of those 24 are catastrophic tiles. Cached `conf` is float16, so τ-boundary counts differ by
100,493 px (0.031%) — use float32 before any fine sweep.

Also filled §7.2/§7.3 exactly at all three τ (per-class counts sum to exactly 323,184,908 — a
fourth independent consistency check) and committed `results/week2/tau{0.5,0.3,0.1}/`.

---

## 2026-08-20 (Wed) — diagnostics written up

**5 commits.** Discard-rate visualisations, confusion-matrix charts, class composition analysis,
presence-head diagnostic docs, `week2wrapup.md`. Documentation day; no new measurements.

---

## 2026-08-18 (Tue) — baseline reproduced ✅

**5 commits.** The gate that makes every later number meaningful.

**Tried:** `python eval.py ./configs/cfg_loveda.py` on 1669 LoveDA val tiles.

**Number: mIoU 47.38 against the paper's 47.4 (Δ 0.02).** 0.85 s/image, ~24 min wall, **6115 MB
peak** on a 16 GB RTX 2000 Ada at native 1024×1024. Mini run on 20 images gave 38.97 — filename-
ordered and unrepresentative.

**Broke — the environment, for most of a day.** Three constraints intersect at exactly one point:
SAM 3 needs torch ≥ 2.3; mmcv has prebuilt wheels only for torch 2.1–2.4; mmseg 1.2.2 asserts
`mmcv < 2.2.0`. Resolution: **torch 2.4.1+cu121 · mmcv 2.2.0 (prebuilt wheel) · mmseg 1.2.2 with
`MMCV_MAX` patched to 2.3.0.** `mmcv-lite` is unusable (mmseg eagerly imports compiled CUDA ops);
source builds fail (system nvcc 13.3 vs torch's 12.1). Captured in `scripts/setup_env.sh` — **do
not rebuild by hand.**

**The per-class table is the premise, visible in the baseline's own metrics:** water 89.5 precision
/ 54.7 recall (**+34.8**), background 56.9 / 69.4 (**−12.5**, the inverse), building −1.4 and road
−1.0 balanced. The weakness is confined to amorphous "stuff".

---

## 2026-08-12 (Wed) — the premise, measured

**7 commits.** ROADMAP, ANALYSIS, SETUP_SAM3 written; the co-occurrence premise tested before
anything was built on it.

**Tried:** `sam3_smoke_test.py --raw` (dumps `forward_grounding` outputs, wrapped in bfloat16
autocast), then `cooccurrence_gt.py` over LoveDA val GT masks.

**Numbers:**

- **Fragmentation asymmetry confirmed on our own data:** `building` → **14** instance masks
  (0.51–0.77); `road` → **2** (0.81–0.85) despite presence **0.957**. SAM 3 resolves countable
  "things" and collapses on amorphous "stuff" — SegEarth-OV3 Fig. 1, reproduced independently.
- **Co-occurrence premise holds:** mean |PMI| 1.3–1.7 bits against a **0.004** random control.
  *(⚠️ superseded 21 Aug — the null model was mismatched. Quote **0.574 vs 0.003** instead.)*
- Six of fifteen class pairs flip sign urban vs rural → hierarchical M required.

All required tensors reachable via `model.forward_grounding()`; the public `Sam3Processor` API
discards most of them. Tiles must be square — SAM 3 resizes to 1008×1008.

---

## 2026-08-11 (Tue) — project start

Initial commit. Environment diagnostics (`check_env.sh`, `diagnose_gpu.sh`).
