# FreeTraining-OVSS

Final-year research project, IIITDM Kurnool. Goal is a publishable paper.

**Training-free, annotation-free open-vocabulary semantic segmentation for remote sensing.**
Contribution: recover low-confidence pixels that presence-gated SAM 3 pipelines discard to
"background", using a semantic co-occurrence prior over region proposals.

Do **not** call this "unsupervised" — the vocabulary is given. See @ANALYSIS.md §3.6.

## Read these before proposing anything

- @ANALYSIS.md — problem framing, technical critique, measured PMI findings (§4). The
  empirical foundation. §4 numbers are load-bearing.
- @ROADMAP.md — 12-week plan, phase gates, reading list.
- @WEEK1_RESULTS.md — all measurements to date. Current state of the project.

## Baseline

SegEarth-OV3 (arXiv:2512.08730), our named baseline and closest competitor. Reproduced at
**47.38 mIoU** on LoveDA val vs their reported 47.4. This number is a gate — if any change
moves it, the change is wrong.

Second competitor: ConInfer (arXiv:2603.29271). Novelty claims must differentiate from it.

## Environment — do not "helpfully" upgrade anything

A three-way version deadlock. This is the **only** confirmed working combination; five other
approaches failed:

| | |
|---|---|
| Python | 3.11 (conda env `segov3`) |
| torch | 2.4.1+cu121 — SAM 3 needs ≥2.3, no mmcv wheels exist for 2.5 |
| mmcv | 2.2.0, prebuilt wheel from the torch2.4/cu121 index |
| mmsegmentation | 1.2.2 with `MMCV_MAX` patched to `'2.3.0'` in site-packages |

Never suggest `pip install -U`, mmcv-lite, or building mmcv from source (system nvcc is 13.3
against torch's 12.1 — it cannot work). Full rationale in @WEEK1_RESULTS.md §2.

Hardware: RTX 2000 Ada, 16 GB, power-capped at 70 W. A full LoveDA val pass is ~24 min and
that is the floor.

## Repo conventions

- **Work spans two machines**, same repo name on both, same git remote
  (`github.com/FasterThanAi/FreeTraining-OVSS`). Nothing syncs without push/pull.

  | | Linux workstation (lab) | Mac |
  |---|---|---|
  | repo | `~/FreeTraining-OVSS` | `~/FreeTraining-OVSS` |
  | GPU / LoveDA data | ✅ | ❌ |
  | `~/outputs/week2_tau{0.5,0.3,0.1}` | ✅ | ❌ |
  | `~/SegEarth-OV-3` (baseline clone), `~/sam3` | ✅ | ❌ |
  | role | **all measurement** | docs, analysis, planning |

  ⚠️ `~/final year pro/Final_year_project` on the workstation is a **dead folder** — no `.git`,
  abandoned. Older docs pointed at it; corrected 21 Aug. Do not use it.
  Push from HTTPS on the workstation needs a **GitHub Personal Access Token**, not a password.
- **No pipeline code lives here.** The SAM 1 + CLIP scaffold (`src/`, `pipeline.py`,
  `configs/config.yaml`, `tests/`, `requirements.txt`) was removed on 21 Aug — every design
  decision it encoded had been overturned by ANALYSIS §4, and its `requirements.txt` would have
  wrecked the pinned env. It is in git history if ever needed. The method will be built by
  forking `segearthov3_segmentor.py` in the SegEarth-OV-3 clone, per ROADMAP Phase 3.
- `SegEarth-OV-3/` ships a **vendored `sam3/`** that shadows the editable `~/sam3` install
  when running from inside it. Edits to `~/sam3` have no effect there.
- Datasets, checkpoints and venvs are gitignored. Never commit them. Checkpoint is symlinked
  from the HF cache.
- The GitHub **profile** README sometimes overwrites the project README on Mac-side commits.
  If `README.md` looks wrong: `git checkout HEAD -- README.md`.
- Long runs: `nohup ... > log 2>&1 &`. The trailing `&` is not optional — without it the
  shell blocks and a serial loop silently runs one job at a time.

## Settled design decisions — do not relitigate

These were decided by measurement, not preference. Rationale in the cited sections.

- **Adjacency = shared boundary length**, not centroid distance. ANALYSIS §4.
- **PMI uses BOUNDARY marginals, not area.** ⚠️ *Corrected 21 Aug.* `cooccurrence_gt.py`
  divides a boundary-frequency observation by an area-based expectation, which inflates
  high-perimeter classes. Correcting it moves per-pair values ~0.9 bits and flips 5 signs.
  **Quote `PMI_bnd` from `scripts/pmi_permutation_null.py`, never §4's area figures.**
  Premise restated: **0.574 bits vs a 0.003 permutation floor (216×)**, not 1.3–1.7.
- **Signed PMI, not raw counts.** ✅ Survives the correction — exclusion still carries the
  largest magnitudes (building–water −2.83, road–water −1.88). ANALYSIS §4.2.
- **Hierarchical M is required**: `M_eff = λ·M_global + (1−λ)·M_image`. ✅ Re-tested on the
  corrected marginal: 10/15 pairs flip sign urban vs rural (~6 substantively), mean |diff|
  0.664 bits. λ-sweep stays a mandatory ablation. ANALYSIS §4.4.
- ⛔ **"Road is a hub" is REFUTED** — it was measuring road's perimeter. Road's row collapses
  to ~0 under `PMI_bnd`. Discriminability weighting is still worth doing, but recompute
  `w(n) ∝ Var_c[PMI_bnd]` and note the real hub is **agriculture** (row var 0.04), matching
  WEEK1_RESULTS §8.1(c). ANALYSIS §4.3.
- **M is directed, not symmetric** — but *where* the direction lives was resolved on 25 Aug and
  the earlier phrasing was misleading. **Shared-boundary counts are symmetric by construction**:
  a water pixel touching an agricultural pixel is one boundary with no direction, so
  `M[i,j] == M[j,i]` always and no amount of counting changes it. §8.1(b)'s evidence
  (water→agricultural 19.3M with no reverse) comes from the **confusion** matrix, a different
  object. Directedness is real and enters through the **row-normalised conditional**
  `P(c | neighbour=n) = M[n,c] / Σ_c M[n,c]`, which is asymmetric because the class marginals
  differ (agricultural 44.7% of pixels vs water 18.3%). **`cond` in `M_global_*.npz` is the
  directed object; `counts` and `pmi_bnd` are symmetric and that is not a defect.**
- **PMI must be Dirichlet-smoothed (α>0) before the log.** An unobserved pair gives
  `log2(0) = −inf`; clamping that to 0.0 reports the *strongest* exclusion as
  "indistinguishable from chance". building–water is exactly such a pair, so the clamp would
  silently delete the most reliable fact in the matrix. Caught by a synthetic test, 25 Aug.
  `α=0` reproduces `cooccurrence_gt.py` exactly and is for reproduction checks only.
- **Presence gating is inherited from SegEarth-OV3 but is not free** — it vetoes whole tiles.
  WEEK1_RESULTS §9.2. (ANALYSIS §3.5 previously claimed otherwise; corrected.)

## Where the project stands

*Last rewritten 3 Sep. The dated blocks below carry the detail; this is the orientation.*

**The project has a method, a causally-established mechanism, and a drafted paper.** The
co-occurrence prior it was founded on is dead and stays dead (+0.2 over a neighbour vote, +0.3
with a *perfect* matrix — WEEK3 §6). Do not revive it.

**The claim, in one sentence:** a single confidence threshold is the wrong *shape* for a
multi-class open-vocabulary pipeline, and fitting one threshold per class — on the same
supervision budget the baseline already spends tuning its single τ, with no weights trained —
recovers a real gain.

| | SAM 3 (SegEarth-OV3) | ConInfer (CLIP) |
|---|---|---|
| 5-fold Δ mIoU | **+1.18 ± 0.45** (worst fold +0.80) | **+2.51 ± 0.34** (worst +2.22) |
| Δ excluding catch-all | +1.36 | **+1.94** |
| calibration tiles | ~200 | **~25** |

Confirmed end-to-end, not just as cache arithmetic: predicted 47.16 → 48.35 on 1469 held-out
tiles, `eval.py` measured 47.16 → 48.35, per-class within 0.04 (WEEK3 §9c). **The gain is larger
on the pipeline it was not developed for, and all seven ConInfer classes improve** — so the claim
is about *any* pipeline thresholding per-class scores, not about SAM 3. It **composes with**
ConInfer rather than beating it.

**Three things that are settled and must not be re-argued:**

1. **The mechanism is causal, and the lever is the label space — not the prompt list.**
   A 13-arm vocabulary intervention moved the catch-all's share 0.84% → 58.20% and detection
   0.794 → 0.582 against an arity-matched control at 0.710. Every arm that changes the vocabulary
   while leaving GT share alone moves ≤ 0.052; every arm that raises GT share moves 0.213–0.323.
   Effect saturates by ~35% share. (WEEK3 §7b/§7c)
2. **Share drives detectability; confusability does not.** Broken without a third dataset —
   LoveDA rural has higher share (42.9% vs 26.0%) but *lower* confusability (25.5% vs 43.3%) and
   *worse* detection (0.524 vs 0.730). (WEEK3 §7a)
3. **No label-free rule reaches the oracle, and the reason is coupling, not measurability.**
   Eleven attempts; best proxy +0.42 against a random control's p95 of +0.58. ⚠️ Precision **is**
   label-free (`mean_conf` ranks it at ρ +0.943, p 0.017) — the earlier "precision needs labels"
   argument is **false**. The real bound: ρ(precision, oracle τ) = −0.429, p = 0.419, so the right
   threshold solves a **coupled multi-class IoU objective** no per-class scalar can express.
   (WEEK3 §9d, §9f)

**Report both metrics, always.** Full mIoU is levered by the catch-all (1/N ≈ 14.3% LoveDA,
11.1% OEM). OEM's full **+0.30** is **+1.75** on the catch-all-excluded metric — *larger* than
LoveDA's +1.36. Quoting only full mIoU filed a positive dataset as flat for four days. (WEEK3 §9h)

**Known limits — state them, do not discover them again.** Per-class τ does **not** transfer
across a domain shift (rural +2.77 vs urban +0.10; mismatched calibration lands *below* the
published τ) or to ConInfer-on-OEM (−0.39). Calibration must match the evaluation distribution
(LoveDA train→val = −0.12). Half the rural/urban gap is unexplained — it is 85% `water`+`forest`,
but `water` has identical share and asymmetry in both domains and an 8× different gain.

**Motivation numbers, still load-bearing and still quoted in the paper:**

- **29.68%** of real-class pixels assigned to background at τ=0.5 (323M px) — ⚠️ a LoveDA-**val**
  figure, not a LoveDA figure, and not general: OEM discards 3.78%.
- τ→0.1 recovers ⅔ of it and costs **5.54 mIoU** — 1 right per 1.73 wrong.
- Discard outnumbers real-class confusion **3:1** — the dominant error is silence, not error.

**Status.** ROADMAP Phases 1–6 closed; Phase 7 running. Paper drafted (7,101 words, 26 refs,
1 `\todo`) with an Overleaf bundle script; ~2,000 words must **move** to supplementary, not
shrink. ConInfer's reproduction gate **failed** — report LoveDA with both numbers (published
39.33, ours 36.99), drop their OEM row. Potsdam pre-registered at 4.29% catch-all.
Target EarthVision 2027 (~March 2027, **unverified**); **content freeze 1 Jan 2027.**

### ⛔⭐ LEVER 4 IS A NULL TOO — and the four-lever family is CLOSED. @PRESENCE_POWER_RESULTS.md, 11 Sep.

`s_c = P_fused_c · S_pres_c^γ_c`, γ = 1 the published rule. LoveDA, **full 1669 tiles**, no GPU.
**E − C = +0.16 ± 0.36, 4/5 folds, mean−2sd −0.57.** ⛔ Null. ✅ Levers 1 and 2 reproduce almost
exactly on the same run (**+1.17 ± 0.44** against the recorded +1.18 ± 0.45), so the run is sound.

⭐⭐ **But the fitted γ is exact and predicted:** **Spearman(median `S_pres`, γ) = +0.886 over the
six real classes, exact p = 12/720 = 0.017.** Classes the gate barely touches want **more** of it;
the one it crushes wants **least**. ⭐ **`forest` — most-gated real class, worst class in the
dataset — takes γ 0.44, the lowest of any class, in every fold.** That was M2, the sharp
prediction, named before the run.

⭐⭐ **THE STOPPING POINT, pre-registered.** Four levers, one protocol, one budget, one gate:

| lever | changes | acts | result |
|---|---|---|---|
| **1** per-class τ | which label a score earns | **after** the argmax | ✅ **+1.18** |
| **2** per-class scale | which class wins | **at** the argmax | ✅ **+1.16** |
| 3 head fusion | the score itself | **before** the argmax | ⛔ +0.04 / +0.22 |
| 4 presence weight | the score itself | **before** the argmax | ⛔ **+0.16** |

⭐ **Everything that reshapes the DECISION works; everything that reshapes the EVIDENCE does not.**
⚠️ **Post-hoc over four levers, NOT a law** — the last post-hoc reading here ("the levers
substitute") was refuted within a week. What makes it a stopping point is that the decision side is
**exhausted by construction**: per-class τ is *provably* complete after a fixed argmax, and the only
thing left at the argmax is a general reordering, unbounded against 200 tiles. ⛔ **There is no
lever 5.**
⭐ **Both null levers diagnosed correctly before failing to help** — lever 3 recovered the
things/stuff duality unsupervised, lever 4 the presence ordering at p 0.017. The mechanisms are
real; levers 1–2 already reach that mass by another route. A bound with an explanation.

⚠️ **91 of 1669 tiles have `S_pres ≤ 0.001`**, where the ungate is clamped to a no-op — γ cannot act
on 5.5% of tiles. ⚠️ **Synonym approximation**: presence is applied per *query*, `spres` cached per
*class*, so `building,house` / `forest,tree` / `barren,bareland,soil` are approximate — and `forest`
is one. The **M5 correlation is unaffected** (a statement about the fit, not a deployment).
⚠️ `background`'s γ is **not identified** under `--objective real` (folds 2.0/0.2/2.0/0.2/2.0); do
not quote its row. ⚠️ **LoveDA only** — Potsdam is a CPU hour away if the null is questioned.
⛔ **A verdict-logic bug, the fifth of this exact kind and the first written in-session:** the
"stayed at γ = 1" band was ±0.15 against a **grid step of 0.2**, so it reported **0 of 7** where the
honest count is **3 of 7**. Band is now computed from the grid. Table right, prose wrong, again.

### ⛔ LEVER 3 (per-class head fusion) IS NOT ADOPTED — @HEAD_FUSION_RESULTS.md, 11 Sep.

`s_c = max(a_c·P_sem_c, b_c·P_inst_c)`, `max(a,b)=1`, `ρ_c = b_c/a_c`, **ρ = 1 is exactly the
published rule** so rung D nests rung C. Fitted on top of both existing levers.
**Potsdam +0.04 ± 0.11 (3/5), LoveDA +0.22 ± 0.21 (5/5, mean−2sd −0.19).** Neither clears the gate.
⭐ **The method stays two levers, which is a cleaner story than three.**

⭐⭐ **But the fitted ρ is the finding, and it was predicted in advance.** With no supervision about
which class is which, the fit recovers SegEarth-OV3's own things/stuff duality: Potsdam `car`
**2.33** (instance) vs `building` **0.53** (semantic); LoveDA `road` **3.83** vs `forest` **0.25**.
Potsdam keeps `max` for **4 of 6** classes, LoveDA for only **1 of 7** — the two datasets disagree
about how wrong the hardcoded rule is.

⛔ **Object scale does NOT decide which head a class wants — committed prediction, refuted.**
`building` takes **0.53 at 5 cm** and **0.51 at 30 cm**, a **1.04×** ratio across a 6× GSD
difference. `prereg/predict_head_fusion_loveda.md` predicted the opposite sign. No third
explanation is offered.

⛔⭐ **§4.5's reading is CORRECTED, and this project has been quoting it wrong since week one.**
§4.5 is cited throughout as *"`building` returns 14 instance masks, `road` returns 2"*, read as the
instance head failing on stuff. **The fit says `road` wants the instance head most (3.83) and
`building` least (0.51)** — and §4.5's own table has the answer in the column nobody quoted:
`road`'s **2** masks score **0.81–0.85**, `building`'s **14** score **0.51–0.77**. ⭐ **The fusion
follows the scores, which is what `max` reads; the counts are never seen downstream.** The
things/stuff duality stands; the inference from *fragment count* to *which head should win* does not.

✅ **The query→class collapse is validated** — LoveDA has 11 prompts for 7 classes
(`building,house`, `forest,tree`, `barren,bareland,soil`) and the identity gate is **0.00000**.
⚠️ LoveDA ran on an **800-tile `--sample`** (a full `--cache-heads` is ~97 GB), so levers 1 and 2
are noisier there (+0.79 ± 0.74, +1.05 ± 0.69) than their recorded +1.18 / +1.16. Quote the
increment.
⚠️ **L5 failed (1 of 7 classes at ρ ≈ 1, predicted ≥ 3), so the fit was checked before L1 was read**,
as the pre-registration requires: gate exact, ρ stable across folds (`forest` 0.25 ×5, `water`
1.00 ×5), levers 1–2 reproduce, unit tests return ρ = 1 on a null. The suspicion is discharged with
evidence; L5's failure is a finding about LoveDA, not a fault.

### ⭐⭐ URBAN IS REPAIRED — lever 2 works where lever 1 fails. @SUBSTITUTION_RESULTS.md, 11 Sep.

⛔ The standing caveat — *"the +1.18 is a RURAL result; urban is not distinguishable from zero"* —
**is now scoped to lever 1 only.** On the identical 677 urban tiles and identical folds:

| LoveDA urban | mean | sd | folds+ | gate (`mean−2sd>0` and 5/5) |
|---|---|---|---|---|
| lever 1 — per-class τ | +0.08 | 0.22 | 2/5 | ⛔ fail |
| ⭐ **lever 2 — per-class scale** | **+1.25** | **0.56** | **5/5** | ✅ **PASS** |

⭐ **Per-domain totals: rural +3.99, urban +1.33** — positive on both halves of LoveDA under the
project's own gate, and in **both** metrics (rural full +1.22 / excluded **+1.96**; urban +1.25 /
+1.37). ⚠️ Rural's catch-all pays −3.20, so rural's excluded figure is the larger one.
⚠️ **Still quote the breakdown** — the claim narrows from *"urban does not work"* to *"on urban the
gain comes from the second lever"*, it does not disappear.
✅ **§9e reproduced for free**: rural τ +2.77 ± 0.92, identical to §9e, through a different script,
cache and fold partition.
⭐ **`water` and `forest` carry both domains** — §9g's 85%-of-the-gap classes — and the fitted `w`
differs between domains on `forest` alone (**2.43 rural vs 1.33 urban, 1.83×**; six of seven other
classes agree within 25%), matching §9g's "opposite regimes" (recall 9.9 vs 68.9). ⛔ **`w` does
not transfer across domains either** — the calibrate-on-your-own-distribution rule now covers both
levers.
⚠️ **A cached-histogram prediction.** §9c's rule applies: verify end-to-end in the segmentor before
it is written up.
⚠️ **Random controls bound VARIANCE, not effect** — a random LoveDA subset genuinely has the gain,
so its mean is positive by construction. They show a stratum has ~half a random draw's fold spread
(0.53/0.56 vs 1.21/0.93) and that both controls fail the gate the strata pass. The urban claim
rests on the lever-vs-lever comparison on shared folds, not on them.
⚠️ ⛔ **`argmax_reorder.py`'s verdict prose is wrong on `--control` arms** — it calls an
intentionally random draw a "partial or mixed cache". Fourth occurrence of this failure mode
(WEEK3 §11). Tables correct, prose wrong; do not quote a control arm's verdict section.

### ⭐ The objective SEPARATES — proved 11 Sep. Read @SEPARABILITY_RESULTS.md.

**For a fixed argmax, a pixel predicted `c` either clears `τ_c` or becomes the catch-all — it can
never become another real class.** So `TP_c`, `FP_c`, `FN_c` depend on `τ_c` alone and the
**`real` objective is a sum of single-variable terms.** Measured at **exactly 0.000000** on all 19
class-sweeps across LoveDA, Potsdam and OEM.
⭐ **Coordinate ascent is therefore EXACT, not greedy** — one sweep per class is the global
optimum (`rounds=1 == rounds=6` everywhere), and the cost is `N×201` rather than `201^N`.
⛔ **It does NOT hold for `--objective all`**: the catch-all couples the vector, an explicit
counterexample exists, and Potsdam shows `max |Δτ| = 0.0050` under `all` against `0.0000` under
`real`. **LoveDA's 0.0000 under `all` is a coincidence of that dataset.**

⚠️ **§9d's stated mechanism is WRONG and must be restated.** It says raising one threshold
"changes every other class's optimum" — false for the real classes. ⭐ The correct version is
stronger: *the objective separates, so each threshold is the peak of that class's own IoU curve,
whose location depends on the full score distribution; a scalar summary such as precision does not
locate that peak.* This also explains why the fitted thresholds are **not monotone in precision**
(`road` 69.7% → 0.675, `barren` 50.9% → 0.375) — the curve's **shape** decides. Conclusion
unchanged.

⭐ **How much of the oracle the fit captures — put this in limitations.** Held-out real-class
mIoU: **LoveDA 83%** (47.46→48.70 of 48.96), **Potsdam 73%**, ⛔ **OpenEarthMap 11%**
(47.28→47.56 of 49.80). **OEM's shortfall is one class**: `water` fitted **0.710** against an
oracle **0.240**, IoU 69.18 → **55.02**, a 14.16 loss. Fix water alone and +0.28 becomes **+2.14**.
⚠️ **Not a sample-size problem** — OEM calibrates on 26% of its tiles against LoveDA's 12%.
⛔ **Do not re-download OEM's full 500-tile split**: it would not fix this and would invalidate
every recorded OEM number.

⛔ **The obvious fix FAILS — tested, negative.** Three selection rules on calibration data only:
plain argmax (deployed), inner-CV mean, and the one-standard-error rule (⭐ named as the
principled choice *before* the run). Δ vs published: LoveDA **+1.23 / +1.23 / +0.76**, Potsdam
**+0.90 / +1.00 / +0.55**, OEM **+0.28 / +0.15 / +0.54**. **1se is −0.19 worse than what is
already deployed; `cv` is −0.01.** Neither helps.
⭐ **Why:** 1se cannot distinguish *flat because uninformative* from *flat near a genuinely
displaced optimum*. On OEM it pinned six of eight classes to the published τ; on LoveDA it dragged
`water` 0.175 → 0.320 and lost 2.66 — **the one class that needed moving.**
⭐⭐ **What it reframes:** OEM `water` at the published 0.100 scores **69.18** against an oracle
69.84 — it never needed calibrating, while `road` had +3.59 available. **The open question is not
"pick a better threshold" but "which classes should be calibrated at all?"** ⚠️ That is a
per-class restatement of **§9f**, which returned a negative at dataset level on two datasets.
⛔ **Do not try a fourth selection rule** — shrinkage, a different SE multiplier, a
variance-weighted blend all founder on the same obstacle. "Do not look for a thirteenth statistic",
one level down.

### Review presentation and the per-class table — 9–10 Sep

**Two decks, both driven by `paper/numbers.tex` so no number is typed twice.**
`slides/main.tex` is the full version; **`slides/review.tex` is the one to use for the
mid-project review** — shorter, styled to the department's Boadilla template, and following the
committee's mandated eight sections. `slides/speaker_notes.tex` gives one entry per slide with the
lines to deliver and the likely question.
⭐ **Objectives are split into completed and in-progress deliberately** (reproduce / quantify /
design are done; scene-adaptive parameters, head fusion and wider transfer are open). ⛔ Do not
present the project as finished in a review deck.
⚠️ `scripts/make_overleaf_bundle.sh --slides` bundles the slides for Overleaf, flattening
`\input{../paper/numbers}` and `\bibliography{../paper/refs}` and refusing any surviving `../`.

⭐ **`scripts/pr_table.py` — per-class precision / recall / IoU under each rule.** Everything else
here is reported as IoU, which hides the mechanism. The LoveDA table reproduces the deployment run
exactly (47.16 → 48.35) and shows the argument directly: `water` 0.5 → **0.175** trading precision
89.7 → 86.3 for recall 54.0 → **63.2**; `road` 0.5 → **0.675** trading the other way (69.7 →
**73.4** / 70.6 → 66.3). **Four classes move down, two up, spread 0.175–0.675.**
⭐ **The ordering is NOT monotone in precision** — `barren` at 50.9 takes 0.375 while `road` at
69.7 takes 0.675 — reproducing §9d's coupling result independently, and it is the cleanest answer
to *"why not derive the threshold from precision?"*
⚠️ **`road` −0.54 and `agricultural` −0.25 are real per-class losses**, accepted because the mean
improved. Quote the table, never the mean alone.
⚠️ Rung C needs a `--cache-full` run: a scale changes the argmax and a histogram cache fixes it.

⚠️ **Two absolute LoveDA figures exist and must not be conflated.** The 5-fold total is
**+2.32** (+1.18 τ, +1.16 scale). The end-to-end verified deployment run is **47.65 → 49.02** on
1469 held-out tiles — **+1.37**, because lever 1 contributed only +0.03 on that particular
200-tile draw. Different protocols; quote them separately, with the spread.

### Demo, application figures and the venue — 7–8 Sep

**The paper now has an application section and two result figures, and the venue is settled.**

✅ **`scripts/demo_app.py`** — per tile: input / ground truth / baseline / calibrated / changed,
with per-class IoU, a pooled IoU across tiles, and the share of changed pixels that became
*correct*. Gradio if already importable, else a self-contained HTML page. ⛔ **It will not install
Gradio into `segov3`** — clone the env if you must, never modify it.
✅ **`scripts/fig_qualitative.py` → `docs/fig8_qualitative`** — four Potsdam tiles. ⚠️ **Row four
is a tile where the method LOSES 7.72 mIoU and it stays in the figure**; the five-fold is
+4.86 ± 0.35, an average, and four wins would misrepresent it. ⛔ Tiles whose gain is mostly the
catch-all (+8.65 → +2.01 excluded) are deliberately excluded.
✅ **`scripts/make_tiles.py`** — cuts OpenAerialMap GeoTIFFs into square tiles (SAM 3 distorts
non-square input) with a provenance manifest. `--source-meta` joins per-image OAM id, location,
GSD and licence, because renaming a download breaks the id-from-filename chain. ⚠️ The PIL path
refuses a decode above `--max-gib`; rasterio does windowed reads and recovers the GSD, in a
**separate** env.
✅ **`scripts/fig_vocabulary.py` → `docs/fig9_vocabulary`** — 25 tiles from 5 OAM UAV scenes over
India at 4–5 cm, two class lists typed at inference. ⭐ **The result is the STABILITY**: a flooded
building partitions to within half a point across a complete vocabulary change (`water` 78.6% vs
`flood water` 79.1%; `concrete roof` 18.2% vs `building` 18.3%). ⚠️ **The solar farm is a PARTIAL
detection — 100.0% one class under the settlement list, 4.6% `solar panel` when the word is
supplied — and is written up as one.** ⛔ **No ground truth exists for this imagery**: every
number is a per-class pixel share, never accuracy.
⚠️ **Vocabulary revisions are recorded in `configs/README_vocabularies.md`.** Only `_b` was ever
revised, once. ⛔ No further revisions — a missing class is reported as the model taking the
nearest available word.

⭐ **VENUE: IEEE TGRS.** Rolling, no deadline, and ROADMAP §7 named it plausible once §7.1
(ConInfer) and §7.2 (a third dataset) landed — both done. ⛔ **No CVPR build and no second paper
folder**: one source of truth, and `numbers.tex` is it. The EarthVision cut happens only if TGRS
rejects, when reviewer comments will say what to keep.

⛔ **A metric bug worth remembering, because it was in our favour.** The demo's per-tile mIoU used
`nanmean` per rung, so deleting a false-positive class dropped it from the denominator and reported
**+17.01 where the truth was +8.65**. That is §8.1's error — a metric moving without segmentation
quality moving — inside the tool built to demonstrate the fix for it. Both rungs now average over
the classes present in ground truth.

### Instrumentation complete — 21 Aug

`measure_discard_rate.py` is now in git (it never had been), alongside `reference/` which pins
the exact baseline code behind 47.38. The instrumented τ=0.5 run **passed the gate exactly**
(47.37 mIoU, 29.68% discard, every per-image statistic identical), so:

- per-class `S_pres` recorded for all 1669 tiles
- `.npz` cache written → any future τ or ablation is a numpy pass, not a 25-min GPU run

**Presence-head collapse: correlate, NOT cause.** Settled by counterfactual, WEEK1_RESULTS
§9.2a/§9.2b. The observation is real — catastrophic tiles (n=198) have median `spres_max`
**0.273** vs **0.918** healthy (n=77), r = **−0.750** over 1669 tiles. But `--no-presence`
refuted causation: recovery is **uncorrelated** with baseline presence (**+0.018**), and the
same intervention pushes healthy, barely-gated tiles from 0.46% to 54.11% discard, costing
**−11.97 mIoU**. **Do not put a causal claim about presence gating in the paper.** Tile 3487
shows the mechanism can occur; it does not generalise.

Three blunt interventions are now measured and all three fail — τ→0.1 (−5.54 mIoU, 1.73 wrong
per right), presence removal (−11.97, no net recovery), do nothing (29.68% discarded). **That is
the motivation section**: every global knob that reaches the residual costs more than it returns,
so recovery must be selective and semantic.

**Presence gating's real job is suppressing `background`** (WEEK1_RESULTS §9.2b). Median
`S_pres` is **0.022** for background against 0.45–0.91 for every real class — SAM 3 has nothing
to detect, background being LoveDA's catch-all rather than a visual concept. This inverts
SegEarth-OV3's stated motivation for gating, and unlike the §9.2b causal claim it is measured.

**Two discard mechanisms, not one** (WEEK1_RESULTS §7.7). Of the 323M background-assigned
pixels: **94.0% (A)** fell below τ, **6.0% (B)** had `conf ≥ τ` and background won the *argmax*.
All of (B) is **water**, 19,378,177 px in **24 tiles**. Two things follow:

- **(B) is unreachable by any τ** — lowering a threshold cannot change an argmax. Part of the
  residual lies outside what threshold tuning can address at all.
- **(B) is arbitration, not recovery** — exactly what signed PMI is for; (A) is recovery.

Say **"assigned to background"**, not "discarded by τ". ⚠️ Cached `conf` is **float16**, so
τ-boundary comparisons are off by ~0.03% (100,493 px). Use float32 before any fine τ sweep.

### Weeks 3 complete + OpenEarthMap replication — 27 Aug. Read @WEEK3_RESULTS.md first.

**⭐ The headline finding is now a two-dataset mechanism, not a LoveDA result.**

| | LoveDA (1669, τ=0.5) | OpenEarthMap (384, τ=0.1) |
|---|---|---|
| baseline mIoU | **47.38** (pub 47.4) | **44.19** (pub **42.9**) |
| `background` share of GT | **36.1%** catch-all | **0.84%** rare, genuine |
| real-class pixels discarded | 29.68% | **3.78%** |
| catastrophic tiles (≥99%) | 198 | **0** |
| corr(`spres_max`, discard) | −0.750 | **+0.094** |
| best detection AUC | 0.622 | **0.913** (`conf2`) |
| honest recovery Δ mIoU | +0.04 | +2.28 ⚠️ |

**Detectability is governed by label design, not by SAM 3.** A catch-all class gives the model a
plausible answer everywhere, so a strong runner-up carries no information.

⛔ **Recovery does not improve land-cover segmentation on either dataset.** OEM's +2.28 is **110%
`background` ceasing to be over-predicted** (+22.67) while real classes net **−2.11**, `building`
alone −3.75. **Never quote +2.28 without the per-class table.**

⭐ **The n=2 confound is BROKEN, and share wins.** `confound_split.py` stratifies LoveDA into
urban/rural, where share and confusability **dissociate**: rural 42.9% share / 25.5% confusability
/ AUC **0.524**; urban 26.0% / 43.3% / AUC **0.730**. Each hypothesis predicts a *different*
stratum to detect worse — **share is correct, confusability is wrong.** Prevalence destroys the
signal, not resemblance. Detection is monotone in share across four strata (OEM 0.84%→0.794,
urban 26.0%→0.730, pooled 36.1%→0.582, rural 42.9%→0.524). ⛔ **iSAID cannot break this confound**
— 97.11% share *and* maximally confusable; don't spend GPU time on it. ⚠️ Stratification, not
intervention. WEEK3 §7a.
⭐⭐ **THE MECHANISM IS CAUSAL — vocabulary intervention, 1 Sep. WEEK3 §7b.** §7a's concession
("stratification, not a randomised intervention") is **retired**. Catch-all share is set by the
*vocabulary*, so it was manipulated directly. **CPU-only and that is better, not just cheaper**:
every class is an independent forward pass with its own prompt and the only cross-class operation
is the `argmax`, so **dropping a class from the vocabulary == dropping its channel, exactly** —
all 13 arms read one `--cache-full` stack and differ in nothing but the vocabulary.
⚠️ **Score `det = max(AUC, 1−AUC)`, never raw AUC** — AUC is symmetric, so 0.208 is a 0.792
detector *inverted*, and the first verdict called that "causal" off a raw drop.
✅ Faithful arm (prompts dropped, pixels relabelled — LoveDA's real situation), share 0.84%→58.20%:
`conf` **0.794→0.582** vs control 0.710 (**2.5×**); `conf2` **0.913→0.590** vs control 0.855
(**5.6×**, 3.1× vs the *worst* control). It drives OEM into the 0.58–0.62 band where LoveDA sits.
⚠️ Quote **endpoints only** — non-monotone inside (0.507→0.624→0.582), Q2 fails at D25 (0.105 vs an
0.08 bar), and P5's line misses at 2 of 3 doses: **direction transfers, rate does not.**
⛔ The **max-merge** family (`A`/`C`) is a *flawed* design kept only because it was pre-registered —
it makes the catch-all a union of strong prompts, so `conf` **inverts** instead of degrading.
`PREREGISTRATION.md` holds all 12 predictions, committed before each run; 9 pass, 3 fail.

⛔ **Two LoveDA claims do NOT generalise and must be labelled as such:** the presence-collapse
correlation (§9.2/§9.2a) and the 29.68% headline itself.

✅ **THE METHOD — calibrated per-class τ.** Fit one threshold per class on ~200 labelled tiles,
evaluate on disjoint tiles. No weights trained; same protocol SegEarth-OV3 uses for its own τ.
**LoveDA +1.18 ± 0.45 mIoU, 5-fold, every fold positive, land cover +8.30, background −0.01** —
81% of the +1.46 oracle bound. `water` +6.78 alone at fitted τ 0.195 vs global 0.5.
✅ **Confirmed end-to-end 28 Aug** — `eval.py` reproduced the cached-histogram prediction
**exactly**: 47.16 → **48.35** on 1469 held-out tiles, every per-class Δ within 0.04. The
segmentor takes `prob_thd` as a scalar (unchanged, gate preserved) **or** a per-class vector.
This also validates the histogram as an instrument, so §9a/§9b's sweeps and oracle bounds inherit
it. ⛔ **⭐ The +1.18 is a RURAL result — `tau_domain.py`, WEEK3 §9e.** 5-fold *within* each LoveDA
domain: **rural +2.77 ± 0.92 (5/5 folds)**, **urban +0.10 ± 0.39 (2/5 folds)** — urban is not
distinguishable from zero. ⚠️ **Never quote +1.18 without this breakdown.** But land cover improves
in *both* (real classes +18.61 rural, **+4.18 urban**); urban's catch-all loses 3.51, and
(4.18−3.51)/7 = +0.10 is the whole flat result — **the OEM artefact of §8.1 with the sign
reversed**, confirming from the other side that full mIoU is a poor metric under a large catch-all.
⛔ **Thresholds are domain-specific and do NOT transfer**: mean |τ difference| 0.227, max 0.500
(road 0.725 rural vs 0.225 urban); the mismatched arm is **−0.40 rural / −1.11 urban, both below
the published τ** — the wrong domain's thresholds are worse than no calibration. ⚠️ **Pooling is
not safe either** (rural keeps 0.77 of 2.32) — a mixed calibration set fits one vector to two
optima, *the same failure as the global τ, one level up*. Rule: **calibrate on the distribution you
will evaluate on.**
⚠️ That deployment run is a **single fit on 200 tiles**, where `background` gains +0.85 (10%
of the total, incidental — the fit objective excludes it) and **`road` loses 0.53**. Quote the
5-fold **+1.18 ± 0.45** as the headline and this run as the verification; they are different
protocols and the match is coincidence. WEEK3 §9c.
⚠️ **Fit with `--objective real`** (excludes the catch-all): optimising full mIoU lets the search
buy the metric by repairing an over-predicted background, which is how OEM produced +5.80 with
land cover −1.77. ⚠️ **Calibration must match the evaluation distribution** — LoveDA train → val
gives −0.12. WEEK3 §9b.

✅ **VERIFIED END-TO-END ON POTSDAM, 6 Sep — 57.60 → 63.27.** Three `eval.py` passes,
1816 held-out tiles: A **57.60**, B (per-class τ) **58.35**, C (+scale) **63.27**. Every rung
within **0.05** of its cached prediction, every class within 0.23. **The scale is +4.92**
against the 5-fold's +4.86 ± 0.35. ⭐⭐ `tree` **37.92 → 59.09 (+21.17)** — the class whose
+54.7 precision–recall gap moved nothing under thresholds and weakened §9g in September, now
explained and confirmed in the pipeline.
⛔ **A config bug was found doing this and it is fixed:** the deploy templates hardcoded
`confidence_threshold=0.5`, which is SAM 3's *decoder* threshold and changes `seg_logits`
themselves. Generated configs therefore ran a different model from the baseline on any
dataset whose base config differs. `cfg_loveda.py` uses 0.5, so it was a no-op there and
**§9c's published verification is unaffected** — but it is in `tau_deploy.py` too, and any
future per-class τ deployment on another dataset would have been wrong. Both templates now
inherit from `_base_`. @ARGMAX_SCALING_RESULTS.md.

⭐⭐ **A SECOND LEVER — scaling before the argmax. @ARGMAX_SCALING_RESULTS.md, 5 Sep.**
`pred = argmax_c(w_c·s_c)` then threshold the RAW score, fitting `(w, τ)` together. This is the
family the completeness argument explicitly excludes — the 6.0% catch-all argmax wins and the
9.9% real-class confusion no threshold can reach. **LoveDA 5-fold: +1.16 ± 0.19 ON TOP of
per-class τ, 5/5 folds, mean−2sd = +0.78, total +2.32.** Every real class improves; catch-all
**−0.03**; catch-all-excluded +1.36. ✅ **Verified end-to-end**: predicted increment +1.31,
`eval.py` measured **+1.34**, per-class within 0.02. ⭐ **The budget does not rise** — 200 tiles
still, and the combined rule at 200 (+1.60) beats τ alone at 800 (+1.26). ⭐ The fit rediscovered
§7.7 unprompted: `background` → 0.41, `water` → 2.55, and §7.7 found every mechanism-(B) pixel is
water. ⚠️ Quote the **increment**, not a subsampled absolute (the subsample carries a ~0.28 offset
that cancels in the increment). ✅ **All four cells are now measured, and the two levers SUBSTITUTE:** LoveDA +1.16 τ / **+1.16** scale, Potsdam +0.75 / **+4.92**, ConInfer **+2.51** / **−0.10** (a clean null, pre-registered, C1 falsified). ⚠️ OpenEarthMap is **underpowered, not missing** — 384 tiles swing 10 points across folds against a ~1 mIoU effect, so it cannot measure either lever. ⛔ **The "levers substitute" reading is RETIRED — refuted 11 Sep, @SUBSTITUTION_RESULTS.md.** LoveDA's domains span **35×** in τ gain (rural +2.77, urban +0.08) and their scale gains are **+1.22 / +1.25** — a **+0.03** difference against se **0.35**, where substitution's own range is 5 points. A refutation from power. ⭐ **They are INDEPENDENT**: the scale gain is invariant to what the threshold collected, consistent with disjoint error mass (τ → the 94.0% below threshold; `w` → the 6.0% argmax-lost + 9.9% real-class confusion). ⚠️ Two strata of one dataset — same gate, needs a fourth.

⛔ **A TWELFTH label-free attempt, and it fails too — @REACHABILITY_RESULTS.md, 4 Sep.**
"Reachable" discard (below τ with a real-class argmax, so a threshold can touch it) vs
"unreachable" (the argmax already picked the catch-all). Label-free, and it looked like it
would resurrect §9f's criterion. It does not: over four live rows it scores ρ **+0.400**
against the discard rate's **+0.800** and the published τ's **+0.949** — **worse than the
statistic §9f already closed.** Potsdam's residual is **98.11% reachable** and gains only
+0.59; LoveDA `water` gains +6.78 on the *second-lowest* self-reachability. §9g's `tree`
anomaly is **not** repaired. ⭐ One real finding survives: **ConInfer's published OEM
threshold cannot fire** (`conf` floor 0.1042 vs `prob_thd` 0.1), so that −0.39 row is an
un-thresholded argmax and not a like-for-like transfer test.
⛔ **Do not look for a thirteenth statistic that fits five points.**

⛔ **⭐ The label-free bound is now CLOSED with a mechanism, and §9a's stated reason was wrong.**
`precision_proxy.py`: **precision IS predictable without labels** — `mean_conf` ρ +0.943 p 0.017,
cross-head `sem_inst_agree` ρ +0.886 p 0.033 (exact enumeration, 720 relabelings) — **and it buys
nothing**, best row +0.42 against a random-control p95 of +0.58 and a +1.24 oracle bound. Because
precision does not determine the threshold: ρ(precision, oracle τ) = −0.429, p 0.419, and the
relation is **non-monotone** (water 88.5→0.175, road 68.5→**0.675**, barren 55.6→0.375). **The real
bound: the optimal per-class τ solves a COUPLED multi-class IoU objective — raising one class's τ
moves pixels into `background` and changes every other class's optimum — so no per-class scalar can
express it, measurable or not.** That is why §9b's 6-parameter fit works and every 1-parameter rule
fails. **Eleven label-free attempts now, all bounded and explained.** WEEK3 §9d.
⛔ **Threshold tuning is closed for LABEL-FREE rules — but not for the reason previously recorded.** Sweeping one
*global* τ finds nothing (+0.04 LoveDA), but **per-class τ is worth +1.46 with real classes
+8.63** — `water` at τ=0.170 alone gains 6.70 IoU, and the oracle thresholds span 0.170–0.595.
**No label-free rule reaches it**: Otsu −0.17, presence-scaled −0.74, equal-commitment −2.98, all
*below* the published τ. The oracle exploits per-class **precision**, which is label-derived by
definition. WEEK3 §9a.

⛔ **Drop the DINOv3 *detection* plan** (labelling is a separate question — see below).** Detection already works on OEM (0.913) and recovery still
fails there, so better detection buys nothing.

### Week 3 detail — LoveDA

`M_global` was built, validated, and found **not to earn its place**. The headline shifted:

- **Labelling is solved.** A plain neighbour vote over SLIC atoms is worth **+3.62 mIoU** given
  an oracle that says which regions to touch.
- **Detection is the entire problem.** Nine signals — `conf` .582, `fconf` .559, `gap` .558,
  `spres_arg` .520, `fgap` .447, `spres_max` .434, region `mean_conf` .576, novelty .514/.528,
  texture **.622** — all at chance against a 43.1% base rate and a ~0.53 floor.
- **Honest recovery is +0.04 mIoU.** With an oracle detector, +3.62. Detection does not find more
  right answers (61.3M vs 59.5M correct); it avoids wrong ones — **228M → 48M, 4.7×**.
- ⛔ **The co-occurrence prior adds +0.2 over a neighbour vote; a *perfect* matrix adds +0.3.**
  It works only where 4+ classes border the region (+2.03 mined, +5.44 oracle) — 10% of the
  residual. **It is an ablation row, not the thesis.**
- ✅ **Atomisation is settled and matters far more:** SLIC oracle ceiling **92.8%** vs connected
  components' 72.8%. Closes ROADMAP Week 8's open question.
- ✅ Gate 1 passes (ρ +0.757, circularity retired at −0.257); ⛔ Gate 2 fails **against ground
  truth too** — a perfect M would *reinforce* forest→agriculture.

⚠️ **Superseded numbers.** +3.47 and 48.4% were measured on connected-component atoms. Quote
**+3.62** and **56.1%**. ⚠️ **`selective_recovery_miou.py` once leaked GT** by scoping regions to
`gt >= 2`; fixed, and `--regions oracle` reproduces it deliberately as an upper bound only.

### Phase 6 — publication hardening. See @ROADMAP.md Phase 6 for the full plan.

Ranked by acceptance probability per hour. ⭐ **§6.1 is worth more than the rest combined**: catch-all
share is a variable *we control* (it is set by the prompt vocabulary), so intervene on it instead of
stratifying — merge OEM classes into `background` at increasing doses, **with a class-count control
that merges the same classes into each other**, and re-measure detection AUC. That upgrades §7 from
"stratification, not intervention" to an actual intervention, closing the weakest joint in the
strongest claim. ~3 GPU hours. **Commit the predicted ordering before the first run.**
Then, CPU-only: **§6.2** stratify by *discard rate* rather than domain — it is label-free, so if the
calibration gain tracks it the paper ends with a deployment criterion instead of a caveat; **§6.3**
report catch-all-excluded mIoU everywhere, now that the artefact is measured in *both* directions
(OEM +22.67 inflating, LoveDA-urban −3.51 deflating); **§6.4** the one label-free family §9d does
*not* bound — a **coupled** objective (maximise cross-head agreement over the whole τ vector) rather
than another per-class scalar, gated at the random control's +0.58.

⛔ **§6.2 answered, and it is a NEGATIVE — WEEK3 §9f.** There is **no label-free rule for *when*
calibration pays.** Stratifying by the label-free catch-all fraction (ρ +0.885 / +0.924 with the
labelled discard, so the proxy is sound): LoveDA is **U-shaped** — the most *reliable* gain is the
**lowest**-residual stratum (+2.13 ± 0.18) and the highest-residual one is worthless as guidance
(+3.22 ± **3.15**, worst fold **−1.22**); OEM is inconclusive with the random control moving nearly
as much. ρ = +0.400 and **−0.500** — opposite signs. ⚠️ Don't rescue it with a variance reading
either; that flips sign too.
⭐ **One bound explains both halves:** you cannot pick the thresholds without labels (§9a/§9d,
eleven attempts) *and* you cannot predict whether picking them will pay without labels (§9f, two
datasets). **The ~200-tile cost is irreducible — it buys the answer and the question together.**
⚠️ §9e's rural/urban gap is therefore **not** residual size. Likeliest cause is **class
composition** (the gain lives in `water`/`forest`, which have the large P−R asymmetries and which
rural has more of) — untested, and a cheap CPU check if it is worth closing.

⭐/⚠️ **§9e's rural/urban gap is HALF explained — WEEK3 §9g.** Decomposition (arithmetic, since
mIoU is an unweighted mean): **`water` 48% + `forest` 37% = 85% of the 2.68-point gap.** Across the
12 (domain, class) cells the **precision−recall GAP** ranks which classes move (ρ **+0.713**,
p 0.013) — a measured confirmation of the method's stated mechanism — while ⭐ **precision alone
explains nothing (+0.168, p 0.60)**, the same lesson as §9d arriving independently.
⛔ **But `water`, the largest contributor at 48%, is NOT explained**: share 11.6% vs 11.8%, P−R gap
+35.0 vs +34.4 — effectively identical — yet Δ IoU +10.15 vs +1.22, an **8× difference**. Roughly
half the gap still has no mechanism. **Say this before quoting the ρ.** `forest` *is* clean (recall
9.9 vs 68.9, discard 67% vs 12% — opposite regimes). Hypothesis for water, **unmeasured**: §9d's
coupling — in urban, lowering water's τ takes pixels from dense `building`/`road`.

⏳ **TIMELINE — there is no near deadline.** EarthVision 2027 runs with CVPR 2027; the date is **not
officially published** and past years put it in **early March 2027**, so plan on **~6 months** and
verify the CFP when it posts. Phase 6's "submit now" stop rule was calibrated for a two-week horizon
and **no longer applies**. See @ROADMAP.md Phase 7: ConInfer (§7.1, one week, its own conda env —
**never touch `segov3`**), datasets 3–4 (§7.2 — **ISPRS registration is the slow path, do it first**;
pre-register the prediction, then repeat the vocabulary intervention so the causal claim rests on two
datasets), the coupled label-free objective (§7.3, one CPU day, gated at +0.58), and a writing pass
(§7.4 — 75 em-dashes, one per 3.6 sentences). ⛔ **Content freeze 1 Jan 2027.** With §7.1–§7.2 done,
**IEEE TGRS becomes plausible** and it is rolling, so there is no deadline to miss.

⛔ **ConInfer work is governed by @CONINFER_RUNBOOK.md — read it before installing anything.**
`scripts/setup_coninfer.sh` **refuses to run while `segov3` is active**, creates a separate
`coninfer` env, and snapshots `segov3`'s package list; `--verify` diffs it afterwards. A matching
package list is *necessary, not sufficient* — the real gate is behavioural: `eval.py
./configs/cfg_loveda.py` must still give **47.38**. ⚠️ Reproduce *their* published number before
evaluating on our splits, and report the row in **both metrics**, since a gain concentrated in the
catch-all is exactly what §9h says to check for.

⭐ **ConInfer runs, reproduction IMPERFECT — see @CONINFER_RESULTS.md and @CONINFER_RUNBOOK.md.**
Published **39.33** LoveDA / **41.95** OEM; ours **36.99** / **29.90**. ⛔ **Drop OEM** (−12.05, and
we hold only 384 of 500 tiles); ✅ **use LoveDA** (−2.34 on the *full* official split) and **report
both numbers**, theirs and ours, so the reader sees the reproduction gap. Ruled out as causes:
label encoding, prompt order, image format, `feature_up` (setting it True changes *nothing*), and
alphabetical subset bias (73 cities, aachen→zanzibar). Stopped there deliberately. ⚠️ **Most of the gap is the BACKBONE** (CLIP ViT-B/16 @448 vs SAM 3 @1024); our
contribution is the +1.16 over SegEarth-OV3, and claiming the 11.54 would be §8.1's error in our
own table. They tune `prob_thd` too (0.8/0.3/0.1) and are ~10× faster at inference — report both.
⭐⭐ **§7.1a DONE — per-class τ TRANSFERS TO A CLIP BACKBONE.** Fitted at *their* τ=0.8 on the
ConInfer cache (both gates passed: instrumented run 36.99 exactly, `conf` in [0.16, 0.96]):
**+2.51 ± 0.34 five-fold, every fold positive (worst +2.22), catch-all-excluded +1.94** — *larger*
than SAM 3's +1.18/+1.36 — and **every one of the 7 classes improves**, which our SAM 3 result does
not manage. Calibration needs only **~25 tiles** here against ~200. ⭐ **The claim is no longer
about SAM 3**: it is a property of *any pipeline that thresholds per-class scores*, demonstrated on
two architectures, two methods and two operating points. ⭐ **And our method COMPOSES with the
nearest competitor** — their 36.99 → 39.52 — so we improve ConInfer rather than beat it.
⚠️ `background` +6.01 is a third of the full gain (not §8.1's 110%, and the excluded column is
independently positive) — quote both. ⚠️ It is a **delta on our reproduction**, not on their 39.33.
⚠️ `tau_cv.py`'s closing "train→val −0.12" line is SAM 3 boilerplate, never tested here.

Next, in order:

1. **Write.** @PAPER_OUTLINE.md has the skeleton, the section map and the figure list.
2. **One timeboxed DINOv3 detection attempt** — hard gate **AUC ≥ 0.70 size-controlled, or stop**.
   `appearance_detection.py` already takes a feature array. One week, not two.
   ⚠️ Do not cite ConInfer's +2.80 as evidence this will work; it does context modelling, not
   residual-class detection. An earlier note in this project overstated that.
3. **A second dataset — non-negotiable.** Every number is one split of one dataset. OpenEarthMap
   is the cheapest second point (~25 min/pass). One dataset is a desk reject.

## How to work on this

- **Measure before deciding.** Every open question in ROADMAP has a cheap experiment attached.
  Run it rather than arguing from intuition.
- **Validation gates are hard stops.** Instrumentation changes must still reproduce 47.37 mIoU
  and 29.68% discard. If they don't, the patch changed behaviour.
- **Cache SAM 3 outputs.** Downstream stages get re-run hundreds of times during ablations.
  Never pay encoder cost for arithmetic on a saved confidence map.
- **Seed everything, log every config.** Ablation tables assembled from unlabelled runs are how
  projects die in week 11.
- Keep a `LOGBOOK.md` entry per working day: what was tried, what broke, what the number was.
- Push back when the plan is wrong. A correction now is cheaper than a rewritten results
  section in week 11.