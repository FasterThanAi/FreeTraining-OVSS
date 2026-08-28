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

Weeks 1–3 complete. Baseline reproduced; premise confirmed and quantified; the proposed
contribution measured and **refuted**, with an oracle bound naming what blocks it (see the
Week 3 block at the end of this section):

- **29.68%** of real-class pixels discarded to background at τ=0.5 (323M pixels).
- τ-sweep: recovering ⅔ of that residual by lowering τ to 0.1 costs **5.54 mIoU**. Threshold
  relaxation buys 1 correct pixel per 1.73 wrong ones. This is the paper's motivation.
- Discard outnumbers real-class confusion **3:1** — the baseline's dominant error is silence,
  not error.

Roadmap phase gates 1, 3 and 5 are all cleared — roughly **three roadmap-weeks ahead of
schedule**. Phase 3 (Build) has not begun: no `M_global`, no RAG, no scoring function.

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
it. ⚠️ That deployment run is a **single fit on 200 tiles**, where `background` gains +0.85 (10%
of the total, incidental — the fit objective excludes it) and **`road` loses 0.53**. Quote the
5-fold **+1.18 ± 0.45** as the headline and this run as the verification; they are different
protocols and the match is coincidence. WEEK3 §9c.
⚠️ **Fit with `--objective real`** (excludes the catch-all): optimising full mIoU lets the search
buy the metric by repairing an over-predicted background, which is how OEM produced +5.80 with
land cover −1.77. ⚠️ **Calibration must match the evaluation distribution** — LoveDA train → val
gives −0.12. WEEK3 §9b.

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