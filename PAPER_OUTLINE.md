# Paper outline — draft skeleton

**Purpose.** Fix the shape of the paper now, so every remaining experiment drops into a section
instead of becoming another loose results file. Start the LaTeX in Week 9 at the latest
(`ROADMAP.md` Phase 5); this document is what gets pasted into it.

**Status:** skeleton with real numbers slotted in. Two gaps, both named in §7 below.

---

## 1. What the paper claims

The project set out to build a corpus-level semantic co-occurrence prior. **That claim did not
survive measurement** (`WEEK3_RESULTS.md` §6: +0.2 over a neighbour vote, oracle +0.3). What did
survive is a sharper and more useful claim:

> **Presence-gated SAM 3 pipelines assign 29.68% of real land-cover pixels to `background`. That
> residual is *labelable* — a plain neighbour vote over SLIC atoms recovers it at +3.62 mIoU given
> an oracle detector — but it is *not detectable*: nine signals spanning the model's confidence,
> presence and fused scores, and the image's own colour and texture, all sit at chance. The
> bottleneck is not deciding **what** a discarded region is, but **whether** it is anything at
> all.**

That is a negative result with an oracle bound and a named mechanism. It is publishable, it is
honest, and it tells the next person exactly where to dig.

**Secondary contributions**, each independently defensible:

1. **Presence gating's measured function inverts its stated one.** SegEarth-OV3 motivates it as
   preventing hallucination of absent classes. On LoveDA its single largest effect is suppressing
   `background` — median `S_pres` **0.022** against 0.45–0.91 for every real class — so τ, not the
   argmax, governs background assignment. (`WEEK1_RESULTS.md` §9.2b)
2. **A null-model correction for co-occurrence analysis on segmentation masks.** PMI over class
   adjacency must use **boundary** marginals; the natural choice of **area** marginals inflates
   high-perimeter classes. This refuted our own published finding ("road is a hub") and generalises
   beyond this dataset. (`ANALYSIS.md` §4.3, `scripts/pmi_permutation_null.py`)
3. **Atomisation dominates the prior.** Connected components of the discard region reach an oracle
   ceiling of 72.8%; SLIC reaches **92.8%**. Twenty ceiling points, against 0.3 for a perfect
   co-occurrence matrix. (`WEEK3_RESULTS.md` §4)

---

## 2. Title candidates

- *The Silent Majority: What Presence-Gated SAM 3 Discards in Remote-Sensing Segmentation*
- *Labelable but Not Detectable: On Recovering the Background Residual in Training-Free OVRSS*
- *Nine Ways Not to Recover a Segmentation Residual*  ← honest, memorable, risky for a thesis

**Recommended:** the second. It states the finding in the title, which is what a negative-result
paper must do to be read.

---

## 3. Section map — which measurement goes where

### 1. Introduction
- The residual: **29.68%** of real-class pixels, **323,184,908 px**. (`WEEK1_RESULTS` §7.1)
- **Discard beats confusion 3:1** — the baseline's dominant error is *silence*, not error. §7.6.
- Contribution statement per §1 above.

### 2. Related work
- SegEarth-OV (CVPR 2025), SegEarth-OV3 (arXiv:2512.08730) — the baseline.
- ConInfer (arXiv:2603.29271) — nearest competitor. Position carefully: **visual** context,
  **patch** level, **per-scene**. ⚠️ Do not cite its +2.80 as evidence appearance solves *our*
  detection problem; it does context modelling, not residual-class detection.
- DenseCRF / CRF-as-RNN — the classical "use spatial context" answer, and the baseline a reviewer
  will raise against any neighbour-propagation result.

### 3. Setup
- LoveDA val, 1669 tiles, 7 classes, τ=0.5.
- **Baseline reproduced: 47.38 vs the paper's 47.4** (Δ 0.02). `WEEK1_RESULTS` §5.
- Instrumentation and its validation gate: every instrumented run reproduces **47.37 / 29.68%**
  exactly. Say this — it is why the numbers can be trusted.

### 4. Anatomy of the residual
- **Two mechanisms** (§7.7): 94.0% fell below τ; **6.0% is background winning the argmax at
  `conf ≥ τ`** — all water, 24 tiles, **unreachable by any threshold**.
- τ-sweep and its **non-linear** cost: 0.5→0.3 costs 0.73 mIoU, 0.3→0.1 costs 4.81. §7.2.
- Boundary-vs-interior decomposition: enrichment only 1.16–1.40×, so the residual is
  region-shaped, not seam-shaped. §9.1a.
- Presence gating: correlate, **not** cause — refuted by counterfactual, r = +0.018 on recovery.
  §9.2b. **Include the refutation of our own hypothesis; it is the section reviewers trust.**

### 5. Method (what we built and why each piece is or is not justified)
- Region atoms: SLIC over the candidate mask. Ceiling table. §4.
- Labelling: neighbour vote + signed PMI mixture, β-swept.
- The co-occurrence prior: construction, boundary marginals, Dirichlet smoothing, directedness via
  the row conditional.

### 6. Results
- **Table 1 — interventions.** τ→0.1 −5.54 · presence off −11.97 · honest recovery −6.19 ·
  best abstention +0.04 · **oracle detector +3.62**.
- **Table 2 — the co-occurrence ablation.** β=0 (48.4%) vs best β (48.6%) vs oracle M (48.7%),
  plus the multi-neighbour stratum where it does work (+2.03 mined, +5.44 oracle, 10% of residual).
- **Table 3 — detection.** Nine signals, AUC 0.434–0.622, base rate 43.1%, floor ~0.53.
- **Figure — oracle vs honest decomposition.** 61.3M correct / 48.0M wrong against 59.5M correct /
  228.0M wrong. *Same correct count; 4.7× fewer wrong.* This figure carries the paper.

### 7. Limitations — write this honestly, it is the section that distinguishes the work
- One dataset (see §7 below).
- Oracle bound is an upper bound under *this* labeller, not the true ceiling.
- Deep features untested at time of writing (or: tested and reported, per the Week 4 gate).
- Recovery evaluated at one τ.

### 8. Conclusion
Labelable, not detectable. The residual is real and reachable in principle; nothing the pipeline
exposes says *which* pixels to reach for.

---

## 4. Figures — budget real time, reviewers read these first

| # | figure | status |
|---|---|---|
| 1 | image / GT / baseline / discard mask, 4 tiles | ✅ `docs/25{22,24,25,27}.png` |
| 2 | oracle-vs-honest recovery decomposition (bar) | ⛔ to make |
| 3 | detection AUC across nine signals, with the 0.53 floor and size-confound line | ⛔ to make |
| 4 | atom purity distribution, cc vs SLIC | ⛔ to make |
| 5 | τ-sweep: recovered pixels vs mIoU cost, three points | ⛔ to make |
| 6 | GT co-occurrence heatmap (`PMI_bnd`) | ✅ from `cooccurrence_gt.py` |

---

## 5. Venue

| venue | fit | note |
|---|---|---|
| **CVPR EarthVision workshop** | **best** | negative/analysis results are welcome; 8 pages; RS audience knows SegEarth-OV3 |
| **IEEE GRSL** | good | 5-page letter suits one focused finding; wants at least 2 datasets |
| IEEE TGRS / JSTARS | reach | expects a method and broad evaluation |
| CVPR/ICCV main | not realistic | do not spend the calendar on this |

---

## 6. What is missing before submission

1. **A second dataset — non-negotiable.** Every number is one split of one dataset. OpenEarthMap
   is the cheapest second point (~25 min/pass, reachable via SegEarth-OV's `dataset_prepare.md`).
   Three datasets would be comfortable; **one is a desk reject.** SegEarth-OV3 reports on 20 — we
   cannot match that and do not need to, but we cannot submit on one.
2. **The DINOv3 detection result**, pass or fail. Timeboxed one week, gate AUC ≥ 0.70.
3. Figures 2–5.
4. A ConInfer comparison row if the code runs (`github.com/Dog-Yang/ConInfer`).

---

## 7. Two honest risks

**The contribution is a negative result.** Mitigation: the oracle bound turns "it doesn't work"
into "here is exactly how much is available and exactly what blocks it", and the three secondary
contributions stand on their own. Lead with the bound, not the failure.

**One dataset.** This is the real threat and it is fixable with GPU hours, not ideas. Do
OpenEarthMap before writing the results section, so the tables are born multi-dataset instead of
being retrofitted.
