# Paper outline

**Purpose.** Fix the shape of the paper so every remaining experiment drops into a section instead
of becoming another loose results file. Start the LaTeX in Week 9 at the latest (`ROADMAP.md`
Phase 5); this document is what gets pasted into it.

**Rewritten 27 Aug** after OpenEarthMap. The previous version claimed *"labelable but not
detectable"* — a LoveDA-only finding that OEM refutes. The claim below is narrower, better
supported, and holds on both datasets.

---

## 1. The claim

The project set out to build a corpus-level semantic co-occurrence prior. **That did not survive
measurement** (`WEEK3_RESULTS.md` §6: +0.2 over a neighbour vote, oracle +0.3). Nor did the
successor claim. What holds on two datasets is this:

> **A single confidence threshold is wrong for different classes in opposite directions. Fitting
> one threshold per class on ~200 labelled tiles — the same protocol SegEarth-OV3 already uses to
> tune its single τ, with no weights trained — gives **+1.18 ± 0.45 mIoU** on LoveDA over a
> reproduced published baseline, with land-cover classes gaining **8.30 IoU** in aggregate and the
> catch-all class untouched. `water`, whose precision/recall is 89.5 / 54.7, gains 6.78 alone at a
> fitted τ of 0.195 against a global 0.5. That captures 81% of an oracle bound of +1.46, and it
> requires calibration data from the evaluation distribution: fitting on LoveDA *train* instead
> gives −0.12, because the splits differ 2× in discard rate at identical background share.**

And, as the measurement study that motivates it:

> **Presence-gated SAM 3 pipelines assign a large fraction of real land-cover pixels to
> `background`, but that fraction, and whether it can be detected at all, is governed by the
> dataset's label design rather than by the model. Where `background` is a catch-all covering a
> third of the scene (LoveDA, 36.1%), 29.68% of real-class pixels are discarded and no signal
> distinguishes them from genuine background — nine tested, best AUC 0.622. Where the vocabulary
> covers the scene (OpenEarthMap, `background` 0.84%), only 3.78% is discarded and the runner-up
> class score separates them at AUC 0.913. Recovery is nevertheless not worthwhile in either
> regime: on LoveDA it yields +0.04 mIoU, and on OpenEarthMap its apparent +2.28 is 110%
> attributable to `background` ceasing to be over-predicted, with the real classes losing 2.11 IoU
> in aggregate.**

**So the paper has both a method and a measurement study**, and they share a mechanism: the
residual exists because a global threshold cannot serve classes with different precision–recall
profiles, and recovering it after the fact does not work because the pixels cannot be identified.
Fixing the threshold beats recovering the residual, which is the useful practical message.

**Secondary contributions**, each independently defensible:

1. **Presence gating's measured function inverts its stated one.** SegEarth-OV3 motivates it as
   preventing hallucination of absent classes; on LoveDA its largest effect is suppressing
   `background` (median `S_pres` **0.022** against 0.45–0.91 for real classes), so τ rather than
   the argmax governs background assignment. (`WEEK1_RESULTS.md` §9.2b)
2. **A null-model correction for co-occurrence analysis on segmentation masks.** PMI over class
   adjacency must use **boundary** marginals; the natural choice of **area** marginals inflates
   high-perimeter classes. This refuted our own published finding ("road is a hub") and generalises
   beyond this dataset. (`ANALYSIS.md` §4.3)
3. **Atomisation dominates the prior.** Connected components of the discard region reach an oracle
   ceiling of 72.8%; SLIC reaches **92.8%**. Twenty ceiling points, against 0.3 for a *perfect*
   co-occurrence matrix.
4. **Per-class thresholding is worth +1.46 mIoU and is unreachable without labels.** The oracle
   thresholds span 0.170–0.595, so one global τ is wrong for different classes in opposite
   directions; `water` alone gains 6.70 IoU at τ=0.170. But what sets the right threshold is
   per-class precision, and no label-free proxy we tested (confidence percentile, presence score,
   Otsu split) predicts it — all three score *below* the published τ on LoveDA. This bounds an
   entire family of trivial alternatives and explains the bound, and it generalises to any
   training-free pipeline that thresholds a per-class score. (`WEEK3_RESULTS.md` §9a)
5. **Oracle bounds throughout.** Every negative result is reported with what a perfect version of
   the missing component would buy, so the reader learns where the headroom is rather than only
   that we failed to reach it.

---

## 2. Title candidates

- *It's the Labels, Not the Model: Why Presence-Gated Segmentation Discards, and Why Recovery Doesn't Help*
- *Catch-All Classes Govern the Background Residual in Training-Free Remote-Sensing Segmentation*
- *Detectable but Not Worth Recovering: The Background Residual in SAM 3 Pipelines*

**Recommended:** the first. It states the mechanism, which is the contribution.

---

## 3. Section map — which measurement goes where

### 1. Introduction
- The residual on LoveDA: **29.68%**, 323,184,908 px, discard beating confusion **3:1**.
- Immediately: **and 3.78% on OpenEarthMap.** Set up the mechanism in the first page — the paper's
  value is the contrast, so do not spend two sections on LoveDA before revealing it.
- Contribution statement per §1.

### 2. Related work
- SegEarth-OV (CVPR 2025), SegEarth-OV3 (arXiv:2512.08730) — the baseline.
- ConInfer (arXiv:2603.29271) — nearest competitor: **visual** context, **patch** level,
  **per-scene**. ⚠️ Do not cite its +2.80 as evidence appearance solves detection; it does context
  modelling, not residual-class detection. An earlier draft of our notes overstated this.
- DenseCRF / CRF-as-RNN — the classical spatial-context answer and the baseline a reviewer will
  raise against neighbour propagation.

### 3. Setup
- LoveDA val 1669 tiles @ τ=0.5; **baseline reproduced 47.38 vs published 47.4** (Δ 0.02).
- OpenEarthMap val **384 of the official 500** (public Kaggle redistribution) @ τ=0.1, its own
  tuned threshold; baseline **44.19 against their published 42.9** (+1.29). State plainly that at
  77% of the split this is a sanity anchor, not a reproduction gate — the tile mix differs.
- Instrumentation and its validation gate: every instrumented LoveDA run reproduces **47.37 /
  29.68%** exactly. Say this — it is why the numbers can be trusted.

### 4. Anatomy of the residual (LoveDA)
- **Two mechanisms** (§7.7): 94.0% below τ; **6.0% is background winning the argmax at `conf ≥ τ`**
  — all water, 24 tiles, **unreachable by any threshold**.
- τ-sweep, **non-linear**: 0.5→0.3 costs 0.73 mIoU, 0.3→0.1 costs 4.81.
- Boundary-vs-interior: enrichment 1.16–1.40×, so the residual is region-shaped, not seam-shaped.
- Presence gating: correlate, **not** cause — refuted by counterfactual, r = +0.018 on recovery.
  **Include the refutation of our own hypothesis; it is the section reviewers trust.**

### 5. The mechanism — two datasets ⭐ *the paper's core*
Table 1 of `WEEK3_RESULTS.md` §7, in full: background share, discard rate, catastrophic tiles,
presence correlation, detection AUC, commitment rate. Every row moves together with the
`background` share. Then the explanation: a catch-all class gives SAM 3 a plausible answer
everywhere, so a strong runner-up carries no information.

**State explicitly that two LoveDA-only claims do not generalise** — the presence-collapse
correlation (−0.750 vs +0.094, 198 catastrophic tiles vs 0) and the 29.68% headline itself.

### 6. Method (what was built, and which parts are justified)
- Region atoms: SLIC, with the ceiling table.
- Labelling: neighbour vote + signed PMI mixture, β-swept.
- The prior: boundary marginals, Dirichlet smoothing, directedness via the row conditional.

### 6a. The method — per-class τ ⭐ *lead with this*
Protocol, the fit objective (land-cover mIoU, excluding the catch-all — and why: optimising full
mIoU lets the search buy the metric by repairing an over-predicted background), 5-fold results,
the calibration learning curve, and the train→val scope limit. `WEEK3_RESULTS.md` §9b.

### 7. Results
- **Table 1 — the method.** 5-fold per-class τ: LoveDA +1.18 ± 0.45 (land cover +8.30,
  background −0.01, water +6.78), against an oracle bound of +1.46 — 81% captured. OEM: land cover
  +12.45 but full mIoU flat, because its `background` sits at 17.13 IoU and pays for the gain.
  Calibration curve beside it: 200 tiles for a reliably positive draw.
- **Table 2 — interventions.** LoveDA: τ→0.1 −5.54 · presence off −11.97 · honest recovery −6.19 ·
  best abstention +0.04 · oracle +3.62. OEM: honest +2.28 · oracle +5.21.
- **Table 3 — the co-occurrence ablation.** β=0 (48.4%) vs best β (48.6%) vs oracle M (48.7%), plus
  the multi-neighbour stratum where it does work (+2.03 mined, +5.44 oracle, 10% of the residual).
- **Table 4 — detection across both datasets.** Nine signals, LoveDA 0.434–0.622 against OEM
  0.601–0.913, with base rates and the ~0.53 empirical floor.
- **Table 5 — threshold tuning, bounded.** Published τ · best global τ · best per-class τ
  (oracle) · three label-free rules. LoveDA: the oracle is worth **+1.46** with real classes
  **+8.63**, and **every label-free rule is worse than the published τ**. The reason is the
  finding: the oracle exploits per-class *precision*, which is label-derived by definition.
- **Table 6 — the per-class decomposition of OEM's +2.28.** `background +22.67`, real classes
  **−2.11**, `building −3.75`. **This table is the honesty of the paper.** It shows a headline mIoU
  gain that is not a segmentation improvement, and it is the one a reviewer would otherwise
  construct themselves.

### 8. Limitations — write this honestly; it distinguishes the work
- Two datasets, and OEM at 384/500 tiles.
- Oracle bounds are upper bounds *under this labeller*, not true ceilings.
- Deep features untested for detection — justified, since detection already works on OEM and
  recovery still fails there.
- Recovery evaluated at one τ per dataset (each dataset's own tuned value), though
  §9a bounds what any other τ could have given.

### 9. Conclusion
The residual is a property of the annotation scheme as much as of the model. Detectability follows
the same axis. Neither makes recovery worthwhile — and the per-class table shows why a positive
mIoU number here would have been misleading.

---

## 4. Figures — reviewers read these first, budget real time

| # | figure | status |
|---|---|---|
| 1 | image / GT / baseline / discard mask, 4 tiles | ✅ `docs/25{22,24,25,27}.png` |
| 2 | **the mechanism**: six panels, `background` share as cause | ✅ `docs/fig2_mechanism.{png,pdf}` · `scripts/fig_mechanism.py` |
| 3 | per-class IoU delta on OEM (background +22.67 vs real classes −2.11) | ✅ `docs/fig3_oem_per_class.{png,pdf}` |
| 4 | detection AUC across signals, both datasets, with the 0.53 floor | ✅ `docs/fig4_detection_auc.{png,pdf}` |
| 5 | atom purity distribution + oracle ceiling, cc vs SLIC | ✅ `docs/fig5_atom_purity.{png,pdf}` |
| 6 | GT co-occurrence heatmap (`PMI_bnd`) | ✅ from `cooccurrence_gt.py` |

Figures 2–5 are rendered by `scripts/fig_mechanism.py` and `scripts/fig_results.py`, both of which print every plotted number against its `WEEK3_RESULTS.md` section on render. A figure that has drifted from its source table is worse than no figure.

---

## 5. Venue

| venue | fit | note |
|---|---|---|
| **CVPR EarthVision workshop** | **best** | measurement/analysis welcome; 8 pages; audience knows SegEarth-OV3 |
| **IEEE GRSL** | good | 5-page letter suits one focused mechanism; two datasets is acceptable |
| IEEE TGRS / JSTARS | reach | expects a method and broader evaluation |
| CVPR/ICCV main | not realistic | do not spend the calendar on this |

---

## 6. What is missing before submission

1. **Third dataset** (Potsdam or Vaihingen). The mechanism predicts the residual's size
   from the `background` share alone — a third point tests that prediction directly, which is a
   better use of GPU time than another method attempt.
2. A ConInfer comparison row if the code runs (`github.com/Dog-Yang/ConInfer`).

---

## 7. Risks

**The contribution is a measurement, not a method.** Mitigation: lead with the mechanism and the
oracle bounds, not with the failures. "Here is what governs the residual, here is exactly how much
is available, here is why taking it does not help" is a complete story. The four secondary
contributions stand on their own.

**Two datasets is thin.** Real, and cheap to improve — a third dataset is GPU hours, not ideas.
Do it before writing the results section so the tables are born multi-dataset.

**The OEM +2.28 will be misread if the per-class table is not adjacent to it.** Never print the
headline without Table 5 on the same page.
