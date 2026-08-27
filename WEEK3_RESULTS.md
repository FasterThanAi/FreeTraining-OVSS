# Week 3 Results — `M_global`, atomisation, and the two-dataset mechanism

**Goal (ROADMAP milestone, week 7):** build the corpus-level co-occurrence prior and validate it
before the scoring function is built on it.

**Status:** 🟢 `M_global` built and validated · 🟢 atomisation settled (SLIC) · 🟢 **replicated on a
second dataset** · ⛔ the co-occurrence term does not earn its place · ⛔ **recovery does not
improve land-cover segmentation on either dataset**

**Date:** 2026-08-27 · LoveDA val 1669 tiles @ τ=0.5 · OpenEarthMap val 384 tiles @ τ=0.1

> ⚠️ **Supersedes the 2026-08-25 version**, whose region-level numbers were computed on
> connected-component atoms (oracle ceiling 72.8%) rather than SLIC (92.8%). **+3.47 and 48.4% are
> superseded by +3.62 and 56.1%.** Do not quote the old figures.

---

## 0. The result in one paragraph

The prior was built and it is accurate — mined from SAM 3's confident predictions it agrees with
the ground-truth matrix at Spearman **+0.757** — and it is nearly useless: against a plain
neighbour vote it adds **+0.2 points**, and a *perfect* matrix adds **+0.3**. Atomisation matters
far more: SLIC lifts the ceiling on any region-level method from 72.8% to **92.8%**. The deeper
finding came from adding a second dataset. **LoveDA and OpenEarthMap disagree about almost
everything, and one mechanism explains all of it: whether the dataset provides a catch-all
`background` class.** LoveDA's is 36.1% of pixels; OEM's is 0.84%. That single difference tracks
the residual size (29.68% vs 3.78%), the detectability (best AUC 0.622 vs 0.913), the presence-
collapse correlation (−0.750 vs +0.094) and the catastrophic tail (198 tiles vs 0). **But it does
not rescue the method.** On LoveDA recovery is worth +0.04 mIoU; on OEM it is worth +2.28 — of
which **110% is `background` ceasing to be over-predicted, while the real classes net −2.11.**

---

## 1. What was built

| script | purpose |
|---|---|
| `build_m_global.py` | mines the prior from confident predictions (`--source pred`, no labels) or GT |
| `validate_m_global.py` | gate 1 (circularity), gate 2 (does M predict the baseline's confusions) |
| `sweep_mining_tau.py` | at what τ should the prior be *mined*? |
| `prior_ceiling.py` | what can the prior recover, against trivial baselines |
| `selective_recovery_miou.py` | does recovery move **mIoU**, and **where does the gain come from** |
| `recoverability_signal.py` | can we detect *which* background pixels are recoverable? |
| `atom_quality.py` | purity, oracle ceiling, region-level detection, per atomisation |
| `appearance_detection.py` | the same question from the image rather than the scores |
| `atoms.py`, `labels.py` | shared atomisers and dataset-agnostic class handling |

**Four definitions settled** (all in `CLAUDE.md`):

- **Below-τ pixels are UNKNOWN, not background.** Counting them as background would build the
  prior out of the mass the project exists to recover.
- **Shared-boundary counts are symmetric by construction.** Direction enters through the
  row-normalised conditional `P(c | neighbour=n)`; measured asymmetry mean **0.115**, max **0.262**.
- **PMI must be Dirichlet-smoothed before the log.** `log2(0) = −inf` clamped to 0.0 reports the
  strongest exclusion as "chance"; `building–water` is exactly such a pair.
- **Class identity comes from the data, never a hardcoded list.** See §11.

---

## 2. The GT reference reproduces `ANALYSIS §4` exactly ✅

Different code path (the `.npz` cache, not the PNG masks), same split, `--drop background`:

| pair | published §4.1 | this run |
|---|---|---|
| building–water | −2.83 | **−2.83** |
| road–water | −1.88 | **−1.88** |
| barren–forest | −1.11 | **−1.11** |
| water–agriculture | +0.25 | **+0.25** |
| road–barren | +0.17 | **+0.17** |
| road–forest | +0.06 | **+0.06** |
| building–agriculture | −0.25 | **−0.25** |
| **mean \|PMI_bnd\|** | **0.574** | **0.574** |

Row variances match too (agriculture 0.039 vs 0.04, road 0.661 vs 0.66). **§4's foundation is
independently reproduced** — a reproducibility claim worth making in the paper.

---

## 3. Mining τ — purity beats coverage (LoveDA)

| mining τ | coverage of GT boundary | tiles w/ no boundary | ρ vs GT (real classes) |
|---|---|---|---|
| 0.00 | 165.6% | 19 (1.1%) | +0.418 |
| 0.10 | 102.5% | 127 (7.6%) | +0.389 |
| 0.30 | 41.8% | 331 (19.8%) | +0.643 |
| 0.50 | 19.6% | 481 (28.8%) | +0.704 |
| **0.70** | **5.0%** | 826 (49.5%) | **+0.757** |
| 0.80 | 1.2% | 1167 (69.9%) | +0.693 |

Fidelity rises as coverage collapses. The prediction that threshold starvation was the problem was
**wrong**: low-confidence pixels are not sparse signal, they are noise that corrupts the statistics.

**`background` is unmineable at any τ, including 0** — 2.7% of counted boundary at τ=0 against
40.8% in GT, because `P_final = P_fused · S_pres` with median `S_pres(background) = 0.022` loses
the argmax too. And losing it **rewires** the graph: `building–road` is −3.15 in GT (LoveDA labels
the pavement between them `background`) and flips to +0.86 once that intermediary is gone.

---

## 4. Atomisation — settled, and worth more than the prior ⭐

| atoms | count | median px | max px | mean purity | **oracle ceiling** |
|---|---|---|---|---|---|
| connected components | 41,952 | 179 | **1,048,576** | 0.728 | **72.8%** |
| **SLIC** | 506,064 | 1,551 | 60,740 | **0.928** | **92.8%** |

Pixels in unlabelable atoms (purity ≤ 0.50) fall from **12.5% → 1.1%**. The largest connected
component was *an entire tile*. **Closes ROADMAP Week 8's open question with a measurement, and is
worth 20 ceiling points against 0.3 for a perfect co-occurrence matrix.**

OEM, same procedure: 32,974 atoms, purity **0.829**, ceiling **82.9%**.

---

## 5. The gates

**Gate 1 — circularity: passes.** ρ = **+0.757** at mining τ=0.70, background dropped; 6/30 sign
flips, every one on a pair GT calls ≈chance (max |GT| among flips **0.39**). All four pairs GT
calls strong survive with the correct sign. `Spearman(row error, discard rate) = −0.257`, so error
does **not** concentrate on the discarded classes — `ANALYSIS §3.2`'s circularity concern is
**retired with a number**. ⚠️ Magnitudes inflated **4.22×**; never feed raw mined PMI into a
scoring function.

**Gate 2 — fails, and against ground truth too.** The **GT** matrix says `forest → agriculture`
**+0.32**, `water → agriculture` **+0.25**, `water → barren` **+0.39**. A perfect matrix would
*reinforce* the baseline's top confusions. Not a mining defect — forest and agriculture genuinely
touch; adjacency and confusability are the same signal.

---

## 6. ⛔ The co-occurrence term does not earn its place

| method | pixel accuracy |
|---|---|
| majority class (always `agriculture`) | 38.6% |
| **β=0.00 — pure neighbour vote, M off** | **48.4%** |
| β=0.25 — best mined mixture | 48.6% |
| **β=0.50 with the ORACLE GT matrix** | **48.7%** |
| β=1.00 — pure co-occurrence, no vote | 20.1% |

**A perfect matrix adds 0.3 points over copying the largest neighbour.** It works only where 4+
classes border the region (+2.03 mined, +5.44 oracle) — **10.0% of the residual**, diluting to
**+0.40 overall**. An ablation row, not a thesis.

---

## 7. ⭐ Two datasets, one mechanism

This is the section that matters.

| | **LoveDA** | **OpenEarthMap** |
|---|---|---|
| tiles / τ | 1669 / 0.5 | 384 / 0.1 |
| baseline mIoU | 47.38 | 44.19 |
| **`background` share of GT** | **36.1%** (catch-all) | **0.84%** (rare, genuine) |
| real-class pixels discarded | **29.68%** (10.88% @ τ=0.1) | **3.78%** |
| catastrophic tiles (≥99%) | 198 | **0** |
| corr(`spres_max`, discard) | **−0.750** | **+0.094** |
| best detection AUC | **0.622** (texture) | **0.913** (`conf2`) |
| region-level `mean_conf` AUC | 0.576 | **0.798** |
| detection base rate | 43.1% | 82.7% |
| pixels the model commits to | 55.7% | **95.9%** |
| honest recovery Δ mIoU | **+0.04** | **+2.28** ⚠️ see §8 |
| oracle recovery Δ mIoU | +3.62 | +5.21 |

**Every row is explained by one thing: whether the vocabulary covers the scene.**

- LoveDA's `background` absorbs a third of every image, so SAM 3 has a plausible answer everywhere
  and a strong runner-up score means nothing. `conf2` AUC **0.541**.
- OEM's `background` is a rare "unlabelled" marker, so a strong runner-up means a real class was
  suppressed. `conf2` AUC **0.913**.

> **Presence-gated pipelines discard heavily into a catch-all class when the dataset provides one,
> and the residual is detectable exactly when it does not.**

**Two LoveDA-specific claims must be relabelled.** §9.2/§9.2a's presence-collapse correlation
(−0.750, 198 catastrophic tiles) does **not** replicate: OEM shows +0.094 and zero catastrophic
tiles. It is a property of LoveDA's annotation scheme, not of SAM 3. Likewise **29.68% is not a
general figure** — the same pipeline discards 3.78% on OEM.

---

## 8. ⛔ Recovery does not improve land-cover segmentation on either dataset

`selective_recovery_miou.py`, SLIC atoms. LoveDA's runs pass the hard gate (recover nothing →
47.37 / 323,084,415).

| dataset | scope | recovered | precision | mIoU | Δ |
|---|---|---|---|---|---|
| LoveDA | honest | 5.7% | 40.0% | 47.41 | **+0.04** |
| LoveDA | oracle | 33.8% | 56.1% | 50.99 | +3.62 |
| OEM | honest | 98.3% | 27.5% | 46.45 | **+2.28** |
| OEM | oracle | 90.2% | 30.0% | 49.37 | +5.21 |

### 8.1 OEM's +2.28 is a metric artefact ⭐

| class | before | after | Δ |
|---|---|---|---|
| **background** | 17.13 | 39.80 | **+22.67** |
| **building** | 75.32 | 71.57 | **−3.75** |
| tree | 63.91 | 63.26 | −0.64 |
| road | 45.88 | 45.32 | −0.56 |
| bareland | 13.77 | 13.40 | −0.36 |
| cropland | 44.08 | 43.98 | −0.10 |
| pavement | 27.88 | 29.94 | +2.05 |
| water | 66.57 | 67.60 | +1.02 |
| grass | 42.92 | 43.15 | +0.23 |

**Real classes net −2.11 IoU. `background` alone gains +22.67 — 110% of the total.**

The mechanism: a background-assigned real-class pixel is *already wrong*, so relabelling it either
fixes it (27.5%) or leaves it wrong in a different class (72.5%) — while `background` sheds 13.8M
false positives either way. Averaging over nine classes converts that into +2.28. **The method
makes land-cover classification worse**; `building −3.75`, damaging the baseline's best class, is
the tell.

**Do not report +2.28 as a segmentation improvement.** Report it with this table beside it, as
evidence that mIoU can be moved by unwinding an over-prediction without recovering anything.

### 8.2 Selectivity is not the contribution either

On OEM the best row **abstains from nothing** — margin 0, purity 0, no size ceiling — and every
filter costs. On LoveDA the best row needs `max px = 500` and still reaches only +0.04. The
"calibrated abstention" framing holds on neither.

### 8.3 A supervision leak, found and fixed — do not let it recur

The first version scoped regions to `(gt >= 2) & (base == 1)` — only pixels GT says are real
classes — handing the method advance knowledge of where to look and immunity from damaging true
background, which is exactly the protection τ-relaxation does not get. It produced a plausible,
quotable **+3.47** that was not a result. `--regions oracle` reproduces it deliberately, labelled
as an upper bound.

---

## 9. ⛔ Detection — nine signals on LoveDA, at chance; seven on OEM, working

LoveDA, over 750,084,573 background-assigned pixels (positives 323,084,415, base rate 43.1%):

| signal | LoveDA | **OEM** |
|---|---|---|
| `conf` (= `P_final`) | 0.582 | **0.794** |
| `conf2` | 0.541 | **0.913** |
| `gap` | 0.558 | 0.601 |
| `fconf` (= `P_fused`, pre-gating) | 0.559 | **0.781** |
| `fgap` | 0.447 | **0.796** |
| `spres_max` | 0.434 | **0.703** |
| `spres_arg` | 0.520 | 0.681 |
| region `mean_conf` (SLIC) | 0.576 | **0.798** |
| novelty vs prototypes (colour) | 0.514 / 0.528 | — |
| gradient energy (texture) | **0.622** | — |

**Three LoveDA hypotheses died.** `τ_low` does not exist in `P_final` (best precision 43.3% against
a 43.1% base rate). Presence gating is not hiding the signal — `P_fused` scores *worse* than the
gated score. Novelty detection fails at 0.514/0.528 against a ~0.53 floor, consistent with its own
premise: a residual class has no compact region of feature space to be far from.

**A confound a negative control caught.** On random-colour images, where no signal can exist,
novelty scored **0.966** — an atom's mean colour has noise scaling as 1/√size, so distance features
partly measure atom size, which correlates with the label. Fixed with size-stratified AUC over
factor-2 bins, unit-checked (pure size-leak 0.503 → 0.512, genuine signal 0.922 → 0.926, noise
~0.53). The empirical floor is **~0.53**, and an `atom size` reference row is printed alongside.

---

## 10. Where the project stands

**Established.**
- Atomisation: SLIC, ceiling 92.8% (LoveDA) / 82.9% (OEM).
- `M_global` is accurate (ρ +0.757) and its circularity risk is retired (−0.257).
- **One mechanism — catch-all vs covering vocabulary — explains the residual's size, its
  detectability, the presence correlation and the catastrophic tail across two datasets.**

**Refuted, each with a number.**

| intervention | LoveDA | OEM |
|---|---|---|
| lower τ to 0.1 | −5.54, 1.73 wrong per right | — |
| remove presence gating | −11.97 | — |
| co-occurrence prior over neighbours | +0.2 (oracle +0.3) | — |
| honest recovery, no abstention | −6.19 | +2.28 (real classes **−2.11**) |
| honest recovery + best abstention | +0.04 | +0.63 |
| detect recoverability | 0.434–0.622 AUC | 0.601–0.913 AUC |
| **improve land-cover IoU by recovery** | **no** | **no** |

**The conclusion.** Detectability is governed by label design. Recovery is not worthwhile either
way — where detection is hard the residual cannot be found, and where it is easy the resulting
labels are not accurate enough to beat leaving them alone.

---

## 11. Engineering notes worth keeping

**Every LoveDA assumption that OEM exposed failed silently, not loudly.** `reduce_zero_label=False`
(OEM's raw 0 *is* background) would have made `valid = gt > 0` delete every background pixel.
`NCLS = 7` crashed — the only free one. The class ladder hardcoded LoveDA *mask values*, so `[2]`
labelled "building" would have named `grass` on OEM and produced a clean table with wrong row
labels. `.png` hardcoded against a directory of `.tif` crashed. Class identity now comes from the
data via `labels.py`, and `background` is located **by name**.

**A symlink loop damaged the dataset directory.** `ln -s target dest` creates the link *inside*
`dest` when `dest` already exists, producing `images/val/val → images/val`. Use `ln -sfn`. The
tell was a file count of 385 where 384 was expected.

---

## 12. Next

1. **Write.** See `PAPER_OUTLINE.md`.
2. **Look up SegEarth-OV3's published OpenEarthMap mIoU** and record it beside our 44.19. With 384
   of 500 tiles it is not a reproduction gate, but it is the only sanity anchor on this dataset.
3. **Drop the DINOv3 plan.** Detection already works on OEM (AUC 0.913) and recovery still fails
   there, so better detection is no longer the bottleneck and deep features have nothing to buy.
4. **Optional third dataset** (Potsdam or Vaihingen) — the mechanism in §7 predicts the residual's
   size from the `background` share alone. A third point would test that prediction directly, which
   is a stronger use of GPU time than another method attempt.

---

## 13. Artefacts

| file | what |
|---|---|
| `~/outputs/week3/M_global_{gt,gt_nobg,pred_t07}.npz` + `.md` | §2, §3 |
| `~/outputs/week3/mining_tau_sweep{,_high}.md` | §3 |
| `~/outputs/week3/validation_t07.md` | §5 |
| `~/outputs/week3/prior_ceiling.md` | §6 |
| `~/outputs/week3/selective_slic{,_oracle}.md`, `selective_filtered.md` | §8 |
| `~/outputs/week3/recoverability_signal{,_fused}.md`, `appearance_detection.md` | §9 |
| `~/outputs/week3/atoms_{cc,slic}.md` | §4 |
| `~/outputs/oem_tau0.1/` | all OEM: `discard_summary.md`, `recoverability.md`, `atoms_slic.md`, `M_pred.npz`, `selective{,_oracle}.md` |
| `~/outputs/week3_fused/cache/` | 1669 LoveDA `.npz` with `fconf`/`fpred` |
| `~/outputs/oem_tau0.1/cache/` | 384 OEM `.npz` |
