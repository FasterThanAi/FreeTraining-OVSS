# Week 3 Results — `M_global`, atomisation, and the detection wall

**Goal (ROADMAP milestone, week 7):** build the corpus-level co-occurrence prior and validate it
*before* the scoring function is built on top of it.

**Status:** 🟢 `M_global` built and validated · 🟢 atomisation settled (SLIC) ·
⛔ **the co-occurrence term does not earn its place** · ⛔ **the residual is labelable but not
detectable** — nine signals at chance against a **+3.62 mIoU oracle bound**

**Date:** 2026-08-27 · all figures 1669 LoveDA val tiles unless stated

> ⚠️ **This file supersedes the 2026-08-25 version.** Numbers there were computed on connected-
> component atoms, whose oracle ceiling is 72.8% against SLIC's 92.8% (§4). Specifically
> **+3.47 and 48.4% are superseded by +3.62 and 56.1%.** Do not quote the old figures.

---

## 0. One-paragraph summary

The prior was built, and it is accurate: mined from SAM 3's own confident predictions it agrees
with the ground-truth matrix at Spearman **+0.757**, and every pair GT calls strong survives with
the right sign. It is also nearly useless — against a plain neighbour vote it adds **+0.2 points**,
and a *perfect* matrix adds **+0.3**. Atomisation turned out to matter far more than the prior:
switching from connected components to SLIC lifts the ceiling on any region-level method from
72.8% to **92.8%**. With the right atoms and an oracle telling us which regions to touch,
recovery is worth **+3.62 mIoU**. Without that oracle it is worth **+0.04**. Nine detection
signals — six from SAM 3's scores, three from appearance — all sit at chance. **Labelling is
solved; detection is the entire problem.**

---

## 1. What was built

| script | purpose |
|---|---|
| `build_m_global.py` | mines the prior from confident predictions (`--source pred`, no labels) or GT (`--source gt`) |
| `validate_m_global.py` | gate 1 (circularity), gate 2 (does M predict the baseline's confusions) |
| `sweep_mining_tau.py` | at what τ should the prior be *mined*? |
| `prior_ceiling.py` | what can the prior actually recover, against trivial baselines |
| `selective_recovery_miou.py` | does recovery move **mIoU** — the metric the paper is judged on |
| `recoverability_signal.py` | can we detect *which* background pixels are recoverable? |
| `atom_quality.py` | purity, oracle ceiling and region-level detection, per atomisation |
| `appearance_detection.py` | the same detection question from the *image* rather than the scores |
| `atoms.py` | shared atomisers, so the region experiments cannot drift apart |

**Three definitions settled** (all now in `CLAUDE.md`):

- **Below-τ pixels are UNKNOWN, not background.** Counting them as background would build the
  prior out of the very mass the project exists to recover.
- **Shared-boundary counts are symmetric by construction.** "M is directed" came from the
  *confusion* matrix (§8.1b), a different object. Direction enters through the row-normalised
  conditional `P(c | neighbour=n)`, asymmetric because the class marginals differ — measured
  mean |P(c\|n) − P(n\|c)| = **0.115**, max **0.262**.
- **PMI must be Dirichlet-smoothed before the log.** An unobserved pair gives `log2(0) = −inf`;
  clamping to 0.0 reports the *strongest* exclusion as "chance". `building–water` is exactly such
  a pair. Caught by a synthetic test.

---

## 2. The GT reference reproduces `ANALYSIS §4` exactly ✅

A different code path (the `.npz` cache, not the PNG masks), same split, `--drop background`:

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

Row variances match too: agriculture 0.039 (published 0.04), road 0.661 (0.66), building 1.297
(1.30), water 1.685 (1.69). **§4's empirical foundation is independently reproduced.** This is
worth stating in the paper — it is a reproducibility claim most work cannot make about its own
earlier sections.

---

## 3. Mining τ — purity beats coverage, monotonically ⭐

Mining τ and inference τ are independent choices. The cache makes every τ free.

| mining τ | coverage of GT boundary | tiles w/ no boundary | ρ (real classes, vs GT) |
|---|---|---|---|
| 0.00 | 165.6% | 19 (1.1%) | +0.418 |
| 0.10 | 102.5% | 127 (7.6%) | +0.389 |
| 0.30 | 41.8% | 331 (19.8%) | +0.643 |
| 0.50 | 19.6% | 481 (28.8%) | +0.704 |
| **0.70** | **5.0%** | 826 (49.5%) | **+0.757** |
| 0.80 | 1.2% | 1167 (69.9%) | +0.693 |
| 0.90 | 0.0% | 1555 (93.2%) | +0.543 |

**Fidelity rises as coverage collapses**, peaking at τ=0.70 on 5% of the adjacency graph. The
prediction that threshold starvation was the problem was **wrong**: low-confidence pixels are not
sparse signal, they are noise that corrupts the statistics. 1.16M boundary pairs is a large
sample; coverage was never the constraint.

### 3.1 `background` is unmineable at any τ, including τ=0 ⭐

| | GT | mined τ=0 | mined τ=0.5 |
|---|---|---|---|
| background share of counted boundary | **40.8%** | **2.7%** | 0.4% |

τ=0 is *no threshold at all* — pure argmax — and background still reaches only 2.7%. Because
`P_final = P_fused · S_pres` and median `S_pres(background) = 0.022` (§9.2b), background loses
the **argmax** too. **Consequence: `background` is not a class in the prior.** It is the
"none of the above" state the method converts *out of*.

### 3.2 Losing background *rewires* the graph rather than shrinking it

With background included, the mined matrix ranked pairs no better than chance (Spearman
**−0.110**, 18/42 sign flips). `building–road` is **−3.15** in GT — they avoid, because LoveDA
labels the pavement between them `background` — and flips to **+0.86** once that intermediary is
gone. Dropping background from both sides lifts agreement to **+0.311**, and to **+0.757** at
mining τ=0.70.

---

## 4. Atomisation — settled, and worth more than the prior ⭐

| atoms | count | median px | max px | mean purity | **oracle ceiling** |
|---|---|---|---|---|---|
| connected components | 41,952 | 179 | **1,048,576** | 0.728 | **72.8%** |
| **SLIC** | 506,064 | 1,551 | 60,740 | **0.928** | **92.8%** |

*Purity* = share of an atom's pixels belonging to its own majority GT class, pixel-weighted.
*Oracle ceiling* = accuracy of giving every atom its own majority GT class — the hard upper bound
for **any** method built on those atoms.

Pixels in unlabelable atoms (purity ≤ 0.50) fall from **12.5% → 1.1%**. The largest connected
component was **an entire tile**: background-assigned pixels are 43% of a typical image, so a
connected component of them is not an object but the union of everything the model was unsure
about.

**This closes ROADMAP Week 8's open question with a measurement, and it is worth 20 ceiling
points — two orders of magnitude more than the co-occurrence prior contributes.**

---

## 5. The gates

### 5.1 Gate 1 — circularity: **passes** ✅

Mining τ=0.70, background dropped: **Spearman(PMI_pred, PMI_gt) = +0.757**, 6/30 sign flips.

Every flip is on a pair GT calls ≈ chance — road–barren (+0.17 → −0.33), road–agriculture
(+0.15 → −0.84), water–barren (+0.39 → −2.06); max \|GT\| among flips is **0.39**. All four pairs
GT calls strong survive with the correct sign: building–water, road–water, barren–forest,
building–barren. **The mined matrix gets the confident facts right and is noisy only where GT has
nothing to say.**

`Spearman(row error, discard rate) = −0.257` — error does **not** concentrate on the classes SAM 3
discards most, so `ANALYSIS §3.2`'s circularity concern is **retired with a number**.

⚠️ **Magnitudes are inflated 4.22×** (mean \|PMI\| 2.424 mined vs 0.574 GT). Rank agreement
survived validation; absolute bits did not. **Never feed raw mined PMI into a scoring function.**

### 5.2 Gate 2 — **fails, and against ground truth too** ⭐

| confusion | px | **GT** `PMI_bnd` | GT would… |
|---|---|---|---|
| forest → agriculture | 23,765,826 | **+0.32** | reinforce |
| water → agriculture | 19,332,270 | **+0.25** | reinforce |
| agriculture → forest | 7,859,015 | **+0.32** | reinforce |
| water → barren | 4,148,715 | **+0.39** | reinforce |

**A perfect matrix would reinforce the baseline's top confusions.** Not a mining defect — no
amount of better mining fixes it. Forest and agriculture genuinely do touch; adjacency and
confusability are the same signal. **The prior cannot be aimed at the confusions**; its target is
the 94% that fell silent, consistent with §7.6's 3:1 discard-to-confusion ratio.

---

## 6. ⛔ The ceiling test — co-occurrence does not earn its place

For each region assigned to background: look only at its confident neighbours, apply M, guess.

| method | pixel accuracy |
|---|---|
| majority class (always `agriculture`) | 38.6% |
| **β=0.00 — pure neighbour vote, M off** | **48.4%** |
| β=0.25 — best mined mixture | 48.6% |
| **β=0.50 with the ORACLE GT matrix** | **48.7%** |
| β=1.00 — pure co-occurrence, no vote | 20.1% |

**A perfect co-occurrence matrix adds 0.3 points over copying the largest neighbour.** The mined
one adds 0.2. Not a mining problem, not circularity — the term carries almost no information
beyond "what is next to it".

### 6.1 Where it *does* help, and why that is not enough

Stratified by how many distinct classes border the region:

| borders | pixels | vote | +mined | **+oracle** |
|---|---|---|---|---|
| 1–3 classes | 133.7M | — | +0.00 | **+0.00** |
| 4 classes | 15.5M | 47.0% | +2.01 | **+4.22** |
| 5 classes | 11.2M | 51.2% | +0.95 | **+6.54** |
| 6 classes | 5.6M | 59.0% | +4.23 | **+6.58** |

Co-occurrence works **exactly where there is something to arbitrate** — and mining captures only
37% of the oracle ceiling there, so better mining could roughly double it. But that stratum is
**10.0% of the residual**, diluting to **+0.40 overall** (oracle +1.06). Meanwhile 73% of
reachable pixels sit in large regions where the oracle ceiling is **+0.9**.

**Co-occurrence is a component, not a thesis.** It belongs in the paper as a measured ablation
row with an oracle bound, not as the contribution.

---

## 7. ⛔ Does recovery move mIoU? — the go/no-go

`selective_recovery_miou.py`, SLIC atoms, inference τ=0.5. Both runs passed the hard gate:
"recover nothing" reproduces **47.37** and **323,084,415**.

| scope | recovered | precision | **mIoU** |
|---|---|---|---|
| **oracle** (told which pixels are real classes) | 33.8% | 56.1% | **50.99 (+3.62)** |
| **honest** (every background-assigned pixel) | 89.0% | 20.7% | **41.18 (−6.19)** |
| honest + best filters (classes × size ≤ 500px × purity ≥ 0.7) | 5.7% | 40.0% | **47.41 (+0.04)** |

> ⚠️ **A supervision leak was found and fixed here, and it must not recur.** The first version
> built regions from `(gt >= 2) & (base == 1)` — only pixels GT says are real classes — which
> hands the method advance knowledge of where to look and makes it immune to damaging true
> background. That is exactly the protection τ-relaxation does not get (§8.2: at τ=0.1 over 70%
> of true background is misassigned, and that is what costs the 5.54 mIoU). It produced a
> plausible, quotable **+3.47** that was not a result. `--regions oracle` reproduces it
> deliberately, labelled as an upper bound.

### 7.1 The decomposition that names the problem ⭐

| | correct px recovered | wrong px recovered |
|---|---|---|
| oracle | **61,342,348** | 48,002,301 |
| honest | **59,514,708** | 227,995,961 |

**Almost the same correct count.** Detection does not help find right answers — the neighbour vote
already finds them. It helps *avoid wrong ones*: **228M → 48M, a 4.7× reduction.** That is the
single most quotable sentence in Week 3.

### 7.2 A hypothesis that failed instructively

Per-class vote accuracy on the *oracle* scope: building 86.8%, water 69.8%, forest 21.9%. On the
*honest* scope building lands at **34.7–36.6% — worse than committing to all classes**, and every
class sits at 35–43%. **The reliability ordering never transferred, because it was never a
property of the class — it was a property of the oracle.**

Note also that every surviving row in the filtered sweep has `max px = 500`: the atom-size ceiling
does all the work, and it only reaches break-even.

---

## 8. ⛔ Detection — nine signals, all at chance

Over every background-assigned pixel: positive = GT real class (**323,084,415**), negative = GT
background (**427,000,158**), base rate **43.1%**.

| signal | level | AUC |
|---|---|---|
| `conf` (= `P_final`) | pixel | **0.582** |
| `gap` = conf − conf2 | pixel | 0.558 |
| `conf2` | pixel | 0.541 |
| `spres_arg` | pixel | 0.520 |
| `fgap` = fconf − conf | pixel | 0.447 |
| `spres_max` | pixel | 0.434 |
| **`fconf` (= `P_fused`, pre-gating)** | pixel | **0.559** |
| `mean_conf` / `max_conf` / `size` | region (SLIC) | 0.576 / 0.516 / 0.467 |
| novelty vs GLOBAL prototypes | region (SLIC) | **0.514** |
| novelty vs PER-IMAGE prototypes | region (SLIC) | **0.528** |
| mean R / G / B | region (SLIC) | 0.586 / 0.580 / 0.585 |
| **gradient energy (texture)** | region (SLIC) | **0.622** |
| *atom size (confound reference)* | region (SLIC) | *0.555* |

All appearance figures are **size-controlled** — see §8.2.

**Three hypotheses died here.**

1. **`τ_low` does not exist.** ROADMAP Week 6 specified a three-way split
   (`ignored` / `unidentified` / `identified`) and it is not realisable from `P_final`: best
   achievable precision 43.3% against a 43.1% base rate.
2. **Presence gating is not hiding the signal.** `P_fused` before the multiply scores **0.559**,
   *worse* than the gated score. §9.2's tile 3487 mechanism is real but does not generalise —
   the second time that tile has promised something the corpus did not deliver.
3. **Novelty detection fails**, at 0.514/0.528 against a ~0.53 floor. And this failure is
   *consistent with its own premise*: if `background` is a residual class it has no compact region
   of feature space to occupy, so there is nothing for a novelty score to be far from.

**Region-level aggregation did not rescue any of it** (0.576 vs 0.582 at pixel level), which rules
out "present but noisy" and leaves "genuinely absent".

The best signal in the entire project is **crude texture at 0.622**, against a 0.555 size-confound
reference. Real, weak, far short of what §7 needs.

### 8.2 A confound that a negative control caught

Run on **random-colour images**, where no signal can exist by construction, novelty-vs-prototypes
scored **0.966**. An atom's mean colour has sampling noise scaling as 1/√size, so any distance
feature partly measures **atom size** — and size correlates with the label, because background
atoms are larger.

Fixed with size-stratified AUC over factor-2 bins, unit-checked on constructed data: pure
size-leak 0.503 → 0.512, size itself 0.779 → 0.571, genuine signal 0.922 → **0.926 preserved**,
pure noise ~0.53. **The empirical floor is therefore ~0.53, not 0.50**, and an `atom size` row is
reported alongside so any signal scoring near it can be discounted.

---

## 9. Where the project stands

**Solved.**

- Atomisation: SLIC, ceiling 92.8%.
- Labelling: a plain neighbour vote is worth **+3.62 mIoU** given the right regions.
- `M_global` is accurate (ρ +0.757) and its circularity risk is retired (−0.257).

**Refuted, each with a number.**

| intervention | result |
|---|---|
| lower τ to 0.1 | −5.54 mIoU, 1.73 wrong per right |
| remove presence gating | −11.97 mIoU, no net recovery |
| co-occurrence prior over neighbours | +0.2 (oracle +0.3) |
| honest region recovery, no abstention | −6.19 |
| honest recovery + best abstention | +0.04 |
| detect recoverability from any SAM 3 signal | 0.434–0.622 AUC |

**The blocker, stated precisely:** the residual is **labelable but not detectable**. Recovery is
worth +3.62 with a perfect detector and +0.04 with the best one available.

---

## 10. Next

1. **Write.** §11 and `PAPER_OUTLINE.md`. The study is substantial and roughly 70% drafted across
   these results files.
2. **One timeboxed DINOv3 attempt**, hard gate **AUC ≥ 0.70 size-controlled or stop**.
   `appearance_detection.py` already takes a feature array; swapping colour for deep features is a
   small change. One week, not two.
   ⚠️ Do **not** cite ConInfer's +2.80 as evidence this will work — ConInfer uses DINOv3 for
   *context modelling*, not for detecting a residual class. Different task, and the earlier
   claim in this project's notes overstated it.
3. **A second dataset.** Every number here is one split of one dataset. OpenEarthMap is the
   cheapest second reference point (~25 min per pass) and any claim above needs it before
   submission.

---

## 11. Artefacts

| file | what |
|---|---|
| `~/outputs/week3/M_global_gt{,_nobg}.npz` + `.md` | GT reference; `_nobg` reproduces §4 |
| `~/outputs/week3/M_global_pred_t07.npz` + `.md` | mined prior, τ=0.70 |
| `~/outputs/week3/mining_tau_sweep{,_high}.md` | §3 |
| `~/outputs/week3/validation_t07.md` | §5 |
| `~/outputs/week3/prior_ceiling.md` | §6 |
| `~/outputs/week3/selective_slic{,_oracle}.md`, `selective_filtered.md` | §7 |
| `~/outputs/week3/recoverability_signal{,_fused}.md` | §8 |
| `~/outputs/week3/atoms_{cc,slic}.md` | §4 |
| `~/outputs/week3/appearance_detection.md` | §8 |
| `~/outputs/week3_fused/cache/` | 1669 `.npz`, now carrying `fconf`/`fpred` |
