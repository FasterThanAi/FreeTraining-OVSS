# Week 3 Results — `M_global`, atomisation, and the two-dataset mechanism

**Goal (ROADMAP milestone, week 7):** build the corpus-level co-occurrence prior and validate it
before the scoring function is built on it.

**Status:** 🟢 `M_global` built and validated · 🟢 atomisation settled (SLIC) · 🟢 **replicated on a
second dataset** · ⛔ the co-occurrence term does not earn its place · ⛔ recovery does not
improve land-cover segmentation on either dataset · ✅ **but calibrated per-class τ does: +1.18 ±
0.45 mIoU on LoveDA with land cover +8.30 (§9b)**

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
| baseline mIoU | 47.38 *(published 47.4)* | 44.19 *(published 42.9)* |
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

### 7a. ⭐ The confound broken — share drives it, confusability does not — 28 Aug

§7 attributed everything to the catch-all's **share** of ground truth. At n=2 that was
confounded: across LoveDA and OpenEarthMap, share moves together with **confusability** — whether
the catch-all *looks like* the real classes — and confusability was arguably the better
explanation. Two observational points cannot separate two variables that move together.

⛔ **A third dataset would not have fixed this.** iSAID's catch-all is **97.11%** of ground truth
*and* maximally confusable (everything outside 15 small object classes), so both variables are
high; it confirms the ordering and discriminates nothing. **Do not spend GPU time on it expecting
otherwise.**

`confound_split.py` breaks it two ways, CPU-only on the existing cache. First, it **measures**
confusability instead of asserting it: the fraction of true catch-all pixels the model gives a real
class. Second, it stratifies LoveDA into **urban and rural, where the two variables dissociate** —
and they do, cleanly and in opposite directions:

| stratum | tiles | share of GT | **confusability** | discard | AUC `conf` | AUC `conf2` |
|---|---|---|---|---|---|---|
| **rural** | 992 | **42.9%** ⬅ higher | 25.5% | 39.3% | **0.524** | 0.495 |
| **urban** | 677 | 26.0% | **43.3%** ⬅ higher | 18.5% | **0.730** | 0.648 |
| *(pooled)* | 1669 | 36.1% | 30.6% | 29.7% | 0.582 | 0.541 |

Each explanation predicts which stratum detects *worse*, and because the variables dissociate they
name **different** strata:

| | predicts worse detection in | observed |
|---|---|---|
| **share** — more catch-all, less signal | `rural` | ✅ **correct** |
| **confusability** — catch-all resembles land cover | `urban` | ⛔ **wrong** |

> ⭐ **Share is supported; confusability is refuted as the driver.** Detection is worse in the
> higher-share stratum *even though the lower-share stratum is the more confusable one*. A
> catch-all that resembles land cover is not what destroys the signal — its sheer prevalence is.
> **§7 stands as written, and now rests on a comparison where the rival explanation predicted the
> opposite result.**

⭐ **And detection is monotone in share across four strata**, the two middle ones coming from a
split where confusability runs the other way:

| stratum | catch-all share | AUC `conf` |
|---|---|---|
| OpenEarthMap | 0.84% | **0.794** |
| LoveDA urban | 26.0% | 0.730 |
| LoveDA (pooled) | 36.1% | 0.582 |
| LoveDA rural | 42.9% | 0.524 |

⚠️ **This is a stratification, not a randomised intervention.** Urban and rural differ in more than
these two variables — class mix, object scale, scene density — so it constrains the explanation
rather than proving it. State that beside the result. Note also that `all` is the union of the two
strata and is not an independent point; the independent ones are OEM, urban and rural.

⚠️ **The verdict was reported backwards on first run.** The script reasoned in prose about
"higher" and "lower" and swapped both branches. It now names, for each hypothesis, the stratum it
predicts will detect *worse*, and checks which is right — a form that cannot be phrased backwards,
and verified to give the same answer with the strata in either order.


### 7b. ⭐⭐ The mechanism is now CAUSAL — vocabulary intervention, 1 Sep

§7a broke the share/confusability confound by *stratifying* LoveDA, and conceded in writing:
**"this is a stratification, not a randomised intervention."** That sentence was the weakest joint
in the project's strongest claim. It is now gone.

**The idea.** Catch-all share is not a property to go looking for — it is **set by the vocabulary
handed to SAM 3**. So intervene on it.

**Why it cost no GPU time, and why that is better than re-running.** In
`segearthov3_segmentor.py:141-189` every class is an *independent forward pass with its own text
prompt*; there is no softmax, no normalisation, no interaction across classes anywhere. The only
cross-class operation in the entire pipeline is the `argmax` in `predict()`. So **dropping a class
from the vocabulary is exactly equivalent to dropping its channel.** One `--cache-full` pass
(which passed the gate exactly — 44.16 mIoU, 3.78% discard) answers every vocabulary question, and
all thirteen arms read *identical model outputs*: the vocabulary is the only thing that differs.
Re-running per arm would have added sampling noise for nothing.

**The design.** Dose arms raise the catch-all's share; each has an **arity-matched control** that
applies the same merge to a real class instead, so class count changes identically and only the
share differs. Two dose families, because the first one was wrong:

- **`A`/`C` — merge channels by max.** Pre-registered, and **flawed**: it makes the catch-all a
  *union of well-detected prompts*, unnaturally competent at its own pixels. A real catch-all is
  the opposite (LoveDA's `background` has median `S_pres` **0.022**).
- ⭐ **`B`/`D` — drop the prompts, relabel the pixels.** The merged classes are absent from the
  **vocabulary** and their pixels are labelled catch-all, which keeps its own single weak prompt.
  **This is LoveDA's actual situation, and it is the arm the paper quotes.**

#### ⚠️ AUC must be scored direction-agnostically, or an inverted signal reads as a destroyed one

`AUC(−score) = 1 − AUC(score)`, so an AUC of **0.208 is a detector of strength 0.792 with its sign
flipped** — not an absent signal. The first version of the verdict scored raw AUC and announced
"CAUSAL" off exactly that. Everything below is scored on **`det = max(AUC, 1−AUC)`**.

#### The result (OpenEarthMap, 384 tiles, τ = 0.1)

| family | signal | A0 (0.84%) | largest dose (58.20%) | control (0.84%) | dose effect | control effect | ratio |
|---|---|---|---|---|---|---|---|
| ⭐ **B/D** *(faithful)* | `conf` | 0.794 | **0.582** | 0.710 | **0.213** | 0.085 | **2.5×** |
| ⭐ **B/D** *(faithful)* | `conf2` | 0.913 | **0.590** | 0.855 | **0.323** | 0.058 | **5.6×** |
| A/C *(max-merge)* | `conf` | 0.794 | 0.785 | 0.794 | 0.009 | 0.000 | — |
| A/C *(max-merge)* | `conf2` | 0.913 | 0.552 | 0.896 | 0.360 | 0.017 | 21× |

> ⭐ **In the faithful arm BOTH signals degrade causally with catch-all share, and neither
> degradation is explained by class count.** The intervention drives OpenEarthMap from its own
> **0.794** into the **0.58–0.62** band where LoveDA actually sits — it reproduces the
> observational regime rather than merely correlating with it.

**The `A`-family failure is itself informative.** There, `conf` does **not** degrade (0.794 →
0.785); it **inverts** (0.208 / 0.181 / 0.215), because a catch-all built as a union of strong
prompts is *confidently right* about the pixels it absorbed. Only `conf2` degrades. That is the
signature of the design flaw, and it is why `B` exists.

#### Pre-registered predictions, scored

`PREREGISTRATION.md`, committed before each run; nothing edited afterwards.

| forward (A/C) | | faithful (B/D) | |
|---|---|---|---|
| **P1** AUC falls | ✅ | **Q1** `conf2` det ≤ 0.65 | ✅ 0.590 |
| **P2** monotone | ⛔ | **Q2** every control within 0.08 | ⛔ **D25 moves 0.105** |
| **P3** controls < 0.05 | ✅ **0.000** | **Q3** dose ≥ 2× control | ✅ 5.6× (3.1× vs the worst control) |
| **P4** dose ≥ 2× control | ✅ | **Q4** `conf` inverts less than in A | ✅ **no `B` arm inverts at all** |
| **P5** ±0.08 of the line | ⛔ | **Q5** `conf` det ≤ 0.70 | ✅ 0.582 |
| **P6** `conf2` steeper | ⛔ raw / ✅ det — under-specified | | |
| **P7** discard rises | ✅ | | |

⚠️ **Report the three failures, not just the eight passes.**

- **Q2 fails at `D25` (0.105 against a 0.08 bar).** Dropping prompts is not perfectly inert —
  classes present in the image with no prompt disturb the scores wherever their pixels are
  labelled. So the control moves a little, and the honest statistic is the **ratio**: 5.6× against
  the matched control, **3.1× even against the worst control**.
- **Not monotone.** `B` gives 0.507 → 0.624 → 0.582 across the doses, with the *smallest* dose the
  minimum. **Quote the endpoints and state that the interior is non-monotone.** A plausible
  reading: at B10 the removed classes are small (`water`+`road`, 9.3% of GT) so the
  catch-all-assigned set is a balanced mix of removed-class and genuinely-suppressed pixels, which
  is the worst case for separability; by B40 it is dominated by removed-class pixels.
- **P5 mostly fails**, so the *rate* is dataset-specific even though the direction is not. Only
  `B25` lands inside the band (0.624 against a predicted 0.549). Report the ordering, drop the line.

#### What this changes

✅ **§7 is an intervention now, not a stratification.** §7a's concession is replaced by this table.
The mechanism — *a catch-all covering a large share of the scene destroys the information that
would let a suppressed real class be detected* — is supported by a manipulation of the causal
variable with class count controlled, and with predictions committed in advance.

⚠️ Still honest about scope: run on **one** dataset, with doses chosen greedily rather than
randomly, and a control that is not perfectly inert. It constrains the explanation far more
tightly than §7a did; it is not a randomised trial over datasets.



### 7c. ⭐ The reverse arm — P8 fails, and it identifies the causal locus — 1 Sep

LoveDA, random 500 tiles (`--sample 500 --seed 0`). **Gate passed: 47.43 mIoU** against the split's
47.38, discard **27.82%** against 29.68%.

| arm | vocabulary | **GT share** | det `conf` | vs A0 |
|---|---|---|---|---|
| **A0** published | unchanged | 35.72% | 0.592 | — |
| **B45** dose | changed | **45.37%** | 0.640 | +0.048 |
| **D45** control | changed | 35.72% | 0.551 | −0.041 |
| **C45** control | changed | 35.72% | 0.592 | +0.000 |
| **R** reverse | changed | 35.72% | 0.540 | −0.052 |

#### ⛔ P8 fails as stated

P8 predicted that removing `background` from LoveDA's vocabulary would lift detectability to
**≥ 0.65**. It moved to **0.540** — *down* 0.052. The mechanism does not run backwards.

#### ⭐ But `R` is a dissociation, and it is worth more than P8 would have been

`R` changes the **vocabulary** and leaves the **label space** untouched: 35.72% of ground truth is
still catch-all, those pixels simply have no prompt that can name them. Compare with the `B` arms,
which change **both** — a prompt is dropped *and* its pixels are relabelled catch-all:

| | vocabulary | GT share | effect |
|---|---|---|---|
| OEM `B` arms | changed | **raised** | **0.213 / 0.323** |
| LoveDA `R` | changed | unchanged | −0.052 |
| LoveDA `C45`, `D45` | changed | unchanged | +0.000, −0.041 |

> ⭐ **Every arm that changes the vocabulary without changing the label space moves ≤ 0.052. The
> arms that raise the catch-all's share of ground truth move 0.213–0.323. The causal locus is the
> LABEL SPACE, not the prompt list.**

That is what §7 claimed from the beginning — *"`background` **share of GT**"* — and the two arms
together isolate it in a way neither could alone. Removing a catch-all prompt does not remove the
pixels that are genuinely none-of-the-above; it only makes them unnameable, so nothing is
recovered. **A practical consequence: you cannot fix a catch-all-heavy benchmark by editing the
prompt list.**

#### ⚠️ The LoveDA dose arm is UNDERPOWERED, not null

`B45` moves **−0.048** (detectability *rose*) while its own control `D45` moves **+0.041** — the
same magnitude, the opposite sign. An arm whose control moves as far as its dose cannot separate
them. **Do not read LoveDA as evidence against the mechanism**; read it as an arm with no power,
for a reason that is visible in the data:

| | share | det `conf` |
|---|---|---|
| OEM `A0` | **0.84%** | **0.794** |
| OEM `B10` | 10.19% | 0.507 |
| LoveDA `A0` | 35.72% | 0.592 |
| OEM `B25` | 39.57% | 0.624 |
| LoveDA `B45` | 45.37% | 0.640 |
| OEM `B40` | 58.20% | 0.582 |

**The effect is concentrated at the low-share end.** Above ~35% share, four points across two
datasets sit in a **0.58–0.64** band and further increases do nothing. LoveDA *starts* at 35.72%,
already inside that band, so its 9.7-point dose has no room to act in. OpenEarthMap could show the
effect only because it starts at 0.84%.

#### ⚠️ What this costs §7a, stated plainly

If the effect saturates by ~35%, then **share cannot explain the urban/rural spread** — 0.730 at
26.0% against 0.524 at 42.9%, both inside the saturated regime. §7a's stratification still
eliminates *confusability* (it predicts the wrong stratum, decisively), and the intervention
establishes *share* as causal at low share. **Neither accounts for the within-LoveDA gradient, and
the paper must say so rather than presenting a single monotone story.** The honest claim is:

> A catch-all class covering a large share of the label space destroys the information that would
> let a suppressed real class be detected. The damage is done by the time the catch-all reaches
> roughly a third of the scene, and it is not undone by removing the class from the vocabulary.

That is narrower than "detection is monotone in share" and it is supported by an intervention with
class count controlled, on two datasets, with predictions committed in advance.

⚠️ **A fourth verdict branch was missing and is now added.** The script could report "causal", "no
effect", or "inconclusive" — with no branch for *the effect ran the other way* or *the control
moved as much as the dose*. On LoveDA it therefore printed "share is NOT the cause" for a result
that was a **rise** in detectability inside an underpowered arm. Both readings are now separate
branches, and the underpowered branch prints the starting share when it is already high.


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

## 9a. ⛔ Threshold tuning — a real oracle gain that no label-free rule reaches

An earlier version of this file said threshold tuning was closed. **That was wrong**, and the
τ-sweep is why: sweeping one *global* τ found nothing (+0.04 on LoveDA), so the family was
declared exhausted. Per-class thresholds were never tested.

`tau_oracle.py`, three rungs, each strictly more powerful (both swept rows are **oracle bounds** —
they choose thresholds on the evaluation labels):

| rung | free params | LoveDA | OEM |
|---|---|---|---|
| published τ | 0 | 47.37 | 44.16 |
| best global τ | 1 | 47.41 (+0.04) | **49.31 (+5.15)** |
| **best per-class τ** | N−1 | **48.83 (+1.46)** | 49.44 (+5.28) |

**LoveDA's +1.46 is the only gain in this project where land cover genuinely improves** — the six
real classes gain **+8.63 IoU** in aggregate against background's +1.59. It is driven almost
entirely by `water` at τ=0.170 (**+6.70 IoU**), whose baseline precision/recall is 89.5 / 54.7.
Chosen thresholds span **0.170 to 0.595**: one global value is wrong for different classes in
*opposite* directions.

**OEM's +5.28 is the familiar artefact.** `background +49.22`, real classes **−1.71**, and 98% of
it comes from a single *global* change (0.1 → 0.025). Report it as a calibration observation about
the baseline's published τ, never as a contribution.

### 9a.1 No label-free rule reaches it, and the reason is structural ⭐

`tau_rules.py`. Each rule spends **one** label-tuned knob — parity with the baseline, which also
tunes τ per dataset with labels — or none. Only scoring uses labels.

| rule | knobs | LoveDA Δ | share of oracle | OEM Δ | share |
|---|---|---|---|---|---|
| per-class Otsu | **0** | −0.17 | −12% | −5.80 | −110% |
| equal-commitment (q-th percentile) | 1 | −2.98 | −204% | +3.53 | 67% |
| presence-scaled (τ_c ∝ `S_pres_c`) | 1 | −0.74 | −50% | +4.77 | 90% |
| *oracle per-class* | N−1 | *+1.46* | *100%* | *+5.28* | *100%* |

**On LoveDA every label-free rule is worse than the published τ.** On OEM the two that look good
capture the background artefact, not land cover (real classes −1.90), and a plain global τ already
gets 98% of it.

**Why.** What the oracle exploits is per-class **precision**. `water` can afford τ=0.170 because it
is right 89.5% of the time it fires; `agricultural` needs 0.595 because it is not. Across the six
LoveDA classes the oracle threshold tracks the precision–recall gap at **r = −0.618** (n=6, so a
direction rather than a law — `water` is the clear case and drives most of it).

**Precision and the P−R gap are computed from labels by definition.** Confidence percentiles,
presence scores and Otsu splits all describe how the *model* is distributed, not how *often it is
right*, and none of them predicts precision. So:

> **Per-class thresholding is worth +1.46 mIoU, and the quantity needed to set it is precisely the
> quantity a training-free method cannot have.**

That is a better negative than the original "threshold tuning is closed": it bounds the family,
explains the bound, and names what would be needed to reach it. It also generalises — any
training-free pipeline that thresholds a per-class score inherits it.

---

## 9b. ✅ A method — per-class τ, calibrated on ~200 labelled tiles ⭐

The oracle bound in §9a said per-class thresholds were worth +1.46 on LoveDA. This is what a
deployable version gets.

**Protocol.** Fit one threshold per class by coordinate ascent on a calibration subset, evaluate
on disjoint tiles. **No model weights are trained.** SegEarth-OV3 already tunes its single τ per
dataset using labels (0.5 / 0.1), so this is the same protocol with N parameters instead of 1 —
parity, not a leak. The published-τ baseline is recomputed on the *same* held-out tiles.

**The fit maximises land-cover mIoU, excluding the catch-all** (`--objective real`). This matters:
optimising *full* mIoU lets the search buy the metric by repairing an over-predicted `background`
at land cover's expense — on OEM it traded `road −2.14` and `building −1.20` for `background
+53.93`. Full mIoU is still what gets reported.

### LoveDA — 5-fold, every fold positive

| fold | published τ | fitted | Δ |
|---|---|---|---|
| 1 | 47.91 | 49.28 | +1.37 |
| 2 | 45.41 | 47.31 | +1.89 |
| 3 | 48.44 | 49.32 | +0.88 |
| 4 | 46.11 | 46.91 | +0.80 |
| 5 | 48.68 | 49.66 | +0.98 |

**+1.18 ± 0.45 mIoU** (worst +0.84), **81% of the +1.46 oracle bound**.

| class | Δ IoU | | class | Δ IoU |
|---|---|---|---|---|
| **water** | **+6.78** | | building | +0.28 |
| barren | +0.98 | | road | +0.10 |
| forest | +0.37 | | agricultural | −0.21 |
| | | | **`background`** | **−0.01** |

**Background is untouched; the whole gain is land cover — +8.30 IoU in aggregate.** Structurally
immune to the background-unwinding artefact that caught three earlier results.

**Why it works, and where.** `water`'s baseline precision/recall is **89.5 / 54.7** — right nine
times in ten when it fires, finds half of what is there. Its fitted τ is **0.195** against a
global 0.5. The fitted thresholds span **0.195–0.600**, so one global value is wrong for different
classes in opposite directions, and the classes that gain are exactly those with a large
precision–recall asymmetry.

### Calibration cost

| tiles | Δ | sd | worst draw |
|---|---|---|---|
| 10 | −2.14 | 1.99 | −5.59 |
| 50 | +0.08 | 0.97 | −0.99 |
| 100 | +0.54 | 0.71 | −0.57 |
| **200** | **+0.79** | **0.35** | **+0.43** |
| 400 | +1.21 | 0.16 | +1.04 |

**~200 tiles is where every draw turns positive.** Below 50 it actively hurts.

### OpenEarthMap — a benchmark artefact, not a method failure

| objective | full mIoU | `background` | 8 real classes |
|---|---|---|---|
| `all` | **+5.80** ± 2.31 | +53.93 | **−1.77** |
| `real` | +0.16 ± 0.93 | −11.04 | **+12.45** |

Land cover improves substantially under the `real` objective — `building +4.35`, `road +2.95`,
`pavement +2.71`, `cropland +2.04` — and **`background` pays for all of it**, leaving full mIoU
flat. OEM's `background` sits at 17.13 IoU with 17.33% precision, so its mIoU is dominated by one
pathologically calibrated class and you can improve land cover *or* the metric, not both. **A
caveat about the benchmark, not about the method.**

### ⚠️ The scope limit — calibration must match the evaluation distribution

Fitting on LoveDA **train** and evaluating on **val** gives **−0.12**. The splits differ sharply:

| | train | val |
|---|---|---|
| `background` share of GT | 35.8% | 36.1% |
| **discarded to background** | **14.54%** | **29.68%** |
| worst class | barren 37.4% | forest 34.6% |

**Identical background share, 2.04× different discard.** Train said `water` needed no adjustment
(fitted 0.500); val's entire gain comes from water at 0.170. State this beside the gain.

> ⭐ **That comparison also constrains §7's mechanism.** Background share cannot be the sole driver
> of the residual — held constant here, the residual doubles. It supports the *confusability*
> reading over the *share* reading, and it means **29.68% is a LoveDA-val number, not a LoveDA
> number.**

### 9c. ✅ Confirmed end-to-end by the pipeline — 28 Aug

§9b is arithmetic on a cached `(gt, pred, conf-bin)` histogram, proved equivalent to the
segmentor's rule by `verify_perclass_tau.py`. **It has now also been run.** `prob_thd` accepts a
per-class vector indexed by the argmax class; `tau_deploy.py` fitted on 200 calibration tiles and
predicted the result, then `eval.py` produced it on the 1469 disjoint tiles.

| | predicted from the cache | **measured by `eval.py`** |
|---|---|---|
| published τ = 0.5 | 47.16 | **47.16** |
| per-class τ | 48.35 | **48.35** |
| **Δ** | **+1.18** | **+1.18** |

**Both absolute values reproduce to the reported precision**, and every per-class Δ agrees to
**≤ 0.04** — inside the float16 cache noise that was predicted in advance. The histogram shortcut
is therefore validated as a measurement instrument, not just as arithmetic: every τ sweep,
oracle bound and cross-validation in §9a/§9b rests on it, and all of them inherit this.

| class | predicted Δ | measured Δ | class | predicted Δ | measured Δ |
|---|---|---|---|---|---|
| **water** | +6.54 | **+6.54** | forest | +0.45 | +0.44 |
| barren | +1.07 | +1.08 | agricultural | −0.25 | −0.26 |
| building | +0.13 | +0.17 | **road** | −0.54 | **−0.53** |
| `background` | +0.88 | +0.85 | | | |

**Real classes +7.44, 90% of the total.** The fitted thresholds span **0.175–0.675** against a
single 0.5, and the precision/recall table shows the mechanism directly: `water` trades precision
89.7 → 86.4 for recall 54.1 → 63.2 at τ=0.175, while `road` moves the other way at τ=0.675
(precision 69.7 → 73.4, recall 70.6 → 66.3).

⚠️ **Two honest notes, both of which must appear beside the number.**

1. **`background` gains +0.85 here, against −0.01 in the 5-fold §9b.** It is 10% of the total, not
   110% as in the OEM artefact (§8.1), and it is *incidental*: the fit objective was `real`, which
   excludes the catch-all, so the optimiser was never rewarded for it. But §9b's "background
   untouched" is a property of that protocol, not of the method — do not generalise it.
2. **`road` loses 0.53.** The fit raised road's threshold on the calibration tiles and that did not
   transfer. A per-class rule can hurt a per-class result; report the full table, never the mean
   alone.

**This run is a single fit on 200 calibration tiles**, so it sits on the learning curve
(+0.79 ± 0.35 at n=200), not on the 5-fold protocol. That +1.18 matches §9b's 5-fold mean is a
coincidence of two different protocols — **quote them separately.** §9b's +1.18 ± 0.45 remains
the headline because it carries an error bar; this run is the end-to-end verification.

> **A bin-edge bug was found by writing that test, and fixed.** `confusion_at` computed its bin
> edge with `(tau * nbins).astype(int)`, which **truncates**: `29/200 * 200` is
> `28.999999999999996`, so the function scored a threshold one bin *below* the one it reported.
> Now `np.rint`. **No recorded number changes** — only 7 of the 201 grid values are affected
> (0.145, 0.285, 0.290, 0.565, 0.570, 0.575, 0.580) and none was ever chosen; 0.170, 0.195, 0.500,
> 0.595 and 0.600 all bin exactly, as do the published τ 0.5 / 0.1. The reported mIoU was always
> the mIoU of the configuration actually evaluated, so the *gains* were never at risk — only a
> *deployed* threshold could have gone wrong, which is exactly what this step exists to catch.

**How it was run** (workstation, ~50 min for both passes):

```bash
cd ~/FreeTraining-OVSS && python scripts/verify_perclass_tau.py     # CPU, must pass first
python scripts/tau_deploy.py --cache ~/outputs/week3_fused/cache --tau 0.5 \
  --calib 200 --seed 0 --split-out ~/splits/loveda_heldout.txt \
  --cfg-out ~/SegEarth-OV-3/configs/cfg_loveda_perclass.py \
  --md ~/outputs/week3/tau_deploy.md
cp ~/FreeTraining-OVSS/reference/segearthov3_segmentor.py ~/SegEarth-OV-3/
cd ~/SegEarth-OV-3
python eval.py ./configs/cfg_loveda.py \
  --cfg-options test_dataloader.dataset.ann_file=$HOME/splits/loveda_heldout.txt
python eval.py ./configs/cfg_loveda_perclass.py
```

Both passes evaluate the **same 1469 held-out tiles**, so the difference is the claim. The
absolute values will not be 47.37 — that is the full split, and this is a subset.

### 9d. ⭐ Precision *is* predictable without labels — and it still does not set the threshold

`precision_proxy.py`, LoveDA, 200 calibration tiles, 1469 held out. Eight label-free per-class
statistics against three label-derived targets. §9a's impossibility argument was that the oracle
exploits per-class **precision**, which is label-derived by definition. **The first half of that is
now measured to be false, and the argument comes out stronger.**

| proxy | ρ vs precision | p *(exact)* | ρ vs oracle τ | p *(exact)* | Δ mIoU held out |
|---|---|---|---|---|---|
| **`mean_conf`** | **+0.943** | **0.017** | −0.371 | 0.497 | **−0.18** |
| **`gate_ratio`** | **+0.943** | **0.017** | −0.371 | 0.497 | −0.05 |
| **`head_conf_ratio`** ⬅ *cross-head* | **+0.943** | **0.017** | −0.486 | 0.356 | −0.06 |
| **`sem_inst_agree`** ⬅ *cross-head* | **+0.886** | **0.033** | −0.429 | 0.419 | **+0.06** |
| `mean_margin` | +0.829 | 0.058 | −0.086 | 0.919 | −0.04 |
| `presence` | +0.657 | 0.175 | −0.086 | 0.919 | **+0.42** |
| `argmax_stability` | +0.371 | 0.497 | −0.543 | 0.297 | −0.04 |
| `inst_fires` ⬅ *cross-head* | −0.086 | 0.919 | **+0.657** | 0.175 | +0.02 |
| *random proxy ×200* | — | — | — | — | *+0.11 ± 0.28, p95 **+0.58*** |
| **oracle per-class τ** | — | — | — | — | **+1.24** |

p-values are exact, by enumerating all **720** relabelings — at six classes |ρ| = 0.6 arises by
chance about a fifth of the time, so a table lookup would be misleading here.

**Four proxies rank the classes by precision at p ≤ 0.033**, two of them cross-head. SAM 3 is
*rank-calibrated across classes* on LoveDA — a small finding in itself, and not one the baseline
reports. **All four buy nothing.** The best row in the whole table is `presence` at +0.42, below the
random control's p95 of +0.58; the two cross-head rows land at **+0.06** and **−0.06**.

⭐ **The §9d prediction, written down before the run, held exactly.** Cross-head agreement was
committed in advance as the one remaining candidate that asks how often the model is *right* rather
than how its scores are *distributed* — and predicted to fail anyway, for the coupling reason
below, "regardless of how good a precision estimate it turns out to be". It is a **good** precision
estimate (ρ +0.886, p 0.033) and it moved mIoU by +0.06. **Measuring precision better is not the
missing ingredient**, and that is now demonstrated rather than argued.

<small>The one proxy pointing at the target rather than at precision is `inst_fires` — how often the
instance head fires at all for a class — at ρ +0.657 against the oracle τ, the largest in that
column. It is SegEarth-OV3's own things/stuff duality reappearing. But p = 0.175 at n=6 and Δ mIoU
+0.02, so it is a curiosity to note, not a result to cite.</small>

**Because the chain breaks one link later than §9a said.** Precision does not determine the
threshold — the relationship is not even monotone:

| class | precision | oracle τ | |
|---|---|---|---|
| water | 88.5 | **0.175** | highest precision → lowest τ |
| building | 75.7 | 0.190 | |
| **road** | 68.5 | **0.675** | ⬅ *mid* precision → **highest** τ |
| agricultural | 66.6 | 0.565 | |
| forest | 61.6 | 0.410 | |
| barren | 55.6 | 0.375 | lowest precision → *middle* τ |

ρ(precision, oracle τ) = **−0.429, p = 0.419**; ρ(P−R gap, oracle τ) = **−0.371, p = 0.497**.
Neither is distinguishable from chance. §9a's `r = −0.618` was already hedged as "a direction
rather than a law, driven by `water`" — this confirms that hedge was necessary and sharpens it.

> ⭐ **The restated bound, which is stronger than the original.** The right per-class threshold is
> not a function of any single-class quality statistic, because it is the solution to a **coupled**
> multi-class IoU objective: raising `road`'s τ pushes pixels into `background`, which changes
> `background`'s IoU, which changes every other class's optimum. A per-class scalar — measurable or
> not — cannot express that. **This is why the 6-parameter fit of §9b works and every 1-parameter
> rule fails, and it explains the failures rather than merely recording them.**

That reframes the negative from *"we cannot measure the quantity that sets the threshold"* to
*"we can measure it, and it is the wrong quantity"* — which is a better result, generalises to any
training-free pipeline thresholding a per-class score, and closes the loophole a reviewer would
otherwise open.

**Provenance.** `fused = max(P_sem, P_inst_agg)` destroys the distinction between the heads, so the
cross-head rows required instrumentation rather than analysis: the segmentor now carries each head
separately and the cache stores their own top-1 (`iconf/ipred`, `sconf/spred`). ✅ The re-run
(`~/outputs/week3_heads`) **passed the validation gate exactly** — 47.37 mIoU, 323,184,908 discarded
(29.68%) — so the extra accumulators are observation-only and the two caches are comparable.

> **What this closes.** Eight label-free proxies, four of which measurably track per-class
> precision, none reaching a +1.24 oracle bound and none beating a random control. Together with
> §9a's three threshold rules that is **eleven label-free attempts**, and the coupling argument says
> why they all fail. This is no longer "we did not find one"; it is a bounded family with a stated
> mechanism.


### 9e. ⛔⭐ The +1.18 is a RURAL result, and the thresholds do not transfer — 31 Aug

`tau_domain.py`, CPU-only on the existing cache. §9b's gain rests on one dataset, and its scope
limit (train→val gives −0.12) was measured but never bounded. LoveDA's urban and rural strata are
the cheapest available test: they differ **2×** in discard rate (18.5% vs 39.3%, §7a) and flip the
sign of 10 of 15 class-adjacency pairs (`ANALYSIS §4.4`), so fitting within and across them is
closer to independent replication than to a re-split.

**Protocol.** 5-fold within each domain; then three transfer arms — **matched** (fit on the target
domain), **mismatched** (fit on the other), **pooled** (fit on a draw across both) — **all at the
same 200-tile budget** and all scored on **identical held-out tiles**. Equal N is not a detail: the
§9b learning curve shows calibration size dominates below 200 tiles, so fitting the mismatched arm
on the whole other domain would have confounded domain shift with data quantity.

#### ⭐ The method is a rural result

| domain | tiles | Δ mIoU | folds positive | worst |
|---|---|---|---|---|
| **rural** | 992 | **+2.77 ± 0.92** | **5/5** | +1.61 |
| **urban** | 677 | **+0.10 ± 0.39** | **2/5** | −0.31 |

**Urban is not distinguishable from zero.** §9b's pooled **+1.18 ± 0.45** is carried by the rural
half of the split. ⚠️ **Never quote +1.18 without this breakdown.** It remains correct as a result
on LoveDA val as published — that is the benchmark — but it is not a claim about urban imagery.

#### But land cover improves in BOTH domains — the catch-all pays for it in urban

| | rural | urban |
|---|---|---|
| real classes, aggregate | **+18.61** | **+4.18** |
| `background` (catch-all) | +0.79 | **−3.51** |
| **full mIoU** | **+2.77** | **+0.10** |

(4.18 − 3.51)/7 = +0.10 — the flat urban result is *entirely* the catch-all paying for land cover.

⭐ **This is the OpenEarthMap artefact of §8.1 with the sign reversed.** There, `background` gained
+22.67 and paid for real classes losing 2.11, inflating mIoU. Here, land cover gains and
`background` pays, deflating it. Same mechanism, opposite direction — **an independent
confirmation that full mIoU is a poor metric wherever a catch-all class is large**, which is the
paper's own argument arriving from the other side. Per class: `water` +10.15 and `forest` +6.95 in
rural against +1.22 and −0.01 in urban — exactly §7.3's two *deep-discard* classes, and rural is
where their mass is.

#### The domains want genuinely different thresholds

| class | rural | urban | difference |
|---|---|---|---|
| **road** | 0.725 | 0.225 | **0.500** |
| **forest** | 0.095 | 0.475 | **0.380** |
| **agricultural** | 0.600 | 0.300 | **0.300** |
| building | 0.430 | 0.305 | 0.125 |
| water | 0.170 | 0.115 | 0.055 |
| barren | 0.375 | 0.375 | 0.000 |

Mean |difference| **0.227**, max **0.500**, against a single published τ of 0.5 for every class and
both domains. This is `ANALYSIS §4.4`'s domain-specificity reappearing **in the method** rather
than in the co-occurrence prior — and unlike §4.4 it is attached to a metric that moves.

#### ⛔ It does not transfer — the wrong domain is worse than no calibration at all

| target | matched | mismatched | pooled |
|---|---|---|---|
| **rural** | **+2.32** ± 0.46 | **−0.40** ± 0.42 | +0.77 ± 0.68 |
| **urban** | +0.02 ± 0.16 | **−1.11** ± 0.14 | −0.32 ± 0.35 |

**Both mismatched arms land below the published τ.** Applying another domain's thresholds is an
active harm, not a smaller benefit. That is the honest scope statement §9b was missing, and it is
stronger than the train→val −0.12 because the budget is controlled.

⚠️ **Pooling is not a safe default either** — rural keeps only +0.77 of +2.32, urban goes −0.32.
A mixed calibration set fits one threshold vector to two different optima, which is **the same
failure as the global τ it replaces, one level up.** The method's own argument recurses.

> **What this changes in the paper.** The claim narrows from *"per-class τ is worth +1.18 on
> LoveDA"* to *"per-class τ is worth +2.77 where the residual is large, nothing where it is small,
> and the thresholds are domain-specific enough that transferring them hurts."* That is a smaller
> claim and a more defensible one, and it comes with a practitioner-facing rule: **calibrate on the
> distribution you will evaluate on, and do not pool across domains that differ this much.**

⚠️ **Two verdict-logic defects, found and fixed.** The gate for "the method holds" asked only
`mean > 0`, so it passed urban's +0.10 with 3 of 5 folds negative; it now requires
`mean − 2·sd > 0` **and** every fold positive. And transfer retention was printed as a percentage
against a +0.02 denominator ("−7301% retained"); ratios are now suppressed below a 0.25 mIoU
matched gain. **The tables were correct throughout — only the prose was wrong**, which is the
third time this session that generated verdict text needed checking against its own table.



### 9f. ⛔ No label-free rule for *when* calibration pays — and it is the same bound as §9d — 1 Sep

`discard_criterion.py`. §9e left the method's scope unusable in practice: *"+2.77 on rural, +0.10
on urban"* is not something a practitioner can act on, because they do not know which stratum they
are in until they have already bought the labels. Those strata also differ 2× in discard rate, so
domain and residual size are confounded there exactly as share and confusability were before §7a.

**The statistic is deliberately label-free**: the fraction of *all* pixels the model assigns to the
catch-all, computable from a forward pass over unlabelled tiles. It is a good stand-in for the
familiar discard rate — Spearman **+0.885** on LoveDA, **+0.924** on OEM — which is the only reason
the substitution is legitimate.

#### LoveDA — the gain is U-shaped in the residual, not monotone

| stratum | catch-all fraction | labelled discard | **Δ mIoU** | sd | worst fold | domain mix |
|---|---|---|---|---|---|---|
| 1 | 0.001–0.186 | 6.0% | **+2.13** | **0.18** | **+1.96** | rural 54 / urban 45 |
| 2 | 0.186–0.358 | 15.5% | +0.78 | 0.40 | +0.16 | rural 42 / urban 57 |
| 3 | 0.359–0.767 | 27.9% | +0.81 | 0.24 | +0.43 | rural 44 / urban 55 |
| 4 | 0.771–1.000 | 85.8% | **+3.22** | **3.15** | **−1.22** | rural 96 / urban 3 |

Random control, same sizes: +1.81, +1.09, +0.62, +0.81 — spread **1.19** against the stratified
**2.43**, only 2.0×, and that entire margin comes from stratum 4, whose **sd (3.15) exceeds the
whole stratified spread**. ρ(fraction, gain) = **+0.400** over four strata.

⛔ **There is no rule here.** The largest and most *reliable* gain is in the stratum with the
**least** residual (+2.13 ± 0.18, worst fold +1.96) — the opposite of the intuition the experiment
was built on. The high-residual stratum has the largest mean but is worthless as guidance: a
3.15 sd and a fold at **−1.22**.

#### OpenEarthMap — inconclusive, control moves as much

Strata +0.51 / −0.08 / +0.47 against a control of −0.10 / +0.06 / +0.32; spread **0.59** against
**0.41**. ρ = **−0.500**, the opposite sign to LoveDA. **Nothing replicates.**

⚠️ Do not rescue this with the variance reading either. ρ(fraction, sd) is +0.800 on LoveDA and
**−0.500 on OEM** — opposite signs, over four and three points. Four points cannot support a
correlation, and this one does not survive the second dataset. **The honest statement is that
neither the mean nor the spread of the gain is predicted by the label-free residual.**

#### ⭐ Why it fails is the same reason §9d fails

§9e's rural/urban difference is therefore **not** explained by residual size. The likely
explanation is **class composition**: the whole gain lives in classes with a large
precision–recall asymmetry (`water` +10.15 and `forest` +6.95 in rural, against +1.22 and −0.01 in
urban, §9e), and rural simply contains more of them. But **which** classes those are is precisely
per-class precision — the quantity §9d showed is measurable label-free and yet **does not determine
the threshold**, because the objective is coupled across classes.

> ⭐ **So one bound explains both halves.** You cannot choose the thresholds without labels (§9a,
> §9d, eleven attempts), and you cannot predict whether choosing them will pay without labels
> either (§9f, two datasets). **The ~200-tile calibration cost is irreducible: it buys the answer
> and the question at the same time.** That is a cleaner closing statement than the deployment
> criterion this experiment was built to find, and it generalises to any training-free pipeline
> that thresholds a per-class score.

⚠️ **The control did its job.** On OEM the random strata move nearly as far as the stratified ones,
which is what "no signal" looks like; without it, +0.51 / −0.08 / +0.47 might have been written up
as a pattern.

### 9g. ⭐/⚠️ What the rural/urban gap is made of — half explained, half not — 1 Sep

`composition.py`. §7c ruled out catch-all share (the effect saturates, both strata are past it);
§9f ruled out residual size (the gain is U-shaped in it and nothing replicates). Class composition
was what remained.

#### The decomposition — arithmetic, not inference

mIoU is the **unweighted** mean over classes, so the gap is exactly the mean of the per-class
differences. A domain does not gain by *containing* more of a class; it gains when that class's own
IoU improves more there.

| class | Δ IoU rural | Δ IoU urban | difference | share of the gap |
|---|---|---|---|---|
| **water** | +10.15 | +1.22 | **+8.93** | **48%** |
| **forest** | +6.95 | −0.01 | **+6.96** | **37%** |
| `background` | +0.79 | −3.51 | +4.29 | 23% |
| building | −0.14 | +0.70 | −0.84 | −4% |
| agricultural | +0.73 | +1.38 | −0.65 | −3% |
| barren | +0.61 | +0.48 | +0.13 | 1% |
| road | +0.31 | +0.42 | −0.11 | −1% |

⭐ **`water` and `forest` carry 85% of the 2.68-point gap.**

#### ⭐ The precision–recall *gap* ranks the cells; precision alone does not

Over all 12 (domain, class) cells, permutation p-values:

| candidate | ρ vs Δ IoU | p |
|---|---|---|
| **precision − recall gap** | **+0.713** | **0.013** |
| recall | −0.608 | 0.041 |
| class share of the domain | +0.497 | 0.102 |
| fraction discarded | +0.490 | 0.113 |
| **precision** | **+0.168** | **0.600** |

**This is a measured confirmation of the method's stated mechanism**, which until now rested on
`water` as an anecdote: a class that is right far more often than it fires can afford a lower
threshold, and calibration collects exactly that. ⭐ **And precision *alone* explains nothing
(+0.168, p 0.60)** — it is the *asymmetry* that matters, which is the same lesson as §9d, arriving
independently: precision is measurable and is still the wrong single quantity.

`forest` is the clean case, and the two domains are in different regimes entirely:

| | discard | precision | recall | P−R gap | Δ IoU |
|---|---|---|---|---|---|
| rural forest | **67.0%** | 35.4 | **9.9** | **+25.5** | **+6.95** |
| urban forest | 12.1% | 61.8 | 68.9 | −7.1 | −0.01 |

#### ⛔ But the largest contributor is NOT explained — state this first

| | share | discard | P−R gap | Δ IoU |
|---|---|---|---|---|
| rural water | 11.6% | 36.7% | **+35.0** | **+10.15** |
| urban water | 11.8% | 25.6% | **+34.4** | **+1.22** |

**Same share to within 0.2 points, same precision–recall gap to within 0.6 points — and an 8×
different gain.** `water` is **48% of the gap**, so the winning statistic ranks the cells on
average while failing on the single largest thing it is meant to explain. A reviewer comparing
those two rows sees it immediately; the paper says it before they do.

The likely residual factor is the **coupling** of §9d — in urban, lowering `water`'s threshold
takes pixels from the dense `building` and `road` classes, so the same threshold move buys less.
That is consistent with everything else here, and it is **not measured**. Treat it as a hypothesis.

> **The honest summary.** The gap is class composition: 85% of it is `water` and `forest`, and
> across the 12 cells the precision–recall asymmetry predicts which classes move (ρ +0.713,
> p 0.013) while precision alone does not. `forest` is fully explained by two domains being in
> opposite regimes. **`water`, at 48% of the gap, is not** — and roughly half the gap therefore
> still lacks a mechanism.

⚠️ **This is an explanation, not a predictor.** Every statistic here is label-derived, so it does
not resurrect the label-free rule §9f ruled out. It says what the gain is made of, not how to know
in advance — and §9d/§9f already say why the second is unreachable.

⚠️ **A verdict guard was added because the first version overclaimed.** The script reported "no
longer unexplained" on the strength of ρ alone. It now checks the top contributors individually:
if a class carries ≥10% of the gap while its winning statistic is effectively identical across
domains, that share is reported as still unexplained. Ranking the cells on average is not the same
as explaining the classes the gap is made of.



### 9h. ⭐⭐ Both metrics — and OpenEarthMap turns POSITIVE — 1 Sep

`metric_report.py`. The catch-all artefact has now been measured twice in opposite directions
(§8.1 inflating, §9e deflating), which is enough to make a **recommendation** rather than a
complaint: a benchmark with a catch-all should report **mIoU over the real classes beside the
headline**. This computes both from the same held-out folds under the §9b protocol, so nothing is
transcribed between tables.

#### ⭐ The leverage, which is not obvious

mIoU is an **unweighted** mean over N classes, so **the catch-all owns exactly 1/N of the metric
however meaningful it is** — **14.3%** on LoveDA for a class the annotation guide defines as
"everything else", **11.1%** on OpenEarthMap for a class covering **0.84% of the pixels**. A
22.67-point move in one class is **3.24 mIoU** on a 7-class benchmark before anything real has
changed.

#### The method in both metrics

| | LoveDA (7 cls) | OpenEarthMap (9 cls) |
|---|---|---|
| baseline, full mIoU | 47.37 | 44.16 |
| baseline, **catch-all-excluded** | 47.68 | **47.54** |
| *baseline distortion* | *+0.31* | ***+3.38*** |
| fitted, full mIoU | 48.53 | 44.47 |
| fitted, **catch-all-excluded** | **49.04** | **49.30** |
| **Δ full** | **+1.16** | **+0.30** |
| ⭐ **Δ catch-all-excluded** | **+1.36** | ⭐ **+1.75** |
| catch-all Δ IoU | −0.02 | **−11.30** |

> ⭐⭐ **The method is positive on BOTH datasets under the recommended metric.** OpenEarthMap was
> written up as flat (+0.16 full mIoU) and a "benchmark artefact". It is not flat — it gains
> **+1.75 catch-all-excluded mIoU**, more than LoveDA does. **The paper gains a second positive
> dataset, which was the thinnest part of the method claim.**

**OEM's headline decomposes exactly:** full **+0.30** = **+1.56 from the eight real classes** and
**−1.26 from `background` alone** (−11.30 IoU spread over 9 classes). The catch-all *cancels a real
gain*, so **the headline understates the method** — the mirror of §8.1, where it overstated it.
`building` +4.34, `road` +2.94, `pavement` +2.71, `cropland` +2.50.

**LoveDA is the clean case:** +1.16 full, +1.36 excluded, catch-all **−0.02**. Both metrics agree,
so the gain is structurally immune to the artefact — which is what §9b claimed and this confirms on
the metric itself rather than on a per-class table.

**And OEM's published baseline is depressed by 3.38 points by one class.** `background` sits at
**17.13** IoU against a real-class mean of **47.54**. SegEarth-OV3's 42.9 and our 44.16 are both
dragged down by a pathologically calibrated catch-all covering 0.84% of the pixels. That reframes
"OEM is a hard benchmark" as partly a property of its metric.

#### ⚠️ A units inconsistency across this document — fix before writing

§9b and §9e quote land cover as an **aggregate (sum over classes)**; this section quotes the
**mean**, which is what "catch-all-excluded mIoU" means and what is comparable to full mIoU.
**They are the same measurements in different units:**

| quoted | as aggregate | **as catch-all-excluded mIoU** |
|---|---|---|
| §9b LoveDA 5-fold | +8.30 | **+1.38** |
| §9e LoveDA rural | +18.61 | **+3.10** |
| §9e LoveDA urban | +4.18 | **+0.70** |
| §9b OEM | +12.45 | **+1.56** |

⛔ **The paper must use the mean everywhere.** An aggregate of +8.30 beside a full mIoU of +1.18
invites the reader to compare two numbers that are 6× apart in units, and a reviewer will read
+8.30 as an mIoU gain.

⚠️ **A reporting bug was fixed here too.** The verdict expressed the catch-all's role as a
percentage of the total change, which on OEM printed *"414% of the change is the catch-all"* —
arithmetically true and unreadable, because the two contributions have **opposite signs**. It is
now a decomposition that names both parts in mIoU units.


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
| per-class τ, label-free | −0.17 (best of 3 rules) | +4.77 but real classes −1.90 |
| **per-class τ, calibrated (§9b)** | ✅ **+1.18 ± 0.45, real +8.30** | +0.16, real **+12.45** |
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

**`--limit` is a smoke test, not a subset.** It takes the first *n* filenames in sorted order,
and LoveDA val is Rural (992) + Urban (677) merged with **disjoint ID ranges** — so `--limit 500`
is essentially rural-only. It surfaced as a run reporting **40.94 mIoU / 41.15% discard** against
the split's 47.38 / 29.68%, which reads as a broken baseline and is nothing of the kind: those are
rural's own figures (5-fold published-τ baselines 40.66–44.16, discard 39.3%, §9e). Caught because
the domain split had already been measured; without §9e it would have looked like a label-alignment
bug and cost a day. `--sample n --seed k` now draws at random, `--limit` prints a warning, and the
two are mutually exclusive.

**A symlink loop damaged the dataset directory.** `ln -s target dest` creates the link *inside*
`dest` when `dest` already exists, producing `images/val/val → images/val`. Use `ln -sfn`. The
tell was a file count of 385 where 384 was expected.

---

## 12. Next

1. **Write.** See `PAPER_OUTLINE.md`.
2. ~~Look up SegEarth-OV3's published OpenEarthMap mIoU~~ ✅ **42.9**, against our **44.19** on
   384 of the official 500 tiles — **+1.29**. Not a reproduction gate at 77% of the split, but it
   is the sanity anchor: a broken class-name-to-label-value mapping would have landed far off in
   one direction, so the OEM prep is sound and §7's mechanism table rests on a correct baseline.
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
