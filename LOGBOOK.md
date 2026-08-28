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

## 2026-08-28 (Fri) — per-class τ is real; iSAID pre-registered

**12 commits, 08:32–10:57.** The day the project got its first genuine land-cover gain, and
immediately bounded it.

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

### Numbers to remember from today

- **per-class τ oracle: LoveDA 48.83 (+1.46), real classes +8.63, water +6.70 at τ=0.170**
- best label-free rule on LoveDA: **−0.17** (Otsu), i.e. all three are worse than doing nothing
- oracle τ vs P−R gap: **r = −0.618**, n=6
- iSAID background share: **97.11%**

### Open at end of day

1. 🔴 **`tau_fit.py` is written but has not been run** — its number is not in `WEEK3_RESULTS.md`
   §9a. **This is the most important open number in the project**: it decides whether +1.46 is a
   bound we report or a method we claim.
2. **iSAID cannot run yet.** `ValidationData/val` holds 458 masks and **no images**; the DOTA-v1.0
   val tiles are a separate download. The pre-registration only needed masks, so it is locked in
   regardless.
3. The **share vs confusability** confound is now the paper's central risk. iSAID resolves it.
4. Potsdam's `bg_idx` / catch-all mismatch — flagged, not resolved.

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
