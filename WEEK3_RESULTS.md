# Week 3 Results — `M_global` construction and validation

**Goal:** build the corpus-level co-occurrence prior, and validate it *before* the scoring
function is built on top of it.

**Status:** 🟢 `M_global` built, mining τ resolved · 🟢 GT reference reproduces `ANALYSIS §4`
exactly · 🔴 **the co-occurrence term does not survive its ceiling test — §4 below is the
decision point for the project**

**Date:** 2026-08-25

---

## 1. What was built

| script | what it does |
|---|---|
| `build_m_global.py` | mines the prior from SAM 3's confident predictions (`--source pred`, no labels) or from GT (`--source gt`, the reference). Runs off the `.npz` cache — CPU-only |
| `validate_m_global.py` | gate 1 (circularity) and gate 2 (does M predict the baseline's confusions) |
| `sweep_mining_tau.py` | at what τ should the prior be *mined*? |
| `prior_ceiling.py` | **the go/no-go test** — what can the prior actually recover? |

Two definitions were settled while writing them, both now in `CLAUDE.md`:

- **Below-τ pixels are UNKNOWN, not background.** Counting them as background would build the
  prior out of the very mass the project exists to recover.
- **Shared-boundary counts are symmetric by construction.** "M is directed" came from the
  *confusion* matrix (§8.1b), a different object. Direction lives in the row-normalised
  conditional `P(c | neighbour=n)`, asymmetric because the class marginals differ
  (mean |P(c\|n) − P(n\|c)| = **0.115**, max **0.262** on GT).

One defect was caught by a synthetic test before it reached real data: an unobserved pair gives
`log2(0) = −inf`, and clamping to `0.0` reports the *strongest* exclusion as "chance".
`building–water` is exactly such a pair. Fixed with Dirichlet smoothing before the log; `α=0`
still reproduces `cooccurrence_gt.py` bit-exactly.

---

## 2. The GT reference reproduces `ANALYSIS §4` exactly ✅

A different code path (the `.npz` cache, not the PNG masks) on the same split:

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

Row variances match too — agriculture 0.039 (published 0.04), road 0.661 (0.66), building 1.297
(1.30), water 1.685 (1.69). **§4's foundation is independently reproduced.** Cite this.

---

## 3. Mining τ — purity beats coverage, monotonically ⭐

Mining τ and inference τ are independent choices; nothing forces the prior to be built at the τ
the baseline runs at. The cache makes every τ free.

| mining τ | coverage of GT boundary | tiles w/ no boundary | ρ (real classes) |
|---|---|---|---|
| 0.00 | 165.6% | 19 (1.1%) | +0.418 |
| 0.10 | 102.5% | 127 (7.6%) | +0.389 |
| 0.30 | 41.8% | 331 (19.8%) | +0.643 |
| 0.50 | 19.6% | 481 (28.8%) | +0.704 |
| **0.70** | **5.0%** | 826 (49.5%) | **+0.757** |
| 0.80 | 1.2% | 1167 (69.9%) | +0.693 |
| 0.90 | 0.0% | 1555 (93.2%) | +0.543 |

**Fidelity rises as coverage collapses**, peaking at τ=0.70 with 5% of the adjacency graph. The
prediction that threshold starvation was the problem was **wrong**: low-confidence pixels are not
sparse signal, they are noise that corrupts the statistics. 1.16M boundary pairs is a large
sample; coverage was never the constraint.

### 3.1 `background` is unmineable at *any* τ, including τ=0 ⭐

| | GT | mined τ=0 | mined τ=0.5 |
|---|---|---|---|
| background share of counted boundary | **40.8%** | **2.7%** | 0.4% |

τ=0 means *no threshold at all* — pure argmax — and background still reaches only 2.7%. This was
never about the threshold: `P_final = P_fused · S_pres` with median `S_pres(background) = 0.022`
(§9.2b) means background loses the **argmax** too. It is invisible to SAM 3 by construction,
LoveDA's background being a catch-all rather than a visual concept.

**Method consequence: `background` is not a class in the prior.** It is the "none of the above"
state — the thing the method converts *out of*. Everything below drops it from both sides.

### 3.2 Losing background *rewires* the graph rather than shrinking it

At τ=0.5 with background included, the mined matrix ranked pairs no better than chance
(Spearman **−0.110**, 18/42 sign flips). `building–road` is **−3.15** in GT — they avoid, because
LoveDA labels the pavement between them `background` — and flips to **+0.86** once that
intermediary is gone. Dropping background from *both* sides lifts agreement to **+0.311**.

---

## 4. The gates

### 4.1 Gate 1 — circularity: **passes**, and better than the flip count suggests ✅

At mining τ=0.70, background dropped: **Spearman(PMI_pred, PMI_gt) = +0.757**, 6/30 sign flips.

Every flip is on a pair GT calls ≈ chance:

| flipped pair | GT | mined |
|---|---|---|
| road–barren | +0.17 | −0.33 |
| road–agriculture | +0.15 | −0.84 |
| water–barren | +0.39 | −2.06 |

Max \|GT\| among flips: **0.39**. All four pairs GT calls strong (\|PMI\| > 0.5) survive with the
correct sign — building–water, road–water, barren–forest, building–barren. **The mined matrix
gets the confident facts right and is noisy only where GT has nothing to say.**

`Spearman(row error, discard rate) = −0.257` — the error does **not** concentrate on the classes
SAM 3 discards most, so `ANALYSIS §3.2`'s circularity concern is retired with a number.

⚠️ **Magnitudes are inflated 4.22×** (mean \|PMI\| 2.424 mined vs 0.574 GT). Rank agreement
survived validation; absolute bits did not. **Never feed raw mined PMI into a scoring function** —
z-score it.

### 4.2 Gate 2 — **fails, and it fails against ground truth too** ⭐

For the prior to *fix* a confusion, it must call that pair implausible. What the **GT** matrix
says about the baseline's top confusions:

| confusion | px | **GT** `PMI_bnd` | GT would… |
|---|---|---|---|
| forest → agriculture | 23,765,826 | **+0.32** | reinforce |
| water → agriculture | 19,332,270 | **+0.25** | reinforce |
| agriculture → forest | 7,859,015 | **+0.32** | reinforce |
| water → barren | 4,148,715 | **+0.39** | reinforce |

**A perfect matrix would reinforce them too.** This is not a mining defect — no amount of better
mining fixes it. Forest and agriculture genuinely do touch; adjacency and confusability are the
same signal.

**Consequence:** the prior cannot be aimed at the confusions. Its target is the **94% that fell
silent** (mechanism A), not the 6% that is confidently wrong. Consistent with §7.6 — discard beats
confusion 3:1, the dominant error is silence.

---

## 5. ⛔ The ceiling test — the co-occurrence term does not earn its place

`prior_ceiling.py`, 1669 tiles, inference τ=0.5, components ≥64px. For each region assigned to
background: look only at its confident neighbours, apply M, guess, check against GT.

**Reachability first.** 68,842 components; 66,051 have at least one confident real-class
neighbour. **166,050,845 reachable pixels = 51.4% of the 323M residual.** The rest have no seed
and no `M_image` — unreachable by this mechanism at any M (`ANALYSIS §3.5`).

| method | pixel accuracy | component accuracy |
|---|---|---|
| majority class (always `agriculture`) | 38.6% | 27.0% |
| **β=0.00 — pure neighbour vote, M switched off** | **48.4%** | **79.1%** |
| β=0.25 — best mined mixture | **48.6%** | 79.0% |
| β=1.00 — pure co-occurrence, no vote | 20.1% | 7.6% |
| **β=0.50 with the ORACLE GT matrix** | **48.7%** | 78.9% |

**The decisive number is the oracle row.** A *perfect* co-occurrence matrix adds **0.3 points**
over copying the largest neighbour. The mined matrix adds **0.2**. This is not a mining-quality
problem and not a circularity problem — **the co-occurrence term carries almost no information
beyond "what is next to it".**

Pure co-occurrence (β=1.00) scores **20.1%**, well below the 38.6% majority baseline. With the
diagonal zero by construction, it cannot answer "same class as its neighbour" — which is what a
partly-discarded region usually is.

### 5.1 Per class

| class | reachable px | neighbour vote | best mined β |
|---|---|---|---|
| building | 19,509,474 | 86.8% | 86.9% |
| water | 22,541,749 | 69.8% | 70.0% |
| road | 15,621,872 | 58.9% | 58.8% |
| barren | 16,553,290 | 45.1% | 44.7% |
| agriculture | 64,177,329 | 39.0% | 39.5% |
| **forest** | 27,647,131 | **21.9%** | **21.9%** |

No class gains more than 0.5 points. **Forest — the class §7.6 named as the headline
opportunity — gains nothing.**

### 5.2 The one genuinely positive result in this table

Component accuracy **79.1%** against pixel accuracy **48.4%** means small components are labelled
well and large ones badly — and large ones hold the pixels. That is §9.1a's two morphologies
reappearing: thin seams are easy, whole dropped regions are hard.

**Honest ceiling for region-level recovery at τ=0.5: 48.4% correct on 51.4% of the residual
≈ 25.0% of the 323M background-assigned pixels.** For comparison, threshold relaxation to τ=0.1
buys 1 correct per 1.73 wrong = **36.6% precision** (§8.2). **So neighbour-based region
propagation is materially more precise than the τ knob** — that part of the thesis survives. It is
the *co-occurrence* component that does not.

---

## 6. Where this leaves the project

**What survives:**
- The problem framing. 29.68% discarded, three global knobs measured and all three fail (§7, §8.2, §9.2b).
- **Region-level recovery**, at 48.4% precision against threshold relaxation's 36.6%.
- The reachability number — **51.4%** — a real, defensible scope statement.
- `ANALYSIS §4` as a *measurement* of land-cover structure. It is correct; it is simply not
  actionable in this scoring formulation.

**What does not:**
- ⛔ "A corpus-level semantic co-occurrence prior over SAM 3 region proposals" as **the**
  contribution. The oracle test caps it at +0.3 points.

**Open, and cheap to check before deciding** — the aggregate could hide an effect on the subset
where the decision is genuinely hard. `prior_ceiling.py` now stratifies by number of distinct
neighbour classes and by component size, and reports `Δ oracle` per stratum. A region touching
one class has nothing to arbitrate; if M has a real effect it must appear where two or more
classes border the region. **Run this before acting on §5.**

---

## 7. Artefacts

| file | what |
|---|---|
| `~/outputs/week3/M_global_gt.npz` / `M_gt.md` | GT reference, all 7 classes |
| `~/outputs/week3/M_global_gt_nobg.npz` / `M_gt_nobg.md` | GT reference, background dropped — reproduces §4 |
| `~/outputs/week3/M_global_pred_t07.npz` / `M_pred_t07.md` | mined prior, τ=0.70 |
| `~/outputs/week3/mining_tau_sweep{,_high}.md` | §3 |
| `~/outputs/week3/validation_t07.md` | §4 |
| `~/outputs/week3/prior_ceiling.md` | §5 |
