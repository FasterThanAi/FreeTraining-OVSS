# Glossary

Every term this project uses in a specific way, in one place. Concepts were scattered across
`ANALYSIS.md`, `CLAUDE.md` and the results files; this is where to look them up.

---

## The pipeline

**SAM 3** — the segmentation model. Given an image and a text prompt it produces three things:

| head | output | shape | what it is |
|---|---|---|---|
| presence | `S_pres` | scalar | *does this concept exist in this image at all?* |
| semantic seg | `P_sem` | dense map | pixel-wise score, continuous on amorphous "stuff" |
| transformer decoder | `P_inst`, `s_conf` | N masks + scores | sharp on countable "things", fragments "stuff" |

**Dual-head fusion** — SegEarth-OV3 combines the two mask sources, taking whichever is more
confident per pixel:

```
P_fused = max( P_sem , max_k [ P_inst^k · s_conf^k ] )
```

**Presence gating** — the fused score is then multiplied by the presence scalar:

```
P_final = P_fused · S_pres
```

`S_pres` is **one number per class per image**, so it acts as a *ceiling on every pixel* of that
class in that tile. This is why a bad presence score can veto a whole image.

**τ (tau)** — the confidence threshold. `argmax` picks a class per pixel; if that pixel's
`P_final` is below τ, it is overwritten with `background`. LoveDA ships τ=0.5, OpenEarthMap τ=0.1.
**Each dataset's τ was tuned by the baseline authors using labels** — which is why a method
spending one label-tuned parameter is at parity with them, not cheating.

---

## The problem

**The residual** — real land-cover pixels that the baseline assigns to `background`. On LoveDA:
**323,184,908 pixels, 29.68% of all real-class pixels**. This project exists to recover them.

**`background` share of GT** — what fraction of *ground-truth* pixels are annotated `background`.
**The single most important number in the project.**

| | LoveDA | OpenEarthMap |
|---|---|---|
| share | **36.1%** | **0.84%** |
| meaning | a **catch-all** for anything outside the 6 named classes | a rare marker for genuinely unlabelable bits |

Nearly every difference between the two datasets follows from this one.

**Two discard mechanisms** (`WEEK1_RESULTS §7.7`) — a pixel reaches `background` two ways:

- **(A) threshold** — `conf < τ`. 94.0% of the residual. Reachable by lowering τ.
- **(B) argmax** — `background` won outright at `conf ≥ τ`. 6.0%, all water, 24 tiles.
  **Unreachable by any τ**, because lowering a threshold cannot change an argmax.

**Catastrophic tile** — a tile losing ≥99% of its real-class pixels. LoveDA has 198 at τ=0.5;
OpenEarthMap has **zero**.

---

## Recovery, and its two stages

**Recovery** = **detect** + **label**. They fail independently and it matters which:

| stage | question | LoveDA | OEM |
|---|---|---|---|
| **detect** | *is this background pixel really land cover?* | ⛔ AUC 0.622 | ✅ AUC 0.913 |
| **label** | *which class is it?* | can't get there | ⛔ 30% precision vs an 82.9% ceiling |

**Atom** — the unit a label is assigned to. Not a pixel: a *region*. Two choices were tested:

- **connected components** of the discard mask — sprawl badly; the largest was **an entire tile**
- **SLIC superpixels** — edge-respecting and size-bounded. **This is the settled choice.**

**Purity** — the share of an atom's pixels belonging to its own majority ground-truth class.
1.00 = perfectly labelable, 0.50 = a coin flip no method can win.

**Neighbour vote** — the simplest labelling rule: give a region the label of the confident region
it shares the most boundary with. It is the baseline every fancier method must beat, and mostly
they don't.

---

## Statistics

**PMI (pointwise mutual information)** — how much more (or less) than chance two classes share a
boundary:

```
PMI(i,j) = log2( P_observed(i,j) / P_expected(i,j) )
```

`> 0` they touch more than chance · `< 0` they avoid each other · `≈ 0` indistinguishable from
random. On LoveDA GT, `building–water = −2.83` is the strongest and most reliable fact in the
matrix.

**Boundary vs area marginals** — ⚠️ *the correction that refuted one of our own findings.*
`P_observed` counts **boundary** pixel-pairs. If `P_expected` is built from **area** shares, the
two sides measure different things, and any thin high-perimeter class looks artificially
attractive. Correcting it collapsed the published "road is a hub" result — road's row was its own
perimeter. **Always quote `PMI_bnd`.**

**`M_global`** — the corpus-level co-occurrence matrix; the "rule book". Mined from SAM 3's own
confident predictions, so no labels are read.

---

## Evaluation

**mIoU** — mean Intersection-over-Union across classes. **Mean over classes, not over pixels**,
which is why fixing one pathologically bad class can move it a lot without improving anything else.

**AUC** — for a detection signal, the probability it ranks a true positive above a true negative.
0.5 is a coin flip. **Our empirical floor is ~0.53**, not 0.50, measured on random-colour controls.

**Base rate** — the score a rule gets by firing everywhere. A signal must beat *this*, not 50%.

**Oracle bound** ⭐ — a measurement where one component is granted access to ground truth, to find
its **ceiling**. *"If this were perfect, what would it be worth?"*

| oracle | what is perfect | answer |
|---|---|---|
| `--regions oracle` | the **detector** | +3.62 mIoU (LoveDA) |
| GT matrix vs mined | the **rule book** | +0.3 over a neighbour vote |
| atom ceiling | the **labeller** | 92.8% |
| per-class τ | the **thresholds** | +1.46 (LoveDA) |

**An oracle bound is never a result** — it uses labels you do not have at inference. It is a
*decision tool*: a small bound means don't build the component; a large one means it is worth the
work. Every oracle row in this project is labelled as such.

**Supervision leak** — accidentally using ground truth in a way that flatters the method.
Happened once here: recovery regions were scoped to `gt >= 2`, telling the method which pixels
were worth touching and making it immune to damaging true background — exactly the protection
τ-relaxation does not get. It produced a quotable **+3.47** that was not real. See
`WEEK3_RESULTS §8.3`.

**Validation gate** — a number a change must reproduce or the change is wrong. The main one:
**every instrumented LoveDA run must report 47.37 mIoU and 323,084,415 background-assigned
pixels.** It has caught real bugs.

---

## Vocabulary to get right in the writeup

**Not "unsupervised"** — the class vocabulary is given. Correct terms: *training-free*,
*annotation-free*, *open-vocabulary*. `ANALYSIS.md §3.6`.

**"Assigned to background", not "discarded by τ"** — 6.0% of the residual is mechanism (B), where
background won the argmax rather than the threshold firing. `WEEK1_RESULTS §7.7`.
