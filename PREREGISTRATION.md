# Pre-registered predictions — the vocabulary intervention (ROADMAP §6.1)

**Written and committed 31 Aug 2026, BEFORE the intervention was run.**
`git log --follow PREREGISTRATION.md` is the timestamp, and the run that tests these
predictions appears in later commits. **Nothing below may be edited after the first
result is seen.** If a prediction is wrong, it stays on the page and the outcome is
reported against it.

## Why pre-register

Every reading in this project has been written down before the number that decided it
(`WEEK1_RESULTS.md` §9.2b is the clearest case — both readings were pre-written, the
measured outcome matched *neither*, and that gap became the finding). This is the same
discipline applied to the experiment the paper's central claim now rests on.

It also answers the objection this experiment exists to defeat. `WEEK3_RESULTS.md` §7a
says plainly: *"this is a stratification, not a randomised intervention."* A reviewer
who sees a post-hoc causal claim built on the same data has every reason to discount it.
A prediction committed before the run does not have that problem.

## What is being tested

`WEEK3_RESULTS.md` §7 claims the catch-all class's **share of ground truth** governs the
residual's detectability. §7a supports this by stratification (LoveDA urban vs rural,
where share and confusability dissociate and share wins). §6.1 intervenes on the variable
directly: catch-all share is set by the **vocabulary handed to SAM 3**, so classes are
merged into the catch-all at increasing doses, against an **arity-matched control** that
merges the same classes into a real class instead.

The existing four observational points fit a line at r = −0.928 (r² = 0.861):

| stratum | catch-all share | AUC `conf` |
|---|---|---|
| OpenEarthMap | 0.84% | 0.794 |
| LoveDA urban | 26.0% | 0.730 |
| LoveDA pooled | 36.1% | 0.582 |
| LoveDA rural | 42.9% | 0.524 |

`AUC = −0.00633 · share + 0.8249`. Anchored at OpenEarthMap's own A0 = 0.794, that line
gives the point predictions below.

### Clarification of evaluation procedure — added 31 Aug 2026, still BEFORE the run

OpenEarthMap's classes are chunky (`bareland` 1.3%, `water` 2.4%, `road` 7.1%, `cropland`
11.8%, `building` 17.8%, `tree` 18.8%, `pavement` 19.8%, `grass` 21.0%), so a greedy merge
cannot land on a nominal target. At `--targets 10 25 40` the arms will reach roughly
**10.3%, 39.9% and 58.7%**.

**P5 is therefore evaluated against the line at each arm's ACHIEVED share, not its nominal
target**, using the formula already stated above: `AUC = 0.794 − 0.00633 · (share − 0.84)`.
This changes no prediction — the line, the anchor and the ±0.08 band are unchanged, and the
achieved shares are determined by the data and the greedy rule, not chosen by us. It is
recorded here rather than applied silently after the fact.

Note the largest arm extrapolates **beyond** every observational point (58.7% against
rural's 42.9%), so P5 is weakest there and P1–P4 carry the argument.

---

## Predictions — forward direction (OpenEarthMap, τ = 0.1)

| # | prediction | falsified if |
|---|---|---|
| **P1** | AUC `conf` **falls** from A0 to the largest dose | it rises, or moves < 0.03 |
| **P2** | the fall is **monotone** across A0 → A10 → A25 → A40 | any dose reverses by > 0.02 |
| **P3** | ⭐ the **control** arms move **< 0.05** from A0 | any control moves ≥ 0.05 |
| **P4** | ⭐ dose effect ≥ **2×** control effect | it is not |
| **P5** | point values within **±0.08** of the line: A10 ≈ **0.736**, A25 ≈ **0.641**, A40 ≈ **0.546** | outside that band |
| **P6** | `conf2` falls **more steeply** than `conf` (its dynamic range across the existing datasets is larger: 0.913 → 0.541 against 0.794 → 0.582) | it falls less steeply |
| **P7** | the discard rate **rises** with the dose | it falls or is flat |

**P3 and P4 are the experiment.** P1 and P2 alone are consistent with "fewer classes is an
easier problem"; only the control separates share from arity.

## Prediction — reverse direction (LoveDA, τ = 0.5, `--drop-catchall`)

| # | prediction | falsified if |
|---|---|---|
| **P8** | removing `background` from the vocabulary **raises** AUC `conf` above the pooled 0.582, to **≥ 0.65** | it stays below 0.62 |

A mechanism that survives being pushed in **both** directions is much harder to argue
with than one confirmed only where it was discovered.

---

## What each outcome means, decided now

| outcome | consequence |
|---|---|
| **P1–P4 hold** | ⭐ §7 upgrades from stratification to **intervention**. §7a's concession is replaced by this table, and the mechanism becomes the paper's causal claim. |
| **P1–P2 hold, P3–P4 fail** | The effect is at least partly class count. **Report as inconclusive** — this is exactly the confound the control exists to expose, and §7a's stratification remains the stronger evidence. |
| **P1 fails** | ⛔ The mechanism is **not causal in the direction claimed.** §7 must be rewritten as an association, and the urban/rural result needs a different explanation — most likely a variable those strata differ in besides share and confusability. **This is a publishable outcome and must not be quietly dropped.** |
| **P5 fails but P1–P4 hold** | The direction is right and the *rate* is dataset-specific. Report the ordering, drop the line. |
| **P8 fails** | The mechanism is asymmetric — removing a catch-all does not recover what adding one destroys. Worth reporting as a limit on the mechanism. |

## Known limitations of this design, stated in advance

- The merge sets are chosen **greedily from the smallest real classes upward**, so the
  dose arms disturb the benchmark as little as possible. They are therefore not a random
  sample of possible merges, and a different merge order could give different point values.
  The control shares the same merge set, so the *comparison* is unaffected.
- The control's receiving class `c0` is chosen as the real class whose share is closest to
  the catch-all's. It is never itself merged.
- Merging by channel-max is exactly how the segmentor already handles synonym groups
  (`segearthov3_segmentor.py:304`), so both arm families are faithful to the pipeline.
- Every class is an independent forward pass with its own text prompt, and the only
  cross-class operation is the `argmax` — so dropping a class from the vocabulary is
  **exactly** equivalent to dropping its channel. This is why all arms can read one cached
  score stack. If that equivalence were ever to break (a future segmentor with cross-class
  normalisation), this design becomes invalid and the arms must be re-run on the GPU.

---

# Addendum — 31 Aug 2026, after the A/C run, before the B/D run

**Nothing above has been edited.** This section records how the original predictions
scored, a design flaw the result exposed, and predictions for the new arms — committed
before those arms are run.

## How the original predictions scored (OpenEarthMap, 384 tiles, τ = 0.1)

| | prediction | outcome |
|---|---|---|
| **P1** | AUC `conf` falls | ✅ 0.794 → 0.215 |
| **P2** | monotone across doses | ⛔ A25 0.181 → A40 0.215 reverses |
| **P3** | ⭐ controls move < 0.05 | ✅ **0.000 — every control sits at 0.794** |
| **P4** | ⭐ dose ≥ 2× control | ✅ 0.580 against 0.000 |
| **P5** | within ±0.08 of the line at achieved share | ⛔ off by 0.527 / 0.368 / 0.216 |
| **P6** | `conf2` falls more steeply than `conf` | ⛔ as written; ✅ direction-agnostic — the prediction was under-specified |
| **P7** | discard rises with the dose | ✅ 3.78 → 7.67 → 25.33 → 43.35 |

## ⚠️ What P1 missed, and why the scoring column changed

**AUC is symmetric: `AUC(−score) = 1 − AUC(score)`.** An AUC of 0.208 is a detector of
strength 0.792 with its sign flipped, *not* an absent signal. P1 asked only whether AUC
falls, so a signal that **inverted** scored as one that was destroyed. Scored
direction-agnostically as `det = max(AUC, 1−AUC)`:

| arm | share | `conf` det | `conf2` det |
|---|---|---|---|
| A0 | 0.84% | 0.794 | 0.913 |
| A10 | 10.19% | 0.792 | 0.625 |
| A25 | 39.57% | 0.819 | 0.548 |
| A40 | 58.20% | 0.785 | 0.552 |
| C40 | 0.84% | 0.794 | 0.896 |

**`conf` is flat — share does not destroy it, it flips it.** **`conf2`, the runner-up, does
fall to near chance while the control holds at 0.90** — and §7's mechanism is stated about
the runner-up. So the claim survives for the signal it is actually about, and fails for the
signal §7a's monotone table happened to use. Both must be reported.

## ⚠️ The design flaw

The `A` arms raise share by **max-merging channels into the catch-all**, which makes it a
*union of well-detected prompts* — unnaturally competent at its own pixels. That is why
`conf` inverted: the merged catch-all is confidently right about the classes it absorbed.
A real catch-all is the opposite; LoveDA's `background` has median `S_pres` **0.022**.

## The faithful analogue — new arms, predictions committed before the run

`B_k` removes the merged classes from the **vocabulary** entirely and labels their pixels
catch-all, leaving the catch-all its own single weak prompt — LoveDA's actual situation.
`D_k` is its arity-matched control: the same prompts dropped, but the pixels labelled a
real class, so share is untouched.

| # | prediction | falsified if |
|---|---|---|
| **Q1** | ⭐ `conf2` det falls to **≤ 0.65** at the largest B dose | it stays above 0.70 |
| **Q2** | `D` controls stay within **0.08** of A0 on `conf2` det | any control moves ≥ 0.08 |
| **Q3** | ⭐ B dose effect ≥ **2×** D control effect on `conf2` det | it is not |
| **Q4** | `conf` inverts **less** in B than in the matching A arm (the catch-all keeps its weak prompt, so it is not confidently right about absorbed pixels) | B inverts as strongly or more |
| **Q5** | `conf` det in B falls **below A0's 0.794**, to ≤ 0.70 at the largest dose | it stays above 0.75 |

**Q3 is the experiment**, as P4 was. Q4 and Q5 together are the real test of the mechanism:
if a *weak* catch-all absorbing more area degrades the top score rather than inverting it,
the intervention reproduces the observational gradient and §7 is causal as written. If
`conf` stays flat in B as it did in A, then **share does not destroy top-score
detectability at all**, §7a's monotone-in-share table needs another explanation, and that
is what the paper must say.
