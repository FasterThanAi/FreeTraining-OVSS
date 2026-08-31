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
