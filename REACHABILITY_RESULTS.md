# Reachability — the twelfth label-free attempt, and it fails

**4 Sep 2026** · `scripts/reachability.py` · 5 (pipeline, dataset) rows, CPU-only

⛔ **The hypothesis is refuted, both halves.** Recorded because a bounded negative with a
stated mechanism is worth more than a silent dead end, and because one real finding
fell out of it.

---

## The hypothesis

A pixel reaches the catch-all two ways. It falls below the threshold — or the argmax
picks the catch-all outright. **Lowering a threshold cannot change an argmax**, so
anything in the second group is outside what per-class τ can *ever* fix.

    REACHABLE    pred != bg and conf < τ    a threshold move recovers it
    UNREACHABLE  pred == bg                 no τ vector touches it

The prompt was an ordering nobody had written down: sort every (pipeline, dataset) pair
in this project by the operating threshold it ships with, and the calibration gain is
monotone. ⭐ And the reachable share is **label-free** — it needs no ground truth — so if
it explained that ordering it would resurrect the deployment criterion §9f closed.

---

## ⛔ The dataset-level half fails

| tag | τ | **reachable** | discard | **Δ mIoU** |
|---|---|---|---|---|
| SAM3/Potsdam | 0.1 | **93.9%** | 4.7% | +0.59 ± 0.50 |
| ConInfer/LoveDA | 0.8 | 91.4% | 20.7% | **+2.51** ± 0.34 |
| SAM3/LoveDA | 0.5 | 88.7% | 29.7% | +1.18 ± 0.45 |
| SAM3/OEM | 0.1 | 86.6% | 3.8% | +0.16 ± 0.93 |
| *ConInfer/OEM* | *0.1* | *0.0%* ⛔ | *1.8%* | *−0.39 ± 1.28* |

Over the four live rows:

| predictor | ρ vs Δ mIoU |
|---|---|
| **reachable share** *(the hypothesis)* | **+0.400** |
| discard rate *(§9f's closed negative)* | +0.800 |
| published τ | **+0.949** |

**Reachability is the worst of the three.** Potsdam has the *highest* reachable share and
the *third* gain.

⭐ **The sharpest refutation is Potsdam on its own.** Its residual is **98.11% reachable**
— mechanism (B) is 1.25%, so a threshold can touch essentially all of it — and the gain is
**+0.59**. Maximum reachability, small gain. The hypothesis predicts the opposite.

⚠️ **What does order the gain is the published τ (ρ +0.949), and that is close to
arithmetic**: a higher τ discards more, so there is more for a threshold to repair. It is
not a discovery and it is not measured on enough points to be one. Four rows, two of them
sharing a dataset.

---

## ⛔ The per-class half fails

Self-reachability — of class *c*'s discarded pixels, the fraction with `pred == c` and
`conf < τ`, so lowering *c*'s own threshold recovers them **with the right label**.

| dataset | ρ(self-reachable, Δ IoU) | p |
|---|---|---|
| SAM3/Potsdam | −0.100 | 0.950 |
| ConInfer/LoveDA | +0.200 | 0.714 |
| SAM3/LoveDA | **−0.200** | 0.714 |
| SAM3/OEM | +0.381 | 0.360 |

Noise, and the sign is unstable. Two cases make it concrete:

- **LoveDA `water`** gains **+6.78**, the largest by far, on the **second-lowest**
  self-reachability (34.6%). Absolute headroom does not rescue it either: discard ×
  self-reachable is **10.4–13.3% of GT for all six classes** — flat.
- **Potsdam `car`** at **6.9%** self-reachable gains **+3.69**, the largest; **`tree`** at
  **19.2%** gains **+0.32**. Backwards.

⛔ **`tree` was the case this was built to explain** (P−R gap **+54.7**, larger than
LoveDA `water`'s +34.8, and it does not move — WEEK3 §9g). It stays unexplained.
**§9g's precision–recall gap is not displaced, and its anomaly is not repaired.**

---

## ⭐ What did come out of it

**ConInfer's published OpenEarthMap threshold cannot fire.** Mechanism (A) is **exactly
zero** across 384 tiles: the cache's `conf` floor is **0.1042** against a published
`prob_thd` of **0.1**, so the threshold is below the score floor and never fires on a
single pixel. Every catch-all assignment there is an argmax loss.

That reframes our own §7.1a OEM row. We reported per-class τ as "not transferring" there
(−0.39), as though it were a like-for-like test. It is not: **that baseline is an
un-thresholded argmax**, so the fit could only ever *raise* thresholds, never lower them —
a different regime from every other row in the table. See `CONINFER_RESULTS.md`.

⚠️ Also worth one line: on that degenerate cache the **P−R gap runs the other way**
(ρ **−0.762**, p 0.037) against §9g's +0.713. One dataset, and the only cache where the
threshold is inert, so it is a curiosity rather than a counter-example — but it should not
be discovered by a reviewer first.

---

## Bugs this run exposed, both in our own code

⛔ **Spearman broke on ties.** `argsort(argsort(x))` gives tied values distinct ranks in
array order, so a **constant** vector gets ranks 0..n−1 and correlates with whatever it is
paired against. ConInfer/OEM has self-reachable = 0 for every class, and the first run
reported **ρ = +0.214** — that was the class ordering and nothing else. It also inflated
the `published τ` row from a true **+0.949** to **+1.000**. Fixed with tie-averaged ranks
and a guard that returns `nan` when either side is constant, printed as *undefined*.

⛔ **A degenerate row was carrying the entire correlation.** Before the inert row was
excluded, reachable share scored ρ **+1.000**. Excluding it: **+0.400**. The script now
detects inert thresholds, names them, and reports every ρ twice — all rows and live rows —
with the verdict reading the live column.

⚠️ **The stated go/no-go criterion was too loose, and it was mine.** Before the Potsdam run
I wrote that a reachable share above 86.6% would break the τ confound. That checked only
the pairwise τ=0.1 comparison and ignored what the new row does to the *global* ordering.
Potsdam came in at 93.9% — above the stated bar — and the correlation **fell** from +1.000
to +0.400. The script's own verdict, which reads the whole ordering, called it correctly.
**A criterion stated in prose beat by the criterion coded into the test is a reason to
trust the code, not to renegotiate the bar.**

---

## Verdict

⛔ **Reachability is a bounded negative.** It is a worse predictor than the discard rate
§9f already closed, and worse again than simply reading the pipeline's published τ. It does
not explain which classes move, and it does not repair §9g's `tree`.

⭐ **The bound from WEEK3 §9d/§9f is unchanged and now has a twelfth attempt behind it:**
you cannot choose the thresholds without labels, and you cannot predict whether choosing
them will pay without labels. **The ~200-tile calibration cost is irreducible.**

⛔ **Do not go looking for a thirteenth statistic that fits five points.** The same
discipline that governed Potsdam applies here: a story invented now is fitted to the data
it must explain.
