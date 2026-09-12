# Reachability across pipelines and datasets

- rows: **5**  |  source: `/home/priyanshu/outputs/week4/reach_rows.json`

Every row was produced by one pass over that pipeline's own cache, at its own published τ, with the gain computed on the same tiles. Nothing is transcribed between tables.

| tag | τ | classes | **reachable (label-free)** | discard rate | **Δ mIoU** | folds + |
|---|---|---|---|---|---|---|
| SAM3/Potsdam | 0.1 | 6 | **93.9%** | 4.7% | **+0.59** ± 0.50 | 4/5 |
| ConInfer/LoveDA | 0.8 | 7 | **91.4%** | 20.7% | **+2.51** ± 0.34 | 5/5 |
| SAM3/LoveDA | 0.5 | 7 | **88.7%** | 29.7% | **+1.18** ± 0.45 | 5/5 |
| SAM3/OEM | 0.1 | 9 | **86.6%** | 3.8% | **+0.16** ± 0.93 | 3/5 |
| ConInfer/OEM | 0.1 | 9 | **0.0%** | 1.8% | **-0.39** ± 1.28 | 1/5 |

⛔ **Rows with an INERT published threshold: `ConInfer/OEM`.**

Reachable share is 0 there *by construction* — τ sits below the score floor, so the threshold never fires on a single pixel and every catch-all assignment is an argmax loss. Such a row is a real finding about that published configuration, but it **cannot carry a correlation**, and a ρ computed with it in is measuring one point. Every ρ below is reported twice: with, and without.


## First: is this just the discard rate again?

§9f already tested the **discard rate** as a label-free predictor of when calibration pays, and found a U-shape on LoveDA and the opposite sign on OpenEarthMap. If reachable share is collinear with it, this experiment is a restatement of a closed negative.

ρ(reachable share, discard rate) = **+0.600**, p = 0.350 *(exact)*

✅ **They come apart.** The two statistics rank the rows differently, so this is not §9f's measurement under another name.

## Against the gain

| predictor | ρ (all 5 rows) | p | ρ (the 4 live rows) |
|---|---|---|---|
| **reachable share** | **+0.700** | 0.233 *(exact)* | **+0.400** |
| discard rate *(§9f, closed negative)* | +0.900 | 0.083 *(exact)* | +0.800 |
| published τ | +0.894 | 0.100 *(exact)* | +0.949 |

"Live rows" excludes any row whose threshold is inert, where reachable share is 0 by construction. ⭐ **The right-hand column is the honest one** — if the correlation only exists in the left, it is one degenerate point.


⚠️ **5 points.** A correlation over 5 rows is an ordering, not a law; the smallest attainable two-sided p is 0.017. What this table can do is say whether the ordering is there and whether the rival explanation is better — it cannot establish the relationship on its own.

## Verdict

⛔ **The discard rate orders the gain better (ρ +0.800) than reachable share does (ρ +0.400).** The reachability framing adds nothing over the statistic §9f already closed. Report as a bounded negative and move on.

⚠️ **The published τ alone orders the gain at ρ +0.949 on the live rows.** A higher τ mechanically discards more *and* makes more of that discard threshold-driven, so part of any reachability result is arithmetic. Say this beside the ρ, and prefer the per-class evidence — which is within a single operating point and therefore free of it.


## Per-class evidence, one row per dataset

Within a dataset the operating point is fixed, so these are free of the τ confound above.

| tag | ρ self-reachable | p | ρ P−R gap *(§9g)* | p |
|---|---|---|---|---|
| SAM3/Potsdam | **-0.100** | 0.950 | -0.600 | 0.350 |
| ConInfer/LoveDA | **+0.200** | 0.714 | +0.143 | 0.803 |
| SAM3/LoveDA | **-0.200** | 0.714 | +0.200 | 0.714 |
| SAM3/OEM | **+0.381** | 0.360 | -0.238 | 0.582 |
| ConInfer/OEM | **— *(undefined: constant)*** | — | -0.762 | 0.037 |

Self-reachability ranks the classes better than the P−R gap in **2 of 4** datasets where both are defined.

