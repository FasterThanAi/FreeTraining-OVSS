# M_global — source `pred`  (without background)

- tiles: **1669**  |  τ: **0.7**
- α (Dirichlet): **1.0**
- boundary pixel-pairs counted: **1,158,141**
- tiles with no confident boundary at all: **826** (49.5%)

- pixels the model committed to: **676,423,300** of 1,750,073,344 (**38.7%**) — the rest are below τ and contribute nothing.

## Class share of counted boundary

| class | boundary share | area share |
|---|---|---|
| building | 0.7% | 16.5% |
| road | 3.3% | 10.6% |
| water | 7.0% | 14.8% |
| barren | 30.7% | 7.7% |
| forest | 12.0% | 8.8% |
| agriculture | 46.3% | 41.5% |

A class whose boundary share far exceeds its area share is thin and high-perimeter — exactly the confound `PMI_bnd` exists to remove.

## Signed PMI, boundary marginals (`PMI_bnd`)

| | buildi | road | water | barren | forest | agricu |
|---|---|---|---|---|---|---|
| **building** | . | +2.04 | -6.76 | -5.42 | +1.92 | -2.72 |
| **road** | +2.04 | . | -5.67 | -0.33 | +0.23 | -0.84 |
| **water** | -6.76 | -5.67 | . | -2.06 | -3.17 | +0.34 |
| **barren** | -5.42 | -0.33 | -2.06 | . | -4.11 | +0.42 |
| **forest** | +1.92 | +0.23 | -3.17 | -4.11 | . | +0.33 |
| **agriculture** | -2.72 | -0.84 | +0.34 | +0.42 | +0.33 | . |

Mean |PMI_bnd| off-diagonal: **2.424 bits**

| strongest attractions | | strongest avoidances | |
|---|---|---|---|
| building–road | **+2.04** | building–water | **-6.76** |
| building–forest | **+1.92** | road–water | **-5.67** |
| barren–agriculture | **+0.42** | building–barren | **-5.42** |
| water–agriculture | **+0.34** | barren–forest | **-4.11** |
| forest–agriculture | **+0.33** | water–forest | **-3.17** |

## Discriminability weights — recomputed on `PMI_bnd`

ANALYSIS §4.3 is REFUTED; the old weights were calibrated on `road`, whose row was a perimeter artefact. `w(n) ∝ Var_c[PMI_bnd(n, c)]`.

| neighbour class | row variance | weight (normalised) |
|---|---|---|
| building | 13.282 | 0.351 |
| road | 6.593 | 0.174 |
| water | 6.443 | 0.170 |
| forest | 5.235 | 0.138 |
| barren | 4.871 | 0.129 |
| agriculture | 1.457 | 0.038 |

Low variance = a hub: bordering it barely narrows the vocabulary, so it should contribute little. High variance = exclusive and informative.

## Directed conditional `P(column | neighbour = row)`  (α=1.0)

Counts are symmetric by construction — a shared boundary has no direction. Directedness lives here, in the row normalisation, because the class marginals differ.

| | buildi | road | water | barren | forest | agricu |
|---|---|---|---|---|---|---|
| **building** | . | 0.204 | 0.001 | 0.011 | 0.679 | 0.105 |
| **road** | 0.041 | . | 0.002 | 0.363 | 0.210 | 0.384 |
| **water** | 0.000 | 0.001 | . | 0.110 | 0.020 | 0.870 |
| **barren** | 0.000 | 0.039 | 0.025 | . | 0.010 | 0.925 |
| **forest** | 0.038 | 0.058 | 0.012 | 0.027 | . | 0.866 |
| **agriculture** | 0.002 | 0.028 | 0.132 | 0.614 | 0.225 | . |

Mean |P(c\|n) − P(n\|c)|: **0.237**, max **0.738** — the measured size of the asymmetry a symmetric M could not express.
