# M_global — source `pred`

- tiles: **1669**  |  τ: **0.5**
- α (Dirichlet): **1.0**
- boundary pixel-pairs counted: **4,585,774**
- tiles with no confident boundary at all: **481** (28.8%)

- pixels the model committed to: **974,210,481** of 1,750,073,344 (**55.7%**) — the rest are below τ and contribute nothing.

## Class share of counted boundary

| class | boundary share | area share |
|---|---|---|
| background | 0.4% | 2.1% |
| building | 1.5% | 12.8% |
| road | 5.4% | 8.3% |
| water | 7.2% | 12.5% |
| barren | 22.3% | 8.0% |
| forest | 17.7% | 10.0% |
| agriculture | 45.5% | 46.4% |

A class whose boundary share far exceeds its area share is thin and high-perimeter — exactly the confound `PMI_bnd` exists to remove.

## Signed PMI, boundary marginals (`PMI_bnd`)

| | backgr | buildi | road | water | barren | forest | agricu |
|---|---|---|---|---|---|---|---|
| **background** | . | -9.55 | -11.41 | +3.29 | -13.45 | -13.11 | -14.47 |
| **building** | -9.55 | . | +0.86 | -4.20 | -3.28 | +0.69 | -0.64 |
| **road** | -11.41 | +0.86 | . | -5.23 | -0.76 | -0.98 | -0.01 |
| **water** | +3.29 | -4.20 | -5.23 | . | -1.73 | -3.05 | +0.34 |
| **barren** | -13.45 | -3.28 | -0.76 | -1.73 | . | -3.11 | +0.47 |
| **forest** | -13.11 | +0.69 | -0.98 | -3.05 | -3.11 | . | +0.44 |
| **agriculture** | -14.47 | -0.64 | -0.01 | +0.34 | +0.47 | +0.44 | . |

Mean |PMI_bnd| off-diagonal: **4.336 bits**

| strongest attractions | | strongest avoidances | |
|---|---|---|---|
| background–water | **+3.29** | background–agriculture | **-14.47** |
| building–road | **+0.86** | background–barren | **-13.45** |
| building–forest | **+0.69** | background–forest | **-13.11** |
| barren–agriculture | **+0.47** | background–road | **-11.41** |
| forest–agriculture | **+0.44** | background–building | **-9.55** |

## Discriminability weights — recomputed on `PMI_bnd`

ANALYSIS §4.3 is REFUTED; the old weights were calibrated on `road`, whose row was a perimeter artefact. `w(n) ∝ Var_c[PMI_bnd(n, c)]`.

| neighbour class | row variance | weight (normalised) |
|---|---|---|
| background | 36.683 | 0.247 |
| agriculture | 29.711 | 0.200 |
| forest | 21.926 | 0.148 |
| barren | 20.894 | 0.141 |
| road | 18.103 | 0.122 |
| building | 12.994 | 0.087 |
| water | 8.267 | 0.056 |

Low variance = a hub: bordering it barely narrows the vocabulary, so it should contribute little. High variance = exclusive and informative.

## Directed conditional `P(column | neighbour = row)`  (α=1.0)

Counts are symmetric by construction — a shared boundary has no direction. Directedness lives here, in the row normalisation, because the class marginals differ.

| | backgr | buildi | road | water | barren | forest | agricu |
|---|---|---|---|---|---|---|---|
| **background** | . | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| **building** | 0.000 | . | 0.140 | 0.006 | 0.033 | 0.406 | 0.415 |
| **road** | 0.000 | 0.039 | . | 0.003 | 0.188 | 0.128 | 0.643 |
| **water** | 0.053 | 0.001 | 0.002 | . | 0.096 | 0.030 | 0.817 |
| **barren** | 0.000 | 0.002 | 0.046 | 0.031 | . | 0.029 | 0.892 |
| **forest** | 0.000 | 0.034 | 0.039 | 0.012 | 0.037 | . | 0.877 |
| **agriculture** | 0.000 | 0.014 | 0.077 | 0.129 | 0.439 | 0.341 | . |

Mean |P(c\|n) − P(n\|c)|: **0.211**, max **0.947** — the measured size of the asymmetry a symmetric M could not express.
