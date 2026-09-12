# M_global — source `gt`  (without background)

- tiles: **1669**
- α (Dirichlet): **1.0**
- boundary pixel-pairs counted: **23,371,493**
- tiles with no confident boundary at all: **18** (1.1%)

## Class share of counted boundary

| class | boundary share | area share |
|---|---|---|
| building | 4.5% | 11.3% |
| road | 16.4% | 7.3% |
| water | 14.1% | 18.3% |
| barren | 14.7% | 6.8% |
| forest | 20.9% | 11.5% |
| agriculture | 29.4% | 44.7% |

A class whose boundary share far exceeds its area share is thin and high-perimeter — exactly the confound `PMI_bnd` exists to remove.

## Signed PMI, boundary marginals (`PMI_bnd`)

| | buildi | road | water | barren | forest | agricu |
|---|---|---|---|---|---|---|
| **building** | . | +0.21 | -2.83 | -0.67 | +0.26 | -0.25 |
| **road** | +0.21 | . | -1.88 | +0.17 | +0.06 | +0.15 |
| **water** | -2.83 | -1.88 | . | +0.39 | -0.01 | +0.25 |
| **barren** | -0.67 | +0.17 | +0.39 | . | -1.11 | +0.05 |
| **forest** | +0.26 | +0.06 | -0.01 | -1.11 | . | +0.32 |
| **agriculture** | -0.25 | +0.15 | +0.25 | +0.05 | +0.32 | . |

Mean |PMI_bnd| off-diagonal: **0.574 bits**

| strongest attractions | | strongest avoidances | |
|---|---|---|---|
| water–barren | **+0.39** | building–water | **-2.83** |
| forest–agriculture | **+0.32** | road–water | **-1.88** |
| building–forest | **+0.26** | barren–forest | **-1.11** |
| water–agriculture | **+0.25** | building–barren | **-0.67** |
| building–road | **+0.21** | building–agriculture | **-0.25** |

## Discriminability weights — recomputed on `PMI_bnd`

ANALYSIS §4.3 is REFUTED; the old weights were calibrated on `road`, whose row was a perimeter artefact. `w(n) ∝ Var_c[PMI_bnd(n, c)]`.

| neighbour class | row variance | weight (normalised) |
|---|---|---|
| water | 1.685 | 0.394 |
| building | 1.297 | 0.303 |
| road | 0.661 | 0.155 |
| barren | 0.320 | 0.075 |
| forest | 0.272 | 0.064 |
| agriculture | 0.039 | 0.009 |

Low variance = a hub: bordering it barely narrows the vocabulary, so it should contribute little. High variance = exclusive and informative.

## Directed conditional `P(column | neighbour = row)`  (α=1.0)

Counts are symmetric by construction — a shared boundary has no direction. Directedness lives here, in the row normalisation, because the class marginals differ.

| | buildi | road | water | barren | forest | agricu |
|---|---|---|---|---|---|---|
| **building** | . | 0.237 | 0.025 | 0.116 | 0.313 | 0.310 |
| **road** | 0.065 | . | 0.048 | 0.208 | 0.273 | 0.407 |
| **water** | 0.008 | 0.056 | . | 0.242 | 0.259 | 0.436 |
| **barren** | 0.035 | 0.231 | 0.231 | . | 0.121 | 0.382 |
| **forest** | 0.068 | 0.215 | 0.175 | 0.085 | . | 0.458 |
| **agriculture** | 0.048 | 0.227 | 0.209 | 0.191 | 0.325 | . |

Mean |P(c\|n) − P(n\|c)|: **0.115**, max **0.262** — the measured size of the asymmetry a symmetric M could not express.
