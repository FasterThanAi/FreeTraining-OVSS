# M_global — source `gt`

- tiles: **1669**
- α (Dirichlet): **1.0**
- boundary pixel-pairs counted: **23,371,493**
- tiles with no confident boundary at all: **18** (1.1%)

## Class share of counted boundary

| class | boundary share | area share |
|---|---|---|
| background | 40.8% | 36.1% |
| building | 12.7% | 7.2% |
| road | 10.6% | 4.7% |
| water | 8.2% | 11.7% |
| barren | 4.5% | 4.4% |
| forest | 10.9% | 7.4% |
| agriculture | 12.3% | 28.6% |

A class whose boundary share far exceeds its area share is thin and high-perimeter — exactly the confound `PMI_bnd` exists to remove.

## Signed PMI, boundary marginals (`PMI_bnd`)

| | backgr | buildi | road | water | barren | forest | agricu |
|---|---|---|---|---|---|---|---|
| **background** | . | +0.82 | +0.43 | +0.37 | -0.40 | +0.29 | +0.08 |
| **building** | +0.82 | . | -3.15 | -6.04 | -2.95 | -2.78 | -2.97 |
| **road** | +0.43 | -3.15 | . | -2.97 | +0.01 | -0.85 | -0.46 |
| **water** | +0.37 | -6.04 | -2.97 | . | +0.37 | -0.79 | -0.22 |
| **barren** | -0.40 | -2.95 | +0.01 | +0.37 | . | -0.96 | +0.52 |
| **forest** | +0.29 | -2.78 | -0.85 | -0.79 | -0.96 | . | +0.02 |
| **agriculture** | +0.08 | -2.97 | -0.46 | -0.22 | +0.52 | +0.02 | . |

Mean |PMI_bnd| off-diagonal: **1.308 bits**

| strongest attractions | | strongest avoidances | |
|---|---|---|---|
| background–building | **+0.82** | building–water | **-6.04** |
| barren–agriculture | **+0.52** | building–road | **-3.15** |
| background–road | **+0.43** | building–agriculture | **-2.97** |
| background–water | **+0.37** | road–water | **-2.97** |
| water–barren | **+0.37** | building–barren | **-2.95** |

## Discriminability weights — recomputed on `PMI_bnd`

ANALYSIS §4.3 is REFUTED; the old weights were calibrated on `road`, whose row was a perimeter artefact. `w(n) ∝ Var_c[PMI_bnd(n, c)]`.

| neighbour class | row variance | weight (normalised) |
|---|---|---|
| water | 5.322 | 0.354 |
| building | 3.963 | 0.264 |
| road | 1.953 | 0.130 |
| barren | 1.380 | 0.092 |
| agriculture | 1.308 | 0.087 |
| forest | 0.965 | 0.064 |
| background | 0.138 | 0.009 |

Low variance = a hub: bordering it barely narrows the vocabulary, so it should contribute little. High variance = exclusive and informative.

## Directed conditional `P(column | neighbour = row)`  (α=1.0)

Counts are symmetric by construction — a shared boundary has no direction. Directedness lives here, in the row normalisation, because the class marginals differ.

| | backgr | buildi | road | water | barren | forest | agricu |
|---|---|---|---|---|---|---|---|
| **background** | . | 0.291 | 0.186 | 0.139 | 0.045 | 0.172 | 0.169 |
| **building** | 0.934 | . | 0.016 | 0.002 | 0.008 | 0.021 | 0.020 |
| **road** | 0.714 | 0.019 | . | 0.014 | 0.059 | 0.078 | 0.116 |
| **water** | 0.685 | 0.002 | 0.018 | . | 0.076 | 0.082 | 0.137 |
| **barren** | 0.400 | 0.021 | 0.139 | 0.138 | . | 0.072 | 0.229 |
| **forest** | 0.645 | 0.024 | 0.076 | 0.062 | 0.030 | . | 0.162 |
| **agriculture** | 0.559 | 0.021 | 0.100 | 0.092 | 0.084 | 0.143 | . |

Mean |P(c\|n) − P(n\|c)|: **0.162**, max **0.644** — the measured size of the asymmetry a symmetric M could not express.
