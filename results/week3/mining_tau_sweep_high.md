# Week 3 — at what τ should `M_global` be mined?

- tiles: **1669**  |  α: **1.0**
- GT reference: **23,371,493** boundary pixel-pairs

Mining τ and inference τ are independent. Lower τ buys coverage of the adjacency graph at the cost of label purity.

| mining τ | coverage of GT boundary | tiles w/ no boundary | bg share of boundary | ρ all classes | ρ real only | sign flips (real) |
|---|---|---|---|---|---|---|
| 0.50 | 19.6% (4,585,774) | 481 (28.8%) | 0.4% | **-0.110** | **+0.704** | 8/30 |
| 0.60 | 11.3% (2,639,979) | 613 (36.7%) | 0.4% | **-0.091** | **+0.732** | 8/30 |
| 0.70 | 5.0% (1,158,141) | 826 (49.5%) | 0.0% | **+0.506** | **+0.757** | 6/30 |
| 0.80 | 1.2% (281,462) | 1167 (69.9%) | 0.0% | **+0.504** | **+0.693** | 6/30 |
| 0.90 | 0.0% (10,800) | 1555 (93.2%) | 0.0% | **+0.543** | **+0.543** | 8/30 |

GT background share of boundary: **40.8%** — the column above shows how much of that survives mining. ρ is Spearman against the GT matrix over class pairs; ≥0.7 is a usable prior, ≤0.35 is not.

## Verdict

⚠️ **Real-class structure is recoverable at τ = 0.70** (ρ = +0.757), but background never becomes mineable at any τ — `P_final(bg) ≤ S_pres(bg) ≈ 0.022`, so it cannot clear a meaningful threshold. **Consequence: background must be handled OUTSIDE the co-occurrence prior** — as an "unknown/none-of-the-above" outcome rather than a class with adjacency statistics. That is a method decision, and this table is the evidence for it.

> Coverage is not the target — fidelity is. A τ that counts more boundary but ranks pairs worse is the wrong choice, so read ρ, not the coverage column.

Usable (ρ ≥ 0.7, real classes): τ=0.50, τ=0.60, τ=0.70