# Week 3 — at what τ should `M_global` be mined?

- tiles: **1669**  |  α: **1.0**
- GT reference: **23,371,493** boundary pixel-pairs

Mining τ and inference τ are independent. Lower τ buys coverage of the adjacency graph at the cost of label purity.

| mining τ | coverage of GT boundary | tiles w/ no boundary | bg share of boundary | ρ all classes | ρ real only | sign flips (real) |
|---|---|---|---|---|---|---|
| 0.00 | 165.6% (38,691,899) | 19 (1.1%) | 2.7% | **+0.209** | **+0.418** | 12/30 |
| 0.10 | 102.5% (23,947,064) | 127 (7.6%) | 0.2% | **+0.057** | **+0.389** | 12/30 |
| 0.20 | 61.0% (14,251,797) | 256 (15.3%) | 0.2% | **-0.027** | **+0.539** | 12/30 |
| 0.30 | 41.8% (9,770,410) | 331 (19.8%) | 0.2% | **-0.030** | **+0.643** | 8/30 |
| 0.40 | 29.7% (6,931,538) | 405 (24.3%) | 0.3% | **-0.023** | **+0.675** | 6/30 |
| 0.50 | 19.6% (4,585,774) | 481 (28.8%) | 0.4% | **-0.110** | **+0.704** | 8/30 |
| 0.70 | 5.0% (1,158,141) | 826 (49.5%) | 0.0% | **+0.506** | **+0.757** | 6/30 |

GT background share of boundary: **40.8%** — the column above shows how much of that survives mining. ρ is Spearman against the GT matrix over class pairs; ≥0.7 is a usable prior, ≤0.35 is not.

## Verdict

⚠️ **Real-class structure is recoverable at τ = 0.70** (ρ = +0.757), but background never becomes mineable at any τ — `P_final(bg) ≤ S_pres(bg) ≈ 0.022`, so it cannot clear a meaningful threshold. **Consequence: background must be handled OUTSIDE the co-occurrence prior** — as an "unknown/none-of-the-above" outcome rather than a class with adjacency statistics. That is a method decision, and this table is the evidence for it.

> Coverage is not the target — fidelity is. A τ that counts more boundary but ranks pairs worse is the wrong choice, so read ρ, not the coverage column.

Usable (ρ ≥ 0.7, real classes): τ=0.50, τ=0.70