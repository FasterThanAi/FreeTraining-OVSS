# Week 3 — ceiling test: what can the prior actually recover?

- tiles: **1669**  |  inference τ: **0.5**  |  min component: **64px**
- pixels assigned to background: **323,084,415**
- components ≥ min size: **68,842**, of which **66,051** have at least one confident real-class neighbour
- **reachable pixels: 166,050,845** (**51.4%** of the residual)

The rest have no confident neighbour to condition on — no seed, empty `M_image`. Unreachable by this mechanism at any M (ANALYSIS §3.5).

| method | pixel accuracy | component accuracy | × majority baseline |
|---|---|---|---|
| majority class | **38.6%** | 27.0% | 1.00× |
| prior (mined M) β=0.00  ← pure neighbour vote, no M | **48.4%** | 79.1% | 1.25× |
| prior (mined M) β=0.10 | **48.5%** | 79.0% | 1.26× |
| prior (mined M) β=0.25 | **48.6%** | 79.0% | 1.26× |
| prior (mined M) β=0.50 | **48.5%** | 78.6% | 1.26× |
| prior (mined M) β=0.75 | **44.1%** | 67.8% | 1.14× |
| prior (mined M) β=1.00  ← pure co-occurrence, no vote | **20.1%** | 7.6% | 0.52× |
| prior (GT M) β=0.00 | **48.4%** | 79.1% | 1.25× |
| prior (GT M) β=0.10 | **48.4%** | 79.1% | 1.25× |
| prior (GT M) β=0.25 | **48.6%** | 79.1% | 1.26× |
| prior (GT M) β=0.50 | **48.7%** | 78.9% | 1.26× |
| prior (GT M) β=0.75 | **45.4%** | 61.4% | 1.17× |
| prior (GT M) β=1.00  ← pure co-occurrence, no vote | **15.7%** | 6.9% | 0.41× |

## Per class — where does M help, if anywhere?

| class | reachable px | majority | neighbour vote (β=0) | best mined β |
|---|---|---|---|---|
| building | 19,509,474 | 0.0% | 86.8% | 86.9% |
| road | 15,621,872 | 0.0% | 58.9% | 58.8% |
| water | 22,541,749 | 0.0% | 69.8% | 70.0% |
| barren | 16,553,290 | 0.0% | 45.1% | 44.7% |
| forest | 27,647,131 | 0.0% | 21.9% | 21.9% |
| agriculture | 64,177,329 | 100.0% | 39.0% | 39.5% |

Best mined variant: **prior (mined M) β=0.25**. The `neighbour vote` column is β=0.00 — the same procedure with M switched off. **Read the two rightmost columns against each other: that difference is what the entire co-occurrence contribution is worth.**


## Where the decision is actually hard

A region touching one class has nothing to arbitrate. `Δ` is the co-occurrence contribution **on that stratum** — best mined β minus β=0. If M has a real effect it must show up here.

| stratum | reachable px | β=0 (vote) | best mined β | **Δ** | oracle GT β | Δ oracle |
|---|---|---|---|---|---|---|
| 1 neighbour class | 70,341,715 | 45.2% | 45.2% (0.00) | **+0.00** | 45.2% (0.00) | +0.00 |
| 2 neighbour classes | 38,532,584 | 51.7% | 51.7% (0.00) | **+0.00** | 51.7% (0.00) | +0.00 |
| 3 neighbour classes | 24,838,750 | 49.7% | 49.7% (0.00) | **+0.00** | 49.7% (0.00) | +0.00 |
| 4 neighbour classes | 15,487,234 | 47.0% | 49.0% (0.25) | **+2.01** | 51.2% (0.75) | +4.22 |
| 5 neighbour classes | 11,207,704 | 51.2% | 52.1% (0.25) | **+0.95** | 57.7% (0.75) | +6.54 |
| 6 neighbour classes | 5,642,858 | 59.0% | 63.2% (0.50) | **+4.23** | 65.6% (0.75) | +6.58 |
| 64–1k px | 13,216,486 | 81.7% | 81.7% (0.00) | **+0.00** | 81.7% (0.00) | +0.00 |
| 1k–10k px | 32,305,186 | 69.1% | 69.2% (0.25) | **+0.05** | 69.1% (0.00) | +0.00 |
| 10k–100k px | 64,406,336 | 44.6% | 45.5% (0.50) | **+0.97** | 45.4% (0.50) | +0.80 |
| >100k px | 56,122,837 | 33.1% | 33.5% (0.75) | **+0.45** | 34.1% (0.75) | +0.99 |

`Δ oracle` is the ceiling on the co-occurrence term for that stratum: what a PERFECT matrix would add. If it is small, better mining cannot rescue it and the term does not belong in the method.


## Verdict

⚠️ **The prior does not beat copying the largest neighbour** (48.6% vs 48.4%). The gain is spatial smoothness, not semantics — and a reviewer will say DenseCRF does that already (ANALYSIS §6, baseline row 4). Either find where M beats the vote (check the per-class table — exclusion-heavy classes like `water` are where it should) or drop M and keep the vote.

> Honest ceiling for a region-level prior at τ=0.5: **48.6% correct on 51.4% of the residual** ≈ **25.0%** of the 323,084,415 background-assigned pixels. Compare against the τ-relaxation baseline: 1 correct per 1.73 wrong (WEEK1_RESULTS §8.2).