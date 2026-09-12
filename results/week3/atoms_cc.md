# Week 3 — atom quality (`cc`)

- tiles: **1669**  |  τ: **0.5**  |  min atom: **64px**
- atoms: **41,952**, covering **747,942,166** px
- median atom size: **179** px, p90 **4,337**, max **1,048,576**

## 1. Purity — can an atom be labelled at all?

Share of an atom's pixels belonging to its own majority GT class, pixel-weighted. 1.00 = perfectly labelable, 0.50 = a coin flip no method can win.

| purity | share of pixels at or below |
|---|---|
| ≤ 0.50 | 12.5% |
| ≤ 0.60 | 28.5% |
| ≤ 0.70 | 45.3% |
| ≤ 0.80 | 61.4% |
| ≤ 0.90 | 78.8% |
| ≤ 0.99 | 94.7% |

**Mean purity (pixel-weighted): 0.728**

## 2. Ceiling — a perfect labeller on these atoms

Give every atom its own majority GT class. That is the hard upper bound for ANY method built on `cc` atoms.

| | |
|---|---|
| oracle-labeller accuracy | **72.8%** |
| pixels it would get right | 544,558,349 of 747,942,166 |

## 3. Detection at region level

Pixel-level AUCs were all ≈0.5. Averaging over thousands of pixels kills independent noise, so a region aggregate can be far sharper — or stay flat, which would mean the signal is genuinely absent.

- atoms whose majority is a real class: **244,840,669** px
- atoms whose majority is background: **503,101,497** px
- base rate: **32.7%**

| region signal | AUC |
|---|---|
| `mean_conf` | **0.519** |
| `max_conf` | **0.456** |
| `size` | **0.595** |

## Verdict

⛔ **The atoms are the bug.** Mean purity 0.728 — a per-atom label is wrong for 27% of its pixels before any scoring happens, and the oracle ceiling is 72.8%. Every negative result so far is consistent with this rather than with absent signal. **Fix atomisation before concluding anything about the method**: run `--atoms slic`, then SAM 3 mask proposals, and re-run the ceiling and detection tests on those.