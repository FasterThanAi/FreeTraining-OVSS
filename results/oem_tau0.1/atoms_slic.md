# Week 3 — atom quality (`slic`)

- tiles: **384**  |  τ: **0.1**  |  min atom: **64px**
- atoms: **32,974**, covering **16,366,310** px
- median atom size: **248** px, p90 **1,403**, max **10,333**

## 1. Purity — can an atom be labelled at all?

Share of an atom's pixels belonging to its own majority GT class, pixel-weighted. 1.00 = perfectly labelable, 0.50 = a coin flip no method can win.

| purity | share of pixels at or below |
|---|---|
| ≤ 0.50 | 8.5% |
| ≤ 0.60 | 18.1% |
| ≤ 0.70 | 27.2% |
| ≤ 0.80 | 36.4% |
| ≤ 0.90 | 48.4% |
| ≤ 0.99 | 71.2% |

**Mean purity (pixel-weighted): 0.829**

## 2. Ceiling — a perfect labeller on these atoms

Give every atom its own majority GT class. That is the hard upper bound for ANY method built on `slic` atoms.

| | |
|---|---|
| oracle-labeller accuracy | **82.9%** |
| pixels it would get right | 13,572,453 of 16,366,310 |

## 3. Detection at region level

Pixel-level AUCs were all ≈0.5. Averaging over thousands of pixels kills independent noise, so a region aggregate can be far sharper — or stay flat, which would mean the signal is genuinely absent.

- atoms whose majority is a real class: **13,381,018** px
- atoms whose majority is background: **2,985,292** px
- base rate: **81.8%**

| region signal | AUC |
|---|---|
| `mean_conf` | **0.798** |
| `max_conf` | **0.768** |
| `size` | **0.203** |

## Verdict

✅ **Region-level detection works** (AUC 0.798) where pixel-level did not. Aggregation was the missing step. Gate recovery on this and re-run `selective_recovery_miou.py`.