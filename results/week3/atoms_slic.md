# Week 3 — atom quality (`slic`)

- tiles: **1669**  |  τ: **0.5**  |  min atom: **64px**
- atoms: **506,064**, covering **748,277,942** px
- median atom size: **1,551** px, p90 **2,587**, max **60,740**

## 1. Purity — can an atom be labelled at all?

Share of an atom's pixels belonging to its own majority GT class, pixel-weighted. 1.00 = perfectly labelable, 0.50 = a coin flip no method can win.

| purity | share of pixels at or below |
|---|---|
| ≤ 0.50 | 1.1% |
| ≤ 0.60 | 5.9% |
| ≤ 0.70 | 10.8% |
| ≤ 0.80 | 16.2% |
| ≤ 0.90 | 23.0% |
| ≤ 0.99 | 35.1% |

**Mean purity (pixel-weighted): 0.928**

## 2. Ceiling — a perfect labeller on these atoms

Give every atom its own majority GT class. That is the hard upper bound for ANY method built on `slic` atoms.

| | |
|---|---|
| oracle-labeller accuracy | **92.8%** |
| pixels it would get right | 694,331,523 of 748,277,942 |

## 3. Detection at region level

Pixel-level AUCs were all ≈0.5. Averaging over thousands of pixels kills independent noise, so a region aggregate can be far sharper — or stay flat, which would mean the signal is genuinely absent.

- atoms whose majority is a real class: **316,197,010** px
- atoms whose majority is background: **432,080,932** px
- base rate: **42.3%**

| region signal | AUC |
|---|---|
| `mean_conf` | **0.576** |
| `max_conf` | **0.516** |
| `size` | **0.467** |

## Verdict

⚠️ **Atoms are adequate (purity 0.928) but detection still fails at region level (best AUC 0.576).** That is the cleaner negative: the signal is genuinely absent from SAM 3's scalar outputs, not merely noisy. Appearance features are the only remaining candidate.