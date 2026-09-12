# Per-class τ across a domain shift — LoveDA rural vs urban

- cache: `/home/priyanshu/outputs/week3_fused/cache` | published τ: **0.5** | fit objective: **`real`**
- rural: **992** tiles | urban: **677** tiles
- calibration size **200 tiles for every arm**, so a difference between arms cannot be a calibration-size effect
- 5 draws per arm; all three arms scored on the **same** held-out tiles within a draw

## 1. Does the method work inside each domain?

5-fold within each domain, calibration and evaluation disjoint, the published-τ baseline recomputed on the same held-out tiles.

| domain | tiles | mean Δ mIoU | sd | worst fold |
|---|---|---|---|---|
| **rural** | 992 | **+2.77** | 0.92 | +1.61 |
| **urban** | 677 | **+0.10** | 0.39 | -0.31 |

### Per-class Δ IoU, by domain

| class | rural | urban |
|---|---|---|
| background *(catch-all)* | **+0.79** | **-3.51** |
| building | **-0.14** | **+0.70** |
| road | **+0.31** | **+0.42** |
| water | **+10.15** | **+1.22** |
| barren | **+0.61** | **+0.48** |
| forest | **+6.95** | **-0.01** |
| agricultural | **+0.73** | **+1.38** |

`rural`: catch-all **+0.79**, real classes **+18.61** in aggregate.

`urban`: catch-all **-3.51**, real classes **+4.18** in aggregate.

## 2. What do the two domains actually want?

One fit per domain on all of its tiles — these are the thresholds, not a held-out score, and they are here to show *whether the domains disagree*.

| class | rural | urban | difference |
|---|---|---|---|
| building | 0.430 | 0.305 | **0.125** |
| road | 0.725 | 0.225 | **0.500** |
| water | 0.170 | 0.115 | **0.055** |
| barren | 0.375 | 0.375 | **0.000** |
| forest | 0.095 | 0.475 | **0.380** |
| agricultural | 0.600 | 0.300 | **0.300** |

Published τ is a single **0.5** for every class and both domains. Mean |difference| between the domains: **0.227**, max **0.500**.

## 3. Transfer — where should the calibration tiles come from?

Each row fits on 200 tiles from the stated source and evaluates on held-out tiles of the target domain. Δ is against the published τ on those same tiles.

| target | matched (own domain) | mismatched (other domain) | pooled (both) |
|---|---|---|---|
| **rural** | **+2.32** ± 0.46 | **-0.40** ± 0.42 | **+0.77** ± 0.68 |
| **urban** | **+0.02** ± 0.16 | **-1.11** ± 0.14 | **-0.32** ± 0.35 |

## Verdict

⛔ **The gain is not uniform across the domains** — it holds on `rural` and not on `urban`. **Any pooled figure is therefore carried by one stratum and must be reported with this breakdown beside it.**

- ✅ `rural`: **+2.77 ± 0.92** (5/5 folds positive, worst +1.61); real classes +18.61, catch-all +0.79.

- ⚠️ `urban`: **+0.10 ± 0.39** (2/5 folds positive, worst -0.31); real classes +4.18, catch-all -3.51. Land cover **does** improve; the catch-all pays for it almost exactly, which is the whole of the flat result. That is the OpenEarthMap artefact (§8.1) with the sign reversed — the same reason full mIoU is a poor metric wherever a catch-all is large.

### Transfer

⛔ **Calibrating on the other domain is worse than not calibrating at all** — `rural` -0.40, `urban` -1.11, every one below the published τ. The fitted thresholds are domain-specific, and applying the wrong domain's is an active harm rather than a smaller benefit. That is the honest scope statement §9b needs.
- fit on `urban` → evaluate on `rural`: **-0.40** against +2.32 matched (-17% retained).
- fit on `rural` → evaluate on `urban`: **-1.11** against +0.02 matched (matched gain too small for a ratio to mean anything).

### Pooling

Drawing the same 200 tiles across both domains gives `rural` **+0.77**, `urban` **-0.32**.
⚠️ **Pooling is not a safe default here** — it keeps only +0.77 of +2.32 on `rural`, -0.32 of +0.02 on `urban`. A calibration set that mixes domains is fitting one threshold to two different optima, which is the same failure as the global τ it replaces, one level up.