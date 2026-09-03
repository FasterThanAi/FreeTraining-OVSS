# Potsdam — the third dataset, and a falsified prediction

**3 Sep 2026** · 2016 val tiles, 512², τ = 0.1 · `clutter` at mask value 6, `bg_idx=5`

Baseline **57.83** against SegEarth-OV3's published **57.8** — the third reproduced
baseline and the tightest of the three (LoveDA −0.02, Potsdam +0.03).

---

## ✅ The residual-size prediction HELD

Committed to git (`32237bd`) from ground-truth masks alone, before any inference.

| | pre-registered | measured |
|---|---|---|
| bracket | 3.78 – 10.88% | ✅ |
| point estimate | ~4.5% | ⭐ **4.68%** |
| vs LoveDA (36.1% → 10.88%) | must be lower | ✅ |
| vs OpenEarthMap (0.84% → 3.78%) | must be higher | ✅ |

Three points, monotone in catch-all share: **0.84% → 3.78%**, **4.29% → 4.68%**,
**36.1% → 10.88%**.

---

## ⛔ The DETECTABILITY prediction FAILED

Committed alongside it: *"detection AUC should be **higher than** 0.622"*, point
estimate ~0.885.

| signal | Potsdam | LoveDA | OpenEarthMap |
|---|---|---|---|
| `conf` | **0.522** | 0.582 | 0.794 |
| `conf2` | **0.567** | 0.541 | 0.913 |
| `spres_max` | **0.578** | 0.434 | 0.703 |
| **best cached** | ⛔ **0.578** | 0.582 | 0.794 |

**Potsdam has a small catch-all (4.29%) and detects like LoveDA (36.1%), not like
OpenEarthMap (0.84%).** Base rate 75.1%, between LoveDA's 43.1% and OEM's 82.7%,
so this is not a base-rate artefact.

The pre-registration named this outcome as disqualifying, in those words:

> ⛔ *"If a dataset near 5% background behaves like LoveDA, or one near 30%
> behaves like OpenEarthMap, the mechanism is wrong and the paper's central claim
> needs rewriting."*

### What must change

§7 claimed **one** variable explains the residual's size *and* its detectability,
on the strength of two datasets where those move together. **At n = 3 they
separate.** The claim splits:

| | status |
|---|---|
| catch-all share → **residual size** | ✅ three datasets, monotone, one **pre-registered and confirmed** |
| catch-all share → **detectability** | ⛔ **falsified across datasets** (0.84% → 0.794, but 4.29% → 0.578) |

⚠️ **This does not overturn §7b.** The vocabulary intervention *within* OpenEarthMap
causally reduced detectability by raising share, with an arity control. Both can be
true: share is **a** cause of detectability within a dataset, and is **not** what
determines it across datasets. Say exactly that, and no more.

⚠️ **Do not repair this by hunting for a variable that fits three points.** Any
story invented now is fitted to the data it must explain, and the whole value of
the pre-registration is that it was not.

---

## The method on a third dataset

| | published τ | fitted | Δ |
|---|---|---|---|
| full mIoU | 57.87 | 58.47 | **+0.60** |
| catch-all-excluded | 66.05 | **67.09** | ⭐ **+1.04** |
| `clutter` | 16.96 | 15.35 | −1.61 |

Five-fold: **+0.59 ± 0.50**, four of five folds positive, but the spread covers
zero — **report it as marginal on full mIoU and positive on land cover**, which is
what both columns say.

| class | Δ IoU |
|---|---|
| **car** | **+3.70** |
| road | +0.69 |
| grass | +0.66 |
| tree | +0.32 |
| building | −0.17 |

⚠️ **`car` carries it, and `tree` does not** — despite `tree` having precision
93.34 / recall 38.63, a **+54.7** gap, larger than LoveDA's `water` (+34.8) that
drove the entire LoveDA result. **The precision–recall asymmetry did not predict
which class would move here.** That weakens §9g's ρ = +0.713 rather than
confirming it, and it should be reported beside that correlation.

⭐ Consistent across all three datasets: `clutter` sits at **16.96** IoU against a
real-class mean of 66.05, depressing Potsdam's published headline by **8.18 points**
before any method is applied. Compare OpenEarthMap's 3.38 and LoveDA's 0.31. The
metric argument gets stronger with every dataset.

---

## Engineering notes

⚠️ **`labels.py` cries wolf on Potsdam.** It warns that the segmentor sends sub-τ
pixels to `bg_idx=0` (`road`) while the catch-all is `clutter` at 6. **False alarm** —
`cfg_potsdam.py` sets `bg_idx=5` explicitly, so they coincide. The check reads the
segmentor's *default*, not the config's override, and should read the actual value.

⚠️ **`tau_cv.py`'s verdict printed LoveDA boilerplate again** ("the single-split
+1.44 was a favourable draw"), for the fourth time on a non-LoveDA cache. The tables
have been right every time; only the generated prose keeps needing a check.
