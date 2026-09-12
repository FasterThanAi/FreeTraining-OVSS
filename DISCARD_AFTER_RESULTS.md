# How much of the residual comes back? ⭐ The missing bridge between motivation and method

**12 Sep 2026, LoveDA, full 1669 tiles, 5-fold held out, τ = 0.5.** `scripts/discard_after.py`.
✅ Accounting identity verified as exact integers at every rung:
`discard_A − discard_X == recovered − newly discarded`.

⛔ **This question had never been asked.** Every discard figure in the project — the 29.68%, the
per-class table, the whole motivation section — is measured at the **published** τ, before either
lever. The paper motivated itself with *"323 million real-class pixels are thrown away"* and never
said how many came back.

---

## 1. The result

| rung | rule | **discarded** | recovered vs A | of those, **correct** | newly discarded |
|---|---|---|---|---|---|
| **A** | published τ | **29.25%** | — | — | — |
| **B** | per-class τ | **26.77%** | 3.67% | **77.7%** | 1.20% |
| **C** | + per-class scale | **25.01%** | **4.65%** | ⭐ **79.2%** | 0.40% |

⚠️ **My prediction was wrong and I am recording it as such.** I expected the total to *barely move*
— that the method redistributes discard rather than reducing it. It reduces it by **4.24 points**,
and the redistribution story only holds for two of the six classes.

---

## 2. ⭐⭐ The number the paper has been missing

`WEEK1 §8.2` established the trivial alternative's cost: lowering τ to 0.1 recovers pixels at
**1.73 wrong for every 1 right** — i.e. **36.6%** of what it recovers is correct.

| | correct | wrong per right |
|---|---|---|
| τ → 0.1 *(the trivial alternative)* | 36.6% | **1.73** |
| ⭐ **per-class τ + scale** | ⭐ **79.2%** | ⭐ **0.26** |

> ⭐⭐ **Reaching the same residual with a per-class rule instead of a global knob is 6.6× more
> accurate per recovered pixel, and recovers at more than twice the hit rate.**

**This is the bridge between the two halves of the paper.** The motivation section proves the
residual is real and that every global knob reaching it costs more than it returns. The method
section proves a per-class rule gains mIoU. **Nothing connected them** — until now the reader had
to take on faith that the gain came from the residual the introduction was about.

---

## 3. Per class — and it splits exactly on which way the threshold moved

| class | discard A | discard C | change | recovered | **correctly** |
|---|---|---|---|---|---|
| ⭐ **water** | 31.9% | **18.6%** | **−13.3** | 13.4% | **95.0%** |
| barren | 24.5% | 19.0% | −5.5 | 5.9% | **88.5%** |
| forest | 34.2% | 29.8% | −4.4 | 4.9% | **92.4%** |
| building | 18.7% | 14.4% | −4.3 | 4.5% | **95.2%** |
| *agricultural* | 31.3% | 30.3% | −1.0 | 1.5% | ⛔ *0.0%* |
| *road* | 23.1% | **23.2%** | **+0.1** | 0.8% | ⛔ *0.0%* |

⭐ **Four classes recover heavily and at 88–95% accuracy.** Those are the four whose fitted τ moved
**down** — the method lowers the bar where the class is already precise, and almost everything it
lets through is right.

⛔ **`road` and `agricultural` recover almost nothing, and none of it is correct.** Those are the
two whose τ moved **up** (`road` 0.5 → 0.675). They are *supposed* to discard more; `road`'s net
change is **+0.1**, i.e. it throws away slightly more than the baseline, deliberately, to raise
precision 69.7 → 73.4. Their 0.8–1.5% of incidental recoveries come from lever 2 changing the
argmax, and land wrong.

> **The method is not a recovery method. It is a per-class decision rule that happens to recover
> where recovery is cheap and to discard harder where it is not.** Four classes down, two classes
> up, net −4.24 points, and the recovered pixels are 79% correct.

---

## 3a. ⭐ Potsdam — it replicates

**12 Sep, 2016 tiles @ τ = 0.1, 5-fold held out.**

| | discard A → C | change | recovered, **correct** | wrong per right |
|---|---|---|---|---|
| **LoveDA** | 29.25% → **25.01%** | **−4.24** | **79.2%** | **0.26** |
| **Potsdam** | 4.69% → **3.64%** | **−1.05** | **64.0%** | **0.56** |

⭐ **The correction recovers, and most of what it recovers is right, on both datasets.** Potsdam's
residual is six times smaller to begin with (4.69% against 29.25%), so a smaller absolute change is
expected; the recovery *quality* is the transferable part and it holds.

⚠️ **The 1.73 comparison is a LoveDA number and does not transfer.** §8.2's τ→0.1 sweep was run on
LoveDA, whose published τ is 0.5. **Potsdam's published τ is already 0.1**, so "lower it to 0.1" is
not an available alternative there and the contrast cannot be drawn. Potsdam's **64.0%** stands on
its own — most of what it recovers is correct — and the 0.26-vs-1.73 line must stay labelled as
LoveDA's.

---

## 4. What to put in the paper

⭐ **Add the 79.2% vs 36.6% comparison to the results section.** It is the single most direct answer
to *"why not just lower the threshold?"*, which is the first question any reviewer asks of this
paper, and it is currently answered only by the negative half (τ→0.1 costs 5.54 mIoU).

⚠️ **Always print `recovered` beside `newly discarded`.** Quoting recovery alone is precisely the
error §8.2 exists to prevent, and the script refuses to report one without the other.

⚠️ **Rates, not absolutes.** These come from a 40,000 px/tile subsample (42.9M real-class pixels
scored). `measure_discard_rate.py` owns the absolute counts at the published τ.
⚠️ **LoveDA only**, and held out 5-fold. Potsdam's `--cache-full` exists; the run is CPU-only.
