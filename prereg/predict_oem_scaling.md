# Pre-registration — scaling before the argmax on OpenEarthMap

**Written 5 Sep 2026, BEFORE the run.** `git log` is the timestamp. Not edited afterwards.

Predictions by the assistant, with the reasoning that generated each, so a reader can
audit whether the mechanism or a lucky guess produced them. LoveDA is the only dataset
where this rule has been tested; OEM is the first transfer test.

---

## What is already known, and is the basis for these

| | LoveDA | OpenEarthMap |
|---|---|---|
| published τ | 0.5 | **0.1** |
| tiles | 1669 | **384** |
| classes | 7 | **9** |
| catch-all share of GT | 36.1% | **0.84%** |
| catch-all IoU / precision | 45.5 / 56.9 | **17.13 / 17.33** — over-predicted |
| real-class pixels discarded | 29.68% | **3.78%** |
| per-class τ result | +1.16 full, +1.36 excl. | **+0.16 full, +1.75 excl.** |
| catch-all under the τ fit | −0.02 | **−11.04** (17.13 → 5.83) |
| **scale result** | **+1.16 over τ** | **← this run** |

LoveDA's fitted scales: `background` **0.41**, `building` 0.59, `road` 0.98,
`water` **2.55**, `barren` 1.07, `forest` **1.54**, `agricultural` 1.02.

---

## The predictions

**S1 — `background`'s fitted scale will be below 0.7.**
*Why:* it is over-predicted at 17.33% precision, so suppressing its argmax wins is the
single cheapest correction available. LoveDA gave 0.41 for a differently-broken catch-all.
⛔ *Falsified if* the fitted scale is ≥ 0.7.

**S2 — the increment (C − B) on catch-all-EXCLUDED mIoU will exceed +0.30.**
*Why:* 13.4% of OEM's catch-all assignments are argmax losses, which no threshold reaches,
and that is the mass this rule exists to attack.
⛔ *Falsified if* it is ≤ +0.30.

**S3 — the increment on FULL mIoU will be smaller than on catch-all-excluded.**
*Why:* §9h — the catch-all owns 1/9 of OEM's metric and any fit optimising land cover
sacrifices it. Both our SAM 3 fit and ConInfer's drove it to ~5.8 / 6.06 from ~17.
⛔ *Falsified if* full ≥ excluded.

**S4 — the increment on catch-all-excluded mIoU will be SMALLER than LoveDA's +1.36.**
*Why:* OEM's residual is 3.78% against LoveDA's 29.68%. There is simply less to reorder.
⛔ *Falsified if* it is ≥ +1.36.

**S5 — the fitted scales will span at least 2× (max ÷ min ≥ 2).**
*Why:* this is the structural precondition. If OEM's per-class scores are already balanced
against each other, the fit lands near w = 1 everywhere and S2 must fail with it. Stating
it separately means a null result can be attributed to a cause rather than left blank.
⛔ *Falsified if* max ÷ min < 2.

---

## What each outcome means, decided in advance

| outcome | reading |
|---|---|
| S1–S3, S5 hold | the rule transfers; the mechanism is the argmax-loss mass, not a LoveDA quirk |
| S5 fails, S2 fails | **a clean explained null** — OEM's scores are already balanced, so there is nothing to reorder. That is a finding about the dataset, not a failure of the rule |
| S5 holds, S2 fails | the scales move but buy nothing — the rule is LoveDA-specific and the paper must say so |
| S4 fails *(bigger than LoveDA)* | residual size does not govern the gain, and the stated reasoning above is wrong |

⚠️ **384 tiles, five folds — about 77 evaluation tiles each.** Expect a wider spread than
LoveDA's ±0.19. The gate remains `mean − 2·sd > 0` **and** every fold positive; a mean
that clears zero on its own does not count.

⛔ **Whatever comes back, it is recorded as it comes.** The Potsdam rule applies: no
statistic invented after the fact to explain five numbers.
