# Week 3 — M_global validation

- classes compared: background, building, road, water, barren, forest, agriculture
- pred source: τ=0.50, α=1.0
- tiles: pred 1669, gt 1669

## Gate 1 — circularity (ANALYSIS §3.2)

| statistic | value |
|---|---|
| KL(pred ‖ gt) over the joint boundary distribution | **3.0754 bits** |
| KL(gt ‖ pred) | 25.5859 bits |
| mean \|ΔPMI_bnd\| off-diagonal | **4.250 bits** |
| max \|ΔPMI_bnd\| | 14.549 bits |
| pairs flipping sign | **18 / 42** |
| Spearman(PMI_pred, PMI_gt) over pairs | **-0.110** |

Spearman is the one to read: it asks whether the mined matrix ranks class pairs the same way ground truth does. Rank agreement is what the scoring function actually consumes; absolute bits are not.

### Per-class error vs how much SAM 3 discards that class

| class | mean \|ΔPMI\| in its row | discard rate | boundary share pred / gt |
|---|---|---|---|
| background | 11.020 | — | 0.4% / 40.8% |
| building | 3.726 | 18.7% | 1.5% / 12.7% |
| forest | 3.636 | 34.6% | 17.7% / 10.9% |
| road | 3.242 | 23.2% | 5.4% / 10.6% |
| barren | 3.074 | 25.0% | 22.3% / 4.5% |
| agriculture | 3.060 | 31.9% | 45.5% / 12.3% |
| water | 1.991 | 32.2% | 7.2% / 8.2% |

**Spearman(row error, discard rate) = -0.429** over 6 real classes.

> ✅ **Circularity is not targeted.** Error does not concentrate on the discarded classes, so M is noisy rather than biased against the classes that need it. State this with the number — §3.2 raised the risk, and this retires it.

## Gate 2 — does M predict the baseline's confusions? (§8.1)

For the prior to *fix* a confusion, it must call that pair implausible. A **positive** `PMI_bnd` on a top confusion means M would **reinforce** the baseline's error.

| rank | true → predicted | confusion px | `PMI_bnd` | M would… |
|---|---|---|---|---|
| 1 | forest → agriculture | 23,765,826 | **+0.44** | reinforce it ⛔ |
| 2 | water → agriculture | 19,332,270 | **+0.34** | reinforce it ⛔ |
| 3 | agriculture → barren | 16,105,731 | **+0.47** | reinforce it ⛔ |
| 4 | barren → agriculture | 13,564,250 | **+0.47** | reinforce it ⛔ |
| 5 | agriculture → forest | 7,859,015 | **+0.44** | reinforce it ⛔ |
| 6 | water → barren | 4,148,715 | **-1.73** | suppress it ✅ |
| 7 | road → forest | 2,627,343 | **-0.98** | suppress it ✅ |
| 8 | agriculture → road | 2,139,670 | **-0.01** | say nothing (≈chance) |

**Spearman(confusion count, PMI_bnd) = +0.441** over 30 ordered real-class pairs.

> ⛔ **The pairs the baseline confuses are the pairs M says belong together.** That is expected — things that touch get confused — but it means a co-occurrence prior alone cannot arbitrate them, because adjacency and confusability point the same way. The method needs the *exclusion* half of the signal (negative PMI) and the appearance term to do the discriminating. Do not claim the prior fixes §8.1's top confusions without showing it.

5 of the top 8 confusions would be **reinforced** by M.