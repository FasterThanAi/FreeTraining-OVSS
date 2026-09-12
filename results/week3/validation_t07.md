# Week 3 — M_global validation

- classes compared: building, road, water, barren, forest, agriculture
- pred source: τ=0.70, α=1.0
- tiles: pred 1669, gt 1669

## Gate 1 — circularity (ANALYSIS §3.2)

| statistic | value |
|---|---|
| KL(pred ‖ gt) over the joint boundary distribution | **1.1224 bits** |
| KL(gt ‖ pred) | 1.3696 bits |
| mean \|ΔPMI_bnd\| off-diagonal | **1.945 bits** |
| max \|ΔPMI_bnd\| | 4.750 bits |
| pairs flipping sign | **6 / 30** |
| Spearman(PMI_pred, PMI_gt) over pairs | **+0.757** |

Spearman is the one to read: it asks whether the mined matrix ranks class pairs the same way ground truth does. Rank agreement is what the scoring function actually consumes; absolute bits are not.

### Per-class error vs how much SAM 3 discards that class

| class | mean \|ΔPMI\| in its row | discard rate | boundary share pred / gt |
|---|---|---|---|
| building | 2.929 | 18.7% | 0.7% / 4.5% |
| water | 2.684 | 32.2% | 7.0% / 14.1% |
| barren | 2.214 | 25.0% | 30.7% / 14.7% |
| forest | 1.599 | 34.6% | 12.0% / 20.9% |
| road | 1.456 | 23.2% | 3.3% / 16.4% |
| agriculture | 0.787 | 31.9% | 46.3% / 29.4% |

**Spearman(row error, discard rate) = -0.257** over 6 real classes.

> ✅ **Circularity is not targeted.** Error does not concentrate on the discarded classes, so M is noisy rather than biased against the classes that need it. State this with the number — §3.2 raised the risk, and this retires it.

## Gate 2 — does M predict the baseline's confusions? (§8.1)

For the prior to *fix* a confusion, it must call that pair implausible. A **positive** `PMI_bnd` on a top confusion means M would **reinforce** the baseline's error.

| rank | true → predicted | confusion px | `PMI_bnd` | M would… |
|---|---|---|---|---|
| 1 | forest → agriculture | 23,765,826 | **+0.33** | reinforce it ⛔ |
| 2 | water → agriculture | 19,332,270 | **+0.34** | reinforce it ⛔ |
| 3 | agriculture → barren | 16,105,731 | **+0.42** | reinforce it ⛔ |
| 4 | barren → agriculture | 13,564,250 | **+0.42** | reinforce it ⛔ |
| 5 | agriculture → forest | 7,859,015 | **+0.33** | reinforce it ⛔ |
| 6 | water → barren | 4,148,715 | **-2.06** | suppress it ✅ |
| 7 | road → forest | 2,627,343 | **+0.23** | reinforce it ⛔ |
| 8 | agriculture → road | 2,139,670 | **-0.84** | suppress it ✅ |

**Spearman(confusion count, PMI_bnd) = +0.440** over 30 ordered real-class pairs.

> ⛔ **The pairs the baseline confuses are the pairs M says belong together.** That is expected — things that touch get confused — but it means a co-occurrence prior alone cannot arbitrate them, because adjacency and confusability point the same way. The method needs the *exclusion* half of the signal (negative PMI) and the appearance term to do the discriminating. Do not claim the prior fixes §8.1's top confusions without showing it.

6 of the top 8 confusions would be **reinforced** by M.