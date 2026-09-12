# Week 3 — can we detect which background pixels are recoverable?

- tiles: **2016**  |  τ: **0.1**
- background-assigned pixels: **29,273,047**
- of those, really a real class (**positive**): **21,977,806**
- really background (**negative**): **7,295,241**
- **base rate: 75.1%** — a rule that fires everywhere scores exactly this precision

## Signal quality

| signal | AUC | best lower bound | precision | recall | kept px |
|---|---|---|---|---|---|
| `conf` | **0.522** | 0.006 | 75.5% | 99.7% | 29,023,100 |
| `conf2` | **0.567** | 0.000 | 75.1% | 100.0% | 29,273,047 |
| `gap` | **0.459** | 0.000 | 75.1% | 100.0% | 29,273,047 |
| `spres_max` | **0.578** | 0.027 | 75.1% | 100.0% | 29,273,019 |
| `spres_arg` | **0.500** | 0.012 | 75.7% | 100.0% | 29,021,347 |
| `fconf` | **0.470** | 0.008 | 75.4% | 99.9% | 29,141,175 |
| `fgap` | **0.479** | 0.000 | 75.1% | 100.0% | 29,273,047 |

AUC 0.50 is a coin flip. Precision must clearly exceed the **75.1%** base rate for the signal to be worth anything — a rule that fires everywhere already achieves that.

## Verdict

⛔ **No cached signal detects recoverability** (best `spres_max`, AUC 0.578). The confidence map cannot tell a suppressed real class from genuine background, so `τ_low` does not exist in this data and ROADMAP Week 6's three-way split is not realisable from `P_final` alone.

**This inverts `ANALYSIS §3.3`.** That section predicted the embedding term would be the weak one, because a region is ambiguous precisely where SAM 3 is unsure. Instead co-occurrence is weak (+0.2 over a neighbour vote) and confidence is blind, which makes **appearance the only remaining candidate** — and the only one that needs a GPU. Test it before re-scoping: pool `F_cond` per region, fit real-class prototypes from confident regions, score background-assigned regions by similarity, and re-run this AUC.

> The two stages are separable and should be reported separately. Labelling is solved — the oracle run shows a plain neighbour vote is worth **+3.47 mIoU** given the right pixels. Everything now rides on detection.