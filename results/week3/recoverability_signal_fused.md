# Week 3 — can we detect which background pixels are recoverable?

- tiles: **1669**  |  τ: **0.5**
- background-assigned pixels: **750,084,573**
- of those, really a real class (**positive**): **323,084,415**
- really background (**negative**): **427,000,158**
- **base rate: 43.1%** — a rule that fires everywhere scores exactly this precision

## Signal quality

| signal | AUC | best lower bound | precision | recall | kept px |
|---|---|---|---|---|---|
| `conf` | **0.582** | 0.025 | 43.3% | 99.6% | 742,485,354 |
| `conf2` | **0.541** | 0.000 | 43.1% | 100.0% | 750,084,573 |
| `gap` | **0.558** | 0.000 | 43.1% | 100.0% | 750,084,573 |
| `spres_max` | **0.434** | 0.000 | 43.1% | 100.0% | 750,084,573 |
| `spres_arg` | **0.520** | 0.029 | 43.1% | 99.9% | 747,943,328 |
| `fconf` | **0.559** | 0.078 | 43.1% | 100.0% | 749,763,663 |
| `fgap` | **0.447** | 0.000 | 43.1% | 100.0% | 750,084,573 |

AUC 0.50 is a coin flip. Precision must clearly exceed the **43.1%** base rate for the signal to be worth anything — a rule that fires everywhere already achieves that.

## Verdict

⛔ **No cached signal detects recoverability** (best `conf`, AUC 0.582). The confidence map cannot tell a suppressed real class from genuine background, so `τ_low` does not exist in this data and ROADMAP Week 6's three-way split is not realisable from `P_final` alone.

**This inverts `ANALYSIS §3.3`.** That section predicted the embedding term would be the weak one, because a region is ambiguous precisely where SAM 3 is unsure. Instead co-occurrence is weak (+0.2 over a neighbour vote) and confidence is blind, which makes **appearance the only remaining candidate** — and the only one that needs a GPU. Test it before re-scoping: pool `F_cond` per region, fit real-class prototypes from confident regions, score background-assigned regions by similarity, and re-run this AUC.

> The two stages are separable and should be reported separately. Labelling is solved — the oracle run shows a plain neighbour vote is worth **+3.47 mIoU** given the right pixels. Everything now rides on detection.