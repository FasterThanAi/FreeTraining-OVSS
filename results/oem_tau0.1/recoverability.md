# Week 3 — can we detect which background pixels are recoverable?

- tiles: **384**  |  τ: **0.1**
- background-assigned pixels: **17,015,313**
- of those, really a real class (**positive**): **14,063,988**
- really background (**negative**): **2,951,325**
- **base rate: 82.7%** — a rule that fires everywhere scores exactly this precision

## Signal quality

| signal | AUC | best lower bound | precision | recall | kept px |
|---|---|---|---|---|---|
| `conf` | **0.794** | 0.029 | 92.6% | 99.3% | 15,086,952 |
| `conf2` | **0.913** | 0.020 | 94.5% | 98.9% | 14,719,192 |
| `gap` | **0.601** | 0.000 | 82.7% | 100.0% | 17,015,313 |
| `spres_max` | **0.703** | 0.775 | 82.7% | 100.0% | 17,007,874 |
| `spres_arg` | **0.681** | 0.055 | 84.1% | 99.7% | 16,664,460 |
| `fconf` | **0.781** | 0.229 | 91.1% | 99.0% | 15,271,955 |
| `fgap` | **0.796** | 0.172 | 90.9% | 98.8% | 15,296,613 |

AUC 0.50 is a coin flip. Precision must clearly exceed the **82.7%** base rate for the signal to be worth anything — a rule that fires everywhere already achieves that.

## Verdict

✅ **`conf2` separates them (AUC 0.913).** A lower bound at **0.020** keeps 99% of the recoverable pixels at 94.5% precision against a 82.7% base rate. This is ROADMAP Week 6's missing `τ_low`, and it is measured rather than hand-tuned. Wire it into the recovery rule as the **detection** stage, with the neighbour vote as the **labelling** stage, and re-run `selective_recovery_miou.py`.

> The two stages are separable and should be reported separately. Labelling is solved — the oracle run shows a plain neighbour vote is worth **+3.47 mIoU** given the right pixels. Everything now rides on detection.