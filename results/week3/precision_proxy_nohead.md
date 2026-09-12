# Cross-head agreement as a label-free precision proxy

- cache: `/home/priyanshu/outputs/week3_fused/cache`  |  1669 tiles  |  calib **200**, held out **1469**  |  published τ **0.5**
- proxies measured on the **calib** tiles, using no labels
- head-separated arrays present: **NO**  ⚠️ re-run `measure_discard_rate.py` with the patched segmentor to record `iconf/ipred/sconf/spred`; without them the cross-head rows below are simply absent and only the weaker proxies are tested

## The classes, and what has to be predicted

| class | precision | recall | P−R gap | oracle τ |
|---|---|---|---|---|
| building | 75.7 | 77.2 | -1.5 | **0.190** |
| road | 68.5 | 70.3 | -1.8 | **0.675** |
| water | 88.5 | 59.9 | +28.6 | **0.175** |
| barren | 55.6 | 57.9 | -2.3 | **0.375** |
| forest | 61.6 | 46.2 | +15.4 | **0.410** |
| agricultural | 66.6 | 66.6 | -0.0 | **0.565** |

## Level 1 — screen: does any proxy rank the classes correctly?

Spearman across the real classes, with a two-sided p from **exact enumeration** of all 720 relabelings. At this n a correlation is a hint, never a result.

| proxy | ρ vs precision | p | ρ vs P−R gap | p | ρ vs oracle τ | p |
|---|---|---|---|---|---|---|
| `mean_conf` ⭐ | +0.943 | 0.017 | +0.314 | 0.564 | -0.371 | 0.497 |
| `mean_margin` | +0.829 | 0.058 | +0.086 | 0.919 | -0.086 | 0.919 |
| `presence` | +0.657 | 0.175 | -0.200 | 0.714 | -0.086 | 0.919 |
| `gate_ratio` ⭐ | +0.943 | 0.017 | +0.314 | 0.564 | -0.371 | 0.497 |
| `argmax_stability` | +0.371 | 0.497 | +0.200 | 0.714 | -0.543 | 0.297 |

## Level 2 — verdict: does it move mIoU on held-out tiles?

Each rule is `τ_c = τ_pub + b · z(proxy_c)`, with the single knob `b` — sign included, so the rule is not told which way the proxy should point — fitted on the calibration tiles and applied to disjoint tiles. Same one-knob budget as `tau_rules.py`, and parity with the baseline, which also tunes its single τ with labels.

| rule | fitted `b` | held-out mIoU | Δ | share of oracle |
|---|---|---|---|---|
| published τ = 0.5 | — | 47.16 | — | — |
| `mean_conf` | -0.080 | 46.98 | **-0.18** | -14% |
| `mean_margin` | -0.040 | 47.13 | **-0.04** | -3% |
| `presence` | -0.330 | 47.58 | **+0.42** | +34% |
| `gate_ratio` | +0.020 | 47.12 | **-0.05** | -4% |
| `argmax_stability` | -0.020 | 47.13 | **-0.04** | -3% |
| _random proxy_ ×200 | _fitted_ | — | _+0.11 ± 0.28_ | _p95 +0.58_ |
| **oracle per-class τ** | _N−1 params_ | 48.40 | **+1.24** | 100% |

⚠️ **A proxy only counts if it beats the random control's 95th percentile (+0.58).** One fitted knob over 6 classes can extract a gain from noise — the same confound that gave a colour-novelty detector AUC 0.966 on random images (§9).

## Verdict

⚠️ **The cross-head proxy was not tested** — `iconf/ipred/sconf/spred` are absent from this cache. Everything above is the weaker set of proxies. Re-run `measure_discard_rate.py` with the patched segmentor (~25 min GPU) before drawing any conclusion about cross-head agreement.

⛔ **No label-free proxy beats the random control.** The best is `presence` at +0.42 against a control p95 of +0.58, and an oracle bound of +1.24.

**This strengthens §9a rather than weakening it.** The impossibility argument previously rested on three rules that describe how the model's scores are *distributed*. Cross-head agreement asks how often the model is *right*, which is the one remaining candidate a reviewer would propose — and it does not work either. Report it as an eliminated alternative, with this table, not as an untested gap.