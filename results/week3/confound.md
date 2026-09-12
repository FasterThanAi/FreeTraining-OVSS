# Share or confusability? — cache

- cache: `/home/priyanshu/outputs/week3_fused/cache`  |  τ = **0.5**  |  catch-all: **`background`**

`share` is the catch-all's share of ground truth. **`conf_out` is confusability, measured**: the percentage of true catch-all pixels the model gives a real class — high means the catch-all looks like real land cover. `base` is the detection base rate.

| group | tiles | share % | **conf_out %** | conf_in % | discard % | base % | AUC `conf` | AUC `conf2` | AUC `gap` |
|---|---|---|---|---|---|---|---|---|---|
| **all** | 1669 | 36.1 | **30.6** | 43.1 | 29.7 | 43.1 | 0.582 | 0.541 | 0.558 |
| **rural** | 992 | 42.9 | **25.5** | 41.3 | 39.3 | 41.3 | 0.524 | 0.495 | 0.520 |
| **urban** | 677 | 26.0 | **43.3** | 48.2 | 18.5 | 48.2 | 0.730 | 0.648 | 0.661 |

## Verdict

Each explanation predicts which stratum detects **worse**. Because the two variables dissociate here, they name different strata, so at most one can be right.

| | predicts worse detection in | observed |
|---|---|---|
| **share** — more catch-all, less signal | `rural` (42.9% of GT) | ✅ **correct** |
| **confusability** — catch-all resembles land cover | `urban` (43.3% escapes to real classes) | ⛔ wrong |

Detection is worse in `rural` (0.524 vs 0.730).

⭐ **SHARE is supported; CONFUSABILITY is refuted as the driver.** Detection is worse in the higher-share stratum even though the *lower*-share stratum is the more confusable one — so a catch-all that resembles land cover is not what destroys the signal, its sheer prevalence is. §7 stands as written, and now rests on a comparison where the rival explanation predicted the opposite.

⚠️ This is a stratification, not a randomised intervention — urban and rural differ in more than these two variables (class mix, object scale, scene density) — so it constrains the explanation rather than proving it. State that beside the result.