# What does ρ actually change? — the anatomy of lever 3’s null

- cache `/home/priyanshu/outputs/loveda_heads/cache` | tiles **800** | subsample **40000** px/tile
- ρ as fitted: `background` 1.47, `building` 0.51, `road` 3.83, `water` 1.00, `barren` 0.33, `forest` 0.25, `agricultural` 0.82

`s_c(ρ) = max(a_c·P_sem_c, b_c·P_inst_c)` with `max(a,b)=1`. Every pixel falls in one of three buckets:

| bucket | what happens |
|---|---|
| **untouched** | the winning head’s arm carries multiplier 1 |
| **rescaled** | the winning head’s arm carries the scaled multiplier |
| **flipped** | the *other* head now supplies the score |

⭐ **Lever 2 can reproduce ρ only where every pixel of the class gets the SAME multiplier** — `w_c` scales the whole class at once. So the part ρ contributes that `w_c` cannot is

> **beyond lever 2 = flipped + min(untouched, rescaled)**

⛔ Not "flipped" alone. A class split evenly between untouched and rescaled is getting a pixel-dependent change even though no pixel changed head.

✅ Identity gate **0.00000**. At ρ = 1 the heads reconstruct the cached score, so these shares describe the deployed pipeline.

## All labelled pixels

| class | ρ | inst head wins | untouched | rescaled | flipped | ⭐ beyond lever 2 |
|---|---|---|---|---|---|---|
| background | 1.47 | 0.2% | 0.2% | 99.7% | 0.2% | **0.3%** |
| building | 0.51 | 3.3% | 96.7% | 1.7% | 1.6% | **3.3%** |
| road | 3.83 | 41.8% | 41.8% | 43.5% | 14.8% | **56.5%** |
| water | 1.00 | 25.0% | 100.0% | 0.0% | 0.0% | **0.0%** |
| barren | 0.33 | 1.0% | 99.0% | 0.2% | 0.8% | **1.0%** |
| forest | 0.25 | 0.8% | 99.2% | 0.1% | 0.8% | **0.8%** |
| agricultural | 0.82 | 1.7% | 98.3% | 1.2% | 0.5% | **1.7%** |

## Restricted to pixels where the class WINS the argmax

These are the pixels whose score actually reaches the threshold, so this is the share that can change an output.

| class | pixels won | untouched | rescaled | flipped | ⭐ beyond lever 2 |
|---|---|---|---|---|---|
| background | 1,440,982 | 0.1% | 96.6% | 3.3% | **3.4%** |
| building | 3,499,858 | 96.6% | 1.3% | 2.1% | **3.4%** |
| road | 2,561,866 | 18.9% | 23.2% | 58.0% | **76.8%** |
| water | 2,936,644 | 100.0% | 0.0% | 0.0% | **0.0%** |
| barren | 4,920,787 | 99.1% | 0.2% | 0.6% | **0.9%** |
| forest | 3,146,201 | 99.6% | 0.1% | 0.3% | **0.4%** |
| agricultural | 13,493,662 | 99.4% | 0.4% | 0.2% | **0.6%** |

## Verdict

⭐ **Only 7.3% of the pixels a real class wins get a change that a per-class constant could not have made** (5.3% flip head; the rest is one uniform multiplier). A per-class constant is exactly what lever 2 already fits, so **ρ is largely lever 2 in disguise, and refitting `w` and `τ` on top absorbs it. That is why lever 3 adds nothing.**

Over all labelled pixels of real classes the beyond-lever-2 share is **10.6%** (5.3% of won pixels flip head).

⭐ **One head dominates outright for: building, barren, forest, agricultural** (the instance head wins <5% or >95% of that class’s pixels). For those classes ρ can only be a no-op or a uniform rescale, whichever direction it moves.

⚠️ **This explains a null; it does not rescue one.** The shares are measured at the fitted ρ, which was chosen to maximise the objective — so they describe the best available head assignment, not an arbitrary one.