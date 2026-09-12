# Lever 3 — per-class head fusion. ⛔ A null, and the pre-registered "most interesting" one

**11 Sep 2026, Potsdam, 2016 tiles @ τ = 0.1.** Predictions committed in
`prereg/predict_head_fusion.md` (`45b7c34`) **before the cache existed**.

---

## 1. The result

✅ **Identity gate 0.00000**, views per tile `[1]`. At ρ = 1 the reconstruction reproduces the
cached `logits` exactly, so ρ ≠ 1 measures the deployed pipeline and not an approximation.

| | mean | sd | folds+ | mean−2sd | gate |
|---|---|---|---|---|---|
| lever 1 — per-class τ | +0.64 | 0.58 | 4/5 | −0.52 | ⛔ |
| lever 2 — per-class scale | **+4.63** | **0.26** | **5/5** | **+4.11** | ✅ |
| **lever 3 — per-class fusion** | **+0.04** | **0.11** | 3/5 | −0.19 | ⛔ |

⛔ **Lever 3 is a null. Per-class head fusion buys nothing on top of per-class τ and per-class
scaling.**

✅ **And the cache validates itself**: levers 1 and 2 reproduce their recorded Potsdam values
(+0.75 and +4.86 ± 0.35) at **+0.64** and **+4.63** through an entirely separate code path, on a
cache written with two extra arrays. Rung A averages **57.94** against the gate's 57.87.

---

## 2. ⭐ But the fitted ρ is a finding, and it was predicted in advance

| class | ρ | reading |
|---|---|---|
| **car** | **2.33** | wants the **instance** head, decisively — 2× the next value |
| road | 1.11 | `max` already right |
| **tree** | **1.00** | `max` already right, in **all five folds** |
| grass | 0.97 | `max` already right |
| clutter | 0.88 | `max` already right |
| **building** | **0.53** | wants the **semantic** head |

**Predictions scored:**

| | prediction | measured | |
|---|---|---|---|
| **H1** | positive, under +0.50, weakest lever | +0.04 ± 0.11 | ⚠️ **half right** — weakest by far, but it is a **null**, not the small positive I claimed |
| **H2** | more classes ρ < 1 than ρ > 1 | 3 vs 2 | ✅ *narrowly; only `building` and `car` are meaningfully off 1* |
| **H3** ⭐ | the countable "thing" class has the **highest** ρ (`car`) | **car 2.33**, highest by 2× | ✅ **decisive** |
| **H4** | ≥ 2 classes keep ρ ≈ 1 in every fold | **4 of 6** | ✅ |
| **H5** ⭐ | `tree` will **not** move | **1.00 in 5/5 folds** | ✅ **exact** |

⚠️ **H1 must be scored honestly as a miss on sign.** I predicted "positive but small". +0.04 ± 0.11
with 3/5 folds is indistinguishable from zero, and calling it a small win would be reading a
number the spread does not support.

> ⭐⭐ **This is the branch the pre-registration named in advance as the most interesting:**
> *"H3 holds but D − C is null — SegEarth-OV3's things/stuff duality is real and measurable in
> the fitted ρ, and correcting it buys nothing because levers 1–2 already collect that mass by
> other means."*

**What the paper can now say, with a number:** SegEarth-OV3's two-head design *is* class-dependent
in the way its authors argue — a per-class fit recovers `car → instance` and `building → semantic`
with no supervision about which is which — **and the hardcoded `max` is nevertheless already the
right fusion**, because the two levers above it reach the same pixels by another route. Four of six
classes do not want to move at all.

---

## 3. ⚠️ `building` went the WRONG way, and that is the most useful thing here

H3's reasoning was *countable thing → instance head*, from `ANALYSIS §4.5` (one LoveDA tile:
`building` returns **14** instance masks, `road` returns **2**). On Potsdam, `car` obeys it and
**`building` does the opposite** — ρ **0.53**, wanting the semantic head.

⭐ **The likely reason is scale, not category.** Potsdam is **5 cm** GSD. A building there covers a
huge contiguous roof area and behaves like *stuff*; a car is genuinely small and discrete. LoveDA is
**30 cm** — buildings are an order of magnitude smaller in pixels, and §4.5's 14-mask observation
was made *there*.

> **Restated hypothesis: the head a class wants is set by its object size in PIXELS, not by whether
> the class is semantically countable.**

⭐ **That is falsifiable, cross-dataset, and cheap**, so it is pre-registered before the LoveDA run
in `prereg/predict_head_fusion_loveda.md`: **LoveDA's `building` should take ρ > 1**, the opposite
sign to Potsdam's 0.53. If it does, object scale explains both; if it also lands below 1, the
scale story is wrong and `building` simply prefers the semantic head everywhere.

---

## 3a. ⭐⭐ LoveDA — the scale hypothesis is REFUTED, and `road` inverts §4.5

**11 Sep, 800-tile `--sample` draw @ τ = 0.5.** Predictions committed in
`prereg/predict_head_fusion_loveda.md` (`935f122`) **before the cache existed**.

✅ **Identity gate 0.00000**, and this one matters: LoveDA has **11 queries for 7 classes**
(`building,house`, `forest,tree`, `barren,bareland,soil`), so it is the first real test of the
query→class collapse in `--cache-heads`. Potsdam's 6 mapped 1:1 and could not exercise it.

| | mean | sd | folds+ | mean−2sd | recorded |
|---|---|---|---|---|---|
| lever 1 — per-class τ | +0.79 | 0.74 | 4/5 | −0.70 | +1.18 ± 0.45 |
| lever 2 — per-class scale | +1.05 | 0.69 | 4/5 | −0.33 | +1.16 ± 0.19 |
| **lever 3 — per-class fusion** | **+0.22** | **0.21** | **5/5** | **−0.19** | — |

⚠️ **Both existing levers are noisier here than on the full split** (sd 0.74 / 0.69 against the
recorded 0.45 / 0.19) because this is an 800-tile draw, not 1669 — disk, not choice. Their means
land within that spread. Read lever 3's increment, not the absolutes.

### ⛔ L1 fails, and it is the cleanest refutation available

| | Potsdam, **5 cm** | LoveDA, **30 cm** | ratio |
|---|---|---|---|
| `building` ρ | **0.53** | **0.51** | **1.04×** |

L1 predicted LoveDA's `building` would take **ρ > 1**, the opposite sign, because at 30 cm a
building is ~36× fewer pixels and should behave like a countable *thing*. It takes **0.51** —
the same value as Potsdam's, across a **6× GSD difference**.

> ⛔ **The object-scale hypothesis is refuted. `building` prefers the semantic head at every
> resolution tested**, and §4.5's "14 instance masks" says nothing about which head wins a pixel.
> Reported as a hypothesis that was committed and lost, per the branch table. **No third
> explanation is offered.**

### ⭐⭐ And `road` inverts §4.5 — the fit tracks mask CONFIDENCE, not mask COUNT

| class | ρ | |
|---|---|---|
| **road** | **3.83** | wants the **instance** head, hardest of any class |
| background | 1.47 | |
| water | 1.00 | `max` already right, all 5 folds |
| agricultural | 0.82 | |
| **building** | **0.51** | wants the **semantic** head |
| barren | 0.33 | |
| **forest** | **0.25** | wants the **semantic** head, hardest of any class |

⛔ **This is the exact inversion of how this project has read `ANALYSIS §4.5` since week one.**
That section is quoted throughout as *`building` returns **14** instance masks and `road` returns
**2***, presented as the canonical case of the instance head failing on stuff. The fit says
`road` wants the instance head most and `building` wants it least.

⭐ **The resolution is in §4.5's own table, in the column nobody quoted:**

| | masks | **scores** | fitted ρ |
|---|---|---|---|
| `building` | **14** | **0.51 – 0.77** | **0.51** → semantic |
| `road` | **2** | **0.81 – 0.85** | **3.83** → instance |

> ⭐⭐ **Road's two instance masks are the most confident in the tile; building's fourteen are the
> weakest. The fusion follows the scores, which is what `max` operates on — not the counts, which
> nothing downstream ever sees.** The project has been quoting the count and reading a
> things/stuff story into it. The score column was recorded in the same table on the same day and
> says the opposite.

⚠️ **This does not overturn the things/stuff duality itself** — fragmentation is real and §4.5
measured it. It overturns the inference from *fragment count* to *which head should win*.

### Predictions scored

| | prediction | measured | |
|---|---|---|---|
| **L1** ⭐ | `building` ρ > 1 on LoveDA | **0.51** (Potsdam 0.53) | ⛔ **refuted** — the whole test, and it lost |
| **L2** | `building` has the highest ρ | it has the **second lowest**; `road` is highest | ⛔ |
| **L3** | road/water/forest/agri stay ≤ 1 | 3 of 4 hold; **`road` 3.83** | ⛔ |
| **L4** | D − C is a null (mean−2sd ≤ 0 **or** < 5/5) | **−0.19**, 5/5 folds | ✅ *by the stated criterion* |
| **L5** | ≥ 3 of 7 classes keep ρ ≈ 1 | **1 of 7** (`water`) | ⛔ |

### ⚠️ L5 failed, so the fit was checked before L1 was read — as the branch table requires

The pre-registration says an L5 failure means *treat the run as suspect and check the fit before
reading L1*. Four checks, all passing, so the suspicion is discharged with evidence rather than
waved off:

| check | result |
|---|---|
| identity gate, incl. the 11→7 query collapse | **0.00000** |
| ρ stable across folds | `forest` 0.25 ×5, `water` 1.00 ×5, `road` 4.00 ×4, `barren` 0.32 ×4 |
| levers 1 and 2 reproduce | +0.79 and +1.05 against +1.18 and +1.16, inside an 800-tile spread |
| unit tests | `test_head_fusion.py` returns ρ = 1 everywhere on a null |

⭐ **So L5's failure is a finding, not a fault: LoveDA wants 6 of 7 classes moved off `max`, where
Potsdam wanted only 2 of 6.** A fit that were wandering would not put `forest` at 0.25 in all five
folds while leaving `water` at exactly 1.00 in all five.

---

## 3b. ⭐ WHY the null — measured, and it is one class

`scripts/head_dominance.py`, LoveDA, at the fitted ρ. Identity gate **0.00000**.

Every pixel of a class falls into one of three buckets under ρ: **untouched** (the winning head's
arm carries multiplier 1), **rescaled** (it carries the scaled multiplier), **flipped** (the other
head now supplies the score). ⭐ **Lever 2 can reproduce ρ only where every pixel gets the *same*
multiplier**, since `w_c` scales the whole class at once. So

> **beyond lever 2 = flipped + min(untouched, rescaled)**

⛔ **Not "flipped" alone** — a class split between untouched and rescaled is getting a
pixel-dependent change even with no pixel changing head. A first version of the script used the
flipped share, and on the synthetic built so that *no* `w` can help it reported **0.0%** and would
have concluded "a rescale in disguise" about the one construction where that is provably false.
Corrected statistic reads **49.9%** there and **0.0%** on the null.

| class | inst head wins | **beyond lever 2** *(won pixels)* | why |
|---|---|---|---|
| ⭐ **road** | **41.8%** | ⭐ **76.8%** | heads genuinely competitive, and the fit used it |
| background | 0.2% | 3.4% | one head dominates |
| building | 3.3% | 3.4% | one head dominates |
| barren | 1.0% | 0.9% | one head dominates |
| agricultural | 1.7% | 0.6% | one head dominates |
| forest | 0.8% | 0.4% | one head dominates |
| water | 25.0% | **0.0%** | ⚠️ heads competitive, but the fit chose ρ = 1 |

**Overall: 7.3%** of the pixels a real class wins get a change a per-class constant could not make.

> ⭐ **For five of six real classes the instance head wins ≤ 3.3% of pixels, so `max` is already
> the semantic head and ρ can only be a no-op or a uniform rescale — which is lever 2's family.
> The fusion choice is structurally empty for them.**

⭐⭐ **`road` is the exception and it proves the rule.** Its two heads are genuinely competitive
(instance wins 41.8%), **58.0%** of its won pixels flip head, and **76.8%** of the change is
beyond anything lever 2 can express — the fit pinned it at ρ 3.83. It had real room and used it.
**And one class out of seven cannot move an unweighted mean.** That is the whole null: the fusion
choice matters for exactly one LoveDA class, and mIoU divides by 7.

⚠️ **`water`'s 0.0% is not evidence of no room.** Its heads *are* competitive (25.0%), and the
fit chose ρ = 1.00, at which nothing changes by construction. The shares are measured **at the
fitted ρ**, so they describe the best available head assignment, not the available headroom.

⚠️ This explains a null; it does not rescue one.

---

## 4. What this changes

| | |
|---|---|
| ⛔ **lever 3 is NOT adopted** | Potsdam **+0.04 ± 0.11** (3/5), LoveDA **+0.22 ± 0.21** (5/5, mean−2sd −0.19). Neither clears the project gate on two datasets. **The method stays two levers.** |
| ⭐ **the baseline's `max`** | already correct for 4 of 6 Potsdam classes — but only **1 of 7** on LoveDA. A result *about SegEarth-OV3*, and the two datasets disagree about how wrong it is |
| ⭐ **the duality is real and measurable** | recovered unsupervised: Potsdam `car` 2.33 vs `building` 0.53; LoveDA `road` 3.83 vs `forest` 0.25 |
| ⛔ **object scale does NOT decide it** | `building` takes **0.53 at 5 cm and 0.51 at 30 cm** — a 1.04× ratio across a 6× GSD difference. Committed prediction, refuted |
| ⛔ **§4.5's reading is corrected** | the fit tracks instance-mask **confidence**, not **count**. `road` (2 masks @ 0.81–0.85) → ρ 3.83; `building` (14 @ 0.51–0.77) → ρ 0.51 |
| ✅ **the query→class collapse is validated** | LoveDA's 11 prompts → 7 classes, identity gate **0.00000** |
| ⏳ **next** | lever 4 — the per-class presence weight. CPU only, full splits, caches already exist |

⚠️ **One dataset.** Every other lever in this project carries three. Lever 3 is closed *on Potsdam*
until LoveDA runs, and LoveDA is the dataset where the residual is 6× larger (29.68% vs 4.68%) and
where `building` is small enough to test §3.

⚠️ A note fired at startup: *"catch-all `clutter` is at mask value 6, not 1 … confirm the config
sets bg_idx=5."* It does — `POTSDAM_RESULTS.md` §Engineering notes. The check cannot read `bg_idx`
off a cache, so it prints a **note** rather than a warning, which is the fix from WEEK3 §11 working
as designed.
