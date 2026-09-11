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

## 4. What this changes

| | |
|---|---|
| ⛔ **lever 3** | closed on Potsdam. **Not** added to the method. |
| ⭐ **the baseline's `max`** | measured to be already correct for 4 of 6 classes — a result *about SegEarth-OV3*, reported whichever way D − C went |
| ⭐ **the things/stuff duality** | confirmed quantitatively, unsupervised, in `car` 2.33 vs `building` 0.53 |
| ⚠️ **the method stays two levers** | which is a *cleaner* story than three, not a worse one |
| ⏳ **still to do** | LoveDA (tests §3's scale hypothesis), then lever 4 — the per-class presence weight |

⚠️ **One dataset.** Every other lever in this project carries three. Lever 3 is closed *on Potsdam*
until LoveDA runs, and LoveDA is the dataset where the residual is 6× larger (29.68% vs 4.68%) and
where `building` is small enough to test §3.

⚠️ A note fired at startup: *"catch-all `clutter` is at mask value 6, not 1 … confirm the config
sets bg_idx=5."* It does — `POTSDAM_RESULTS.md` §Engineering notes. The check cannot read `bg_idx`
off a cache, so it prints a **note** rather than a warning, which is the fix from WEEK3 §11 working
as designed.
