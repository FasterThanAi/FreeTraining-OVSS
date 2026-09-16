# DLRSD — the fifth dataset, and the one with NO catch-all

**15 Sep 2026.** 2100 images, 256x256, 17 classes, 21 UC Merced scene categories of
100. Pre-registered in `prereg/predict_dlrsd.md` (`5081937`) **before any inference
and before a config existed**, because SegEarth-OV3 publishes no DLRSD row and ships
no config — so the vocabulary, the threshold, the folds and the discard target were
all ours to choose, and every one of them could have been tuned after the fact.

---

## 1. ✅✅ VERIFIED END-TO-END — predicted 44.45, `eval.py` measured 44.42

| rung | predicted from the cache | **`eval.py`** | Δ |
|---|---|---|---|
| A — published τ = 0.1 | 37.33 | **37.27** | 0.06 |
| B — per-class τ *(lever 1)* | 39.09 | **39.04** | 0.05 |
| ⭐ **C — + per-class scale *(lever 2)*** | 44.45 | ⭐ **44.42** | ⭐ **0.03** |

**B − A = +1.77 · C − B = +5.38 · total 37.27 → 44.42 = +7.15**, on **1701 held-out
tiles**, with both parameter sets fitted on a disjoint 399-tile calibration draw.

⭐ **44 of 51 per-class predictions land within 0.20**; the worst is `sea` at 0.70.
**Fifth dataset validating the cached-histogram instrument** after LoveDA, Potsdam,
ConInfer and UAVid.

⭐ Configs were written by `reorder_deploy.py --cfg-out` straight from the fit, never
transcribed — the segmentor's length check cannot catch a **permuted** vector, and a
permuted vector produces a perfectly plausible table several points low.

---

## 2. ⭐⭐ WHY THIS DATASET IS WORTH MORE THAN A FIFTH ROW

**Every pixel carries a real class.** 17 classes, all of them real things, index 0
absent from all 2100 label files, and the total closes exactly:

    2100 x 256 x 256 = 137,625,600 = the sum of the 17 class counts, difference ZERO

⭐⭐ **So DLRSD sits at 0.00% catch-all share — a new extreme below OpenEarthMap's
0.84% — and three standing objections to this project's numbers cannot be raised:**

| objection | why it cannot apply |
|---|---|
| ⛔ *"the gain is a repaired catch-all"* | **there is none.** Full mIoU **is** catch-all-excluded mIoU, by construction. This qualified OEM's +2.28 (110% `background`), ConInfer's +2.51 (a third), and Potsdam's baseline (depressed **8.18** points by `clutter`) |
| ⛔ *"you inherited a bad global threshold"* | the oracle's **best possible** single τ is worth **+0.04**; per-class is worth **+2.99** |
| ⛔ *"the vocabulary was tuned"* | DLRSD's 17 published names, verbatim, committed before any run |

⚠️ **And one objection this dataset invites instead, which the paper must state
rather than wait for: there is no published baseline to reproduce.** Every other
dataset here is gated against a released number (LoveDA −0.02, Potsdam +0.03, UAVid
+2.16 and reported as unexplained). The substitutes are the pixel-exact accounting
above and the category fingerprint in §3, and **both are weaker than a gate.**

### The plausibility anchor, since a gate is unavailable

OVRSISBench (arXiv:2604.15652) reports DLRSD for training-free methods:

| method | DLRSD mIoU |
|---|---|
| ClearCLIP | 14.80 |
| SCLIP | 20.17 |
| SegEarth-OV | 23.76 |
| ProxyCLIP | 24.03 |
| Trident | 26.31 |
| ⭐ **ours, SAM 3 baseline** | **37.89** |
| ⭐ **ours, both levers** | ⭐ **44.42** |

⚠️ **Not a reproduction gate** — different backbone generation (CLIP at 384² against
SAM 3 at 1008²), their DLRSD taxonomy is unstated, and SegEarth-OV3 has no row
there. ⭐ But **+14.13 over SegEarth-OV is the same order as SAM 3's uplift over CLIP
on LoveDA**, so a broken preparation would have landed far outside. ⚠️ **Most of that
margin is the backbone, not our contribution** — the same discipline `CONINFER_RESULTS`
applies to its 11.54.

---

## 3. The unscored sink — a design decision, and it had to be made

⛔ **With no catch-all, `seg_pred[max_vals < τ] = bg_idx` has nowhere to point.**
`cfg_dlrsd.py` sets **`bg_idx=17`** against 17 classes indexed 0–16: an index with no
prompt, no GT pixel and no column in the metric. mmseg histograms predictions over
`[0, num_classes-1]`, so a prediction of 17 counts in no class's predicted area and
no intersection, while the true class still counts it in its own.

> ⭐ **A discarded pixel is a false negative for its true class and a false positive
> for nothing.** Verified against mmseg's own `intersect_and_union` by
> `scripts/test_dlrsd_sink.py`.

⭐ **A consequence that is genuinely different from every other dataset here:
discarding a WRONG pixel is free.** Elsewhere it lands in a scored catch-all and
costs that class's precision.

### ⛔⛔ Two bugs this produced, and the crash was the lucky one

1. **`labels.py` locates the catch-all BY NAME**, found none among the 17, and
   nominated the first class — **`airplane` became the discard target.** The
   pre-registration had *warned* about exactly this and provided no way to act on
   it. Writing the warning is not fixing the bug.
2. `measure_discard_rate.py` then crashed: `IndexError: index 17 is out of bounds
   for axis 1 with size 17`.

⛔ **The crash was fortunate.** One accidental extra column and the run would have
completed and reported every discarded pixel in the dataset as an `airplane` false
positive, in a table that looked entirely reasonable. That is `WEEK3 §11`'s Potsdam
mistake in a new costume, inside the script whose purpose is to read the class list
from the data.

⛔ **And it recurred one layer up, where it did NOT crash.** `from_cache` never
receives `bg_idx` — it is not derivable from the cached arrays — so the oracle sweep
and the first 5-fold both ran to completion with `airplane` as the sink and produced
**complete, plausible, entirely void tables**. ⭐ **The tell was printed and scrolled
past:** `tau_oracle.py`'s cross-check read **34.63** where `measure_discard_rate.py`
had reported **37.89**, in a section that says in as many words that the two must
agree.

**Fixed three ways:** the fallback is now **fatal**, not a warning (a warning at the
top of a long run is not a safeguard); the cache carries a `_meta.json` sidecar with
`bg_idx`; and `labels.py` recognises a sink and widens the predicted axis.

---

## 4. The class ladder is pinned BY THE DATA, not by a legend

⛔ The published list is alphabetical and the labels carry 1..17, so *"index i = the
i-th name"* is the obvious reading — and a guess. ⭐ **DLRSD checks itself**, because
it is built on UC Merced and the scene category is in every filename:

| idx | class | expected category | share of its pixels there |
|---|---|---|---|
| 1 | `airplane` | airplane | **100.0%** |
| 5 | `chaparral` | chaparral | 88.1% |
| 6 | `court` | tenniscourt, baseballdiamond | 99.1% |
| 7 | `dock` | harbor | **100.0%** |
| 8 | `field` | agricultural, golfcourse | 98.1% |
| 10 | `mobile home` | mobilehomepark | 99.5% |
| 13 | `sea` | beach, harbor | **100.0%** |
| 14 | `ship` | harbor | 99.4% |
| 15 | `tanks` | storagetanks | **100.0%** |

✅ **All nine checkable classes concentrate where the mapping says they should**, and
each appears in almost exactly its own category's 100 images. `scripts/dlrsd_class_map.py`.

---

## 5. ⛔ D1 FAILED — and the failure is a limit on our own mechanism

| | |
|---|---|
| predicted | below OpenEarthMap's **3.78%**, point estimate 2.5%, bracket **0.5–4.0%** |
| **measured** | ⛔ **6.16%** |

| dataset | catch-all share | discard @ τ=0.1 |
|---|---|---|
| ⛔ **DLRSD** | **0.00%** | ⛔ **6.16%** |
| OpenEarthMap | 0.84% | 3.78% |
| Potsdam | 4.29% | 4.69% |
| LoveDA | 36.1% | 10.88% |

⭐⭐ **The lowest catch-all share in the project produces the second-highest discard
rate. `WEEK3 §7`'s surviving half — catch-all share predicts residual size — does not
survive extrapolation to 0%.** It was established over 0.84–36.1% and confirmed by a
committed prediction on Potsdam; 0% is outside that range and it breaks there.

⚠️ **Two suspects, and neither is established.** DLRSD has **17 classes** against
OEM's 9, and every tile is **upsampled ~3.9x** to reach SAM 3's 1008² — a regime
nothing else in this project has run. The pre-registration's branch for this outcome
says to name the suspects without claiming to know which, and that is what this is.

### The residual is concentrated, and bimodal

mean **6.16%**, median **0.43%**, max 100%, with **34 tiles (1.6%) above 99%** —
LoveDA's bimodal signature: the baseline works on a tile or collapses on it.

| class | % of its pixels lost | | class | % |
|---|---|---|---|---|
| ⛔ **field** | **30.84%** | | grass | 2.87% |
| ship | 8.83% | | buildings | 2.73% |
| pavement | 7.82% | | court | 2.11% |
| trees | 7.52% | | sand | 1.55% |
| bare soil | 5.57% | | dock | 1.03% |
| chaparral | 3.91% | | water | 0.67% |
| mobile home | 3.40% | | airplane | 0.08% |
| cars / tanks | 3.33 / 3.23% | | sea | 0.01% |

---

## 6. ⭐⭐ ONE THRESHOLD IS WORTH NOTHING; SEVENTEEN ARE WORTH THREE

Oracle bound, full split, thresholds chosen **with** the evaluation labels:

| rung | free parameters | mIoU | Δ |
|---|---|---|---|
| published τ = 0.1 | 0 | 37.89 | — |
| ⛔ **best GLOBAL τ = 0.070** | 1 | 37.93 | **+0.04** |
| ⭐ **best PER-CLASS τ** | 16 | **40.88** | ⭐ **+2.99** |

> ⭐⭐ **Tuning the single threshold perfectly buys +0.04. Splitting it per class buys
> +2.99 — seventy-five times more.** This is the paper's central claim in one table,
> and DLRSD states it more starkly than any other dataset.

⭐ **LoveDA's global-τ row is also +0.04.** Two very different datasets, same answer:
the level is already right, the shape is wrong. And the classes want **opposite**
things — fitted thresholds span **0.000 to 0.915**, the widest range in the project:

| | classes |
|---|---|
| want τ **low** (≤ 0.02) — they fire too rarely | `bare soil`, `field`, `pavement`, `ship`, `trees` |
| want τ **high** (≥ 0.30) — they fire too readily | `airplane`, `dock`, `grass`, `sand`, `sea`, `tanks`, `water` |

✅ **D5 passes decisively.** It predicted per-class would beat the best *fitted*
global by ≥ +0.5. The **oracle** global is an upper bound on any fitted global, and
per-class 5-fold reaches **+2.37** against it — **+2.33**, with no fitting of the
global arm required.

---

## 7. ✅ LEVER 1 — +2.37 ± 0.63, 5/5 folds, stratified

| | mean | sd | folds+ | mean−2sd | gate |
|---|---|---|---|---|---|
| LoveDA | +1.18 | 0.45 | 5/5 | +0.28 | ✅ |
| Potsdam | +0.59 | 0.50 | 4/5 | −0.41 | ⛔ |
| UAVid | +1.34 | 0.39 | 5/5 | +0.56 | ✅ |
| ConInfer (CLIP) | +2.51 | 0.34 | 5/5 | +1.83 | ✅ |
| ⭐ **DLRSD** | ⭐ **+2.37** | **0.63** | **5/5** | ⭐ **+1.11** | ✅ **PASS** |

**Second-largest lever-1 gain in the project**, and it captures **79%** of the +2.99
oracle — close to LoveDA's 81%, against Potsdam's 73% and OpenEarthMap's 11%.

### ⛔ Folds are STRATIFIED, not group-disjoint — and that reverses the default

`--group-re` is mandatory wherever groups exist: on UAVid a frame-level split leaked
**+0.54 mIoU and two folds**. DLRSD has 21 obvious groups, so it looks like the same
case. ⛔ **It is the opposite one.** `dlrsd_class_map.py` measured that **5 classes
live in at most 2 of the 21 categories** — `airplane`, `dock` and `tanks` in exactly
**one** — so holding a category out leaves those classes with **no calibration pixels
at all**. The fit would be asked to set a threshold it cannot see, and the failure
would read as the method's.

⭐ **`--stratify-re` gives every fold exactly 20 of each category** (asserted, not
hoped), against group-disjoint folds seeing only 4–5 categories each. ⛔ **Which
correction to use is a property of the data, not a house style** — group when the
groups are near-duplicates, stratify when they are different subjects no class spans.

### Calibration cost

| tiles | mean Δ | sd | worst draw |
|---|---|---|---|
| 10 | ⛔ −0.93 | 1.58 | −2.73 |
| 25 | ⛔ −0.47 | 1.83 | −2.92 |
| 50 | +0.61 | 1.20 | −1.35 |
| **100** | ✅ **+1.58** | 0.64 | **+0.47** |
| 200 | +1.75 | 0.31 | +1.49 |
| 400 | +2.28 | 0.45 | +1.61 |

⭐ **Every draw is positive from ~100 tiles**, against LoveDA's ~200. ⛔ And n=10 and
n=25 are **negative**, reproducing the shape on every dataset measured.

---

## 8. ⭐⭐ LEVER 2 — +5.84 ± 1.03, and unlike UAVid the gain is BROAD

| rung | | mean | sd | folds+ | mean−2sd |
|---|---|---|---|---|---|
| B − A | per-class τ | +2.36 | 0.56 | 5/5 | +1.24 |
| ⭐ **C − B** | **+ per-class scale** | ⭐ **+5.84** | **1.03** | **5/5** | ⭐ **+3.78** |

**A 38.05 → B 40.41 → C 46.25.** Range +4.14 to +6.85. ✅ Search-subsample gate
**0.009** against a 0.15 bar; all rungs evaluated exactly over every pixel.

| dataset | lever 2 |
|---|---|
| UAVid | +5.89 |
| ⭐ **DLRSD** | ⭐ **+5.84** |
| Potsdam | +4.92 |
| LoveDA | +1.16 |
| ConInfer | −0.10 |

⭐⭐ **THE IMPORTANT DIFFERENCE FROM UAVid: seven classes gain more than 8 IoU each.**
There, `tree`+`vegetation` were **96%** of the gain and the headline could not be
quoted without that caveat. Here the top two are **45%** and the top four **74%** —
still concentrated, but a genuinely multi-class result.

⚠️ **The fold-to-fold scale spread of 33.9% is `dock`, whose IoU moves −0.28.** The
seven classes that carry the result have sd **0.05–0.17**. ⛔ The script's generic
*"this is what overfitting looks like"* warning is the template; the verdict below it
("non-uniqueness, not overfitting") is the correct reading, and the evidence is the
sd column — same as UAVid's `building`.

---

## 9. The deployed run, class by class — 13 improve, 3 lose, 2 are dead

| class | A | B | **C** | total |
|---|---|---|---|---|
| ⭐ **grass** | 25.20 | 26.40 | **52.23** | ⭐ **+27.03** |
| court | 40.67 | 40.67 | 56.43 | +15.76 |
| ⭐ **airplane** | 58.20 | **73.85** | 73.82 | **+15.62** |
| tanks | 23.15 | 21.87 | 35.46 | +12.31 |
| bare soil | 19.21 | 19.37 | 31.10 | +11.89 |
| ship | 15.00 | 15.81 | 26.65 | +11.65 |
| field | 13.91 | 11.47 | 22.05 | +8.14 |
| sand | 30.63 | 37.29 | 37.75 | +7.12 |
| sea | 43.98 | 44.94 | 49.16 | +5.18 |
| dock | 52.73 | 57.71 | 57.67 | +4.94 |
| trees | 63.87 | 66.78 | 67.06 | +3.19 |
| pavement | 70.17 | 71.51 | 72.66 | +2.49 |
| ⛔ `mobile home` | 0.00 | 0.00 | 0.29 | **+0.29 — dead** |
| ⛔ `chaparral` | 0.00 | 0.00 | 0.00 | **+0.00 — dead** |
| ⛔ buildings | 62.00 | 62.39 | 61.92 | **−0.08** |
| ⛔ cars | 64.15 | 64.23 | 62.87 | **−1.28** |
| ⛔ **water** | 50.80 | 49.37 | 48.07 | ⛔ **−2.73** |

⚠️ **`water` loses under BOTH levers** (−1.43 then −1.30). Quote it.

### ⭐⭐ Two checks that answer the obvious objections

**1. `aAcc` rises 58.94 → 65.78, +6.84.** Overall pixel accuracy is **pixel-weighted**,
so a rare class cannot move it. The improvement is real at the pixel level, not an
artefact of averaging 17 classes.

**2. ⭐ Precision AND recall both rise** — what `TTA_RESULTS` calls the signature of
better *decisions* rather than redistribution:

| | A | B | C | total |
|---|---|---|---|---|
| mPrecision | 55.44 | 66.42 | 64.17 | ⭐ **+8.73** |
| mRecall | 52.24 | 48.45 | 54.19 | ⭐ **+1.95** |

⭐ **And the split reproduces UAVid exactly.** Lever 1 **trades** — precision **+10.98**,
recall **−3.79**, because it raises bars. Lever 2 **pays the recall back** (+5.74)
without giving the precision away, because it is not lowering a bar: it hands the
pixel to the class that should have won it.

Visible per class, and it is the whole mechanism in two rows:

| class | precision → | recall → |
|---|---|---|
| **airplane** *(lever 1)* | ⭐ **59.03 → 85.05** | 97.65 → 84.87 |
| **grass** *(lever 2)* | ⭐ **53.53 → 82.46** | ⭐ **32.25 → 58.75** |

⭐ `airplane` fired almost everywhere (**97.65%** recall at **59.03%** precision) and a
threshold of **0.955** fixed it — the single clearest instance of the per-class-τ
argument in the project. ⭐ `grass` gains **25+ points of precision AND recall**,
which no threshold can do.

---

## 10. ⛔⛔ TWO PROMPTS ARE DEAD, AND NEITHER LEVER CAN REACH THEM

`chaparral` and `mobile home` score **0.00 IoU** at the baseline, after lever 1, and
after lever 2. Not low — **zero**. They never win a pixel.

⭐ **`mobile home` receives the largest scale in the whole vector — 2.488, at the grid
ceiling — and still reaches 0.29.** The fit is saturated and wants more.

> ⛔ **Neither lever can rescue a word the model does not understand.** Two of 17
> classes is **11.8% of the metric** contributing nothing, and if they merely
> performed averagely it would be worth roughly **+3.5 mIoU** — more than our whole
> method delivers here.

⚠️ `field` is a third suspect: **16.39% precision** at baseline and **30.84%** of its
pixels discarded, the worst of any class.

⛔ **This is NOT fixed in this result, deliberately.** The pre-registration fixes the
vocabulary as DLRSD's published names, verbatim, *"not negotiable after the fact"*.
Changing a prompt because a result disappointed is exactly what pre-registration
exists to prevent. A vocabulary arm is a **separate pre-registered experiment**, as
`prereg/predict_uavid_vocabulary.md` was.

⚠️ **Also unresolved: the `w` grid is binding.** `mobile home` and `ship` sit at the
2.50 ceiling and `buildings` at the 0.40 floor. A wider grid is a cheap check and has
not been run.

---

## 10b. ⭐⭐ TILE BY TILE — 769 of 2100 get WORSE, and why that is not a defect

`fig_dlrsd_compare.py --all` renders every tile and `tile_delta_report.py` scores
them. **1057 improve, 769 get worse, 274 are unchanged.** A reader is entitled to
ask what a dataset-level gain is worth against 769 worse pictures, and a count
cannot answer it.

| | tiles | mean Δ | median | total mass |
|---|---|---|---|---|
| improved | **1057** | **+6.89** | +3.29 | **+7,283** |
| worse | **769** | **−4.29** | −1.32 | **−3,302** |
| unchanged | 274 | — | — | — |
| **net** | | **+1.90** | **+0.07** | ⭐ **+3,980** |

⭐ **Win mass ÷ loss mass = 2.21x.** Each win is **1.61x larger** than each loss and
there are **1.37x** more of them. ⭐ **And the median tile moves +0.07** — the rule
leaves most scenes alone and acts decisively on a minority, which is what a decision
rule should look like and the opposite of churn.

### ⭐⭐ Two failure shapes, which a mean cannot tell apart

| category | mean Δ | worse | loss shape |
|---|---|---|---|
| ⛔ **agricultural** | **−6.68** | **26**/100 | ⭐ **RARE + severe** |
| river | −2.84 | 65/100 | near-universal + mild |
| ⛔ **parkinglot** | **−1.51** | **89**/100 | ⭐ **near-UNIVERSAL + mild** |
| buildings | −0.98 | 53/100 | mixed |
| mobilehomepark | −0.37 | 66/100 | near-universal + mild |
| *baseballdiamond* | *+13.53* | *4/100* | |
| *golfcourse* | *+9.91* | *15/100* | |
| *forest* | *+8.97* | *5/100* | |

> ⭐⭐ **`agricultural` loses 6.68 across a QUARTER of its tiles; `parkinglot` loses
> 1.51 across NINE TENTHS of them. Same kind of number, different findings, and they
> need different fixes.**

- **`agricultural` is a PROMPT problem.** `field` has the worst precision of any class
  (**16.39%**) and the worst discard rate (**30.84%**), and **18 of the 22 tiles our
  rule discards entirely are agricultural**. A handful of scenes destroyed.
- ⛔ **`parkinglot` is the PRICE OF ONE VECTOR.** `pavement` is fitted at **w = 0.829**
  — down-weighted because it over-predicts across the dataset — which is right on
  average and wrong on tiles that genuinely *are* mostly pavement. **No prompt fixes
  this.** It is the honest cost of fitting one parameter set for 2100 images, and it
  belongs in limitations.

### ⭐ 70% of the damage is DISCARDING, not mislabelling

| | pixels | share of the damage |
|---|---|---|
| ⭐ **newly sent to the sink** | **11,131,177** | ⭐ **70%** |
| were right, became wrong | 4,704,016 | 30% |

11.1M is **8.1%** of the dataset. ⭐ **This is why `aAcc` still rises 6.84 points**: the
pixels the rule discards were mostly wrong already, and it fixes far more elsewhere.
⛔ **It also names the most valuable remaining fix** — the thresholds that are too
aggressive on a minority of scenes, which is `field` again.

### ⭐⭐ What the panels show that the tables cannot — `docs/dlrsd_compare`

⭐⭐ **The discarded regions are ROADS, and the method recovers them as `pavement`.**
On `freeway22` and `freeway58` the baseline's discard mask sits almost exactly on the
road surface — the tile is scored 19.3 and 18.8 while the road itself is thrown away.
Our column fills it correctly and the changed panel is near-solid green: **49,973
pixels fixed against 226 broken**, a 221:1 ratio. ⭐ **And the cause is checkable in
the fitted vector: `pavement` is fitted at τ = 0.000 — never discard it.** This is the
qualitative confirmation of the project's motivating claim, that presence-gated
pipelines discard real land cover and per-class calibration recovers it.

⭐ **A second mechanism is visible in the same panel, and it is the other lever.** In
the baseline, `field` (grey) covers both the grass and the bare soil — the class with
the worst precision in the dataset, **16.39%**, over-firing across two neighbours. In
our column it is gone, replaced by correct `grass` and `bare soil`. That is `field` at
**w = 0.805** losing argmaxes it should never have won. **One tile therefore shows the
two levers doing different jobs: a threshold recovering discarded pixels, and a scale
demoting an over-firing class.**

⛔ **`ship` is pinned at the grid ceiling and still loses.** On `harbor13` the ground
truth is `ship` / `water` / `dock`; both rungs predict **`cars`** for the boats and
**`grass`** for the water, 8.8% of pixels change and **0 are fixed and 0 broken**. The
fit gives `ship` **w = 2.488**, the maximum the 0.40–2.50 search allows, and `cars`
still wins. ⚠️ **Same saturation as `mobile home`** (also 2.488, also 0.00 IoU), so the
`w` grid is binding on at least two classes and a wider grid is an untested, cheap
follow-up. ⭐ It is also a failure neither lever can reach: the scale is already
maximal.

⭐ **A small change can cost a lot, and `buildings93` shows it.** 71.8 → 43.0 with only
**2.6%** of pixels changed — 1,647 broken. A class holding a few thousand pixels loses
most of them, its IoU collapses, and the tile mean over three or four classes falls
~29 points. **That is the leverage argument of §12 appearing at tile level**, and it is
a far more instructive failure panel than a fully-discarded tile, because the method is
visibly touching almost nothing.

⚠️ **The figure's tile selection needed two fixes, and both were real.** `--auto` first
chose single-class tiles, where per-tile mean IoU can only be 0 or 100 so the extremes
are ±100 **by construction** — four near-solid panels showing the arithmetic of a
degenerate metric. It then chose a fully-discarded tile as the loss and a 0.0 → 0.0
tile as the no-op. A tile now needs ≥3 ground-truth classes and ≥2% changed; the loss
excludes annihilated tiles; the no-op requires a baseline ≥40 IoU.

### ⛔ 22 tiles are discarded ENTIRELY, and that is a stated failure mode

`agricultural` 18 · `beach` 2 · `chaparral` 1 · `golfcourse` 1. Every pixel sent to
the sink, nothing mislabelled. ⭐ **A tile that falls to zero with nothing wrongly
labelled is a different failure from one that falls to zero wrongly labelled**, and
only the second is a segmentation error — but both are real costs. **1.0% of the
dataset annihilated is a limitation even where the average improves.**

⚠️ **Per-tile mean IoU is NOT the dataset mIoU**, and the two need not agree in sign:
per-tile averages over the classes in that tile, mIoU averages each class once over
every pixel. `field` gains **+8.14 as a class** while individual agricultural tiles
collapse to zero. ⛔ Quote this section to say WHERE the method helps and hurts, never
to argue whether it does — that rests on `eval.py` over every pixel.

⛔ **A verdict bug found writing this, and it is the fifth of its kind in the project.**
`tile_delta_report.py` first divided the COUNT of a category's losing tiles by the
count of all losing tiles, and reported *"agricultural at 3.4%, losses are spread"* —
for the category carrying ~20% of the loss MASS and 18 of the 22 destroyed tiles.
**Counting tiles treats a −60 and a −0.1 as the same event.** Tables right, prose
wrong, again.

---

## 11. Predictions, scored — four hold, three do not

| | prediction | measured | |
|---|---|---|---|
| **D1** | discard below OEM's 3.78%, bracket 0.5–4.0% | ⛔ **6.16%** | ⛔ **FAIL** |
| **D2** | ≥99% of sink pixels are the τ rule; argmax-loss exactly 0 | structural — no background prompt exists | ✅ *used as a config check, not a finding* |
| **D3** | `--objective real` ≡ `--objective all`, max \|Δτ\| = 0.000 | holds **by construction**: with `bg` outside the class list the `real` filter removes nothing | ✅ |
| **D4** | median fitted τ > best global **and** > LoveDA's 0.375 | 0.150 > 0.070 ✅ but 0.150 < 0.375 ⛔ | ⛔ **HALF-FAIL** |
| **D5** | per-class beats the best fitted global by ≥ +0.5 | **+2.33** against the oracle global, which bounds any fitted one | ✅ |
| **D6** | stratified rung-A spread under 3.0 | ⛔ **5.41** | ⛔ **FAIL — and the gate was mis-specified** |
| **D7** | a rare class among the top two contributors | `airplane`, **0.33%** of pixels, **38%** of lever 1's gain | ✅ |

### ⛔ D6 failed, and the rule was mine and bad

The same statistic on three partitions of the same data:

| partition | rung A spread |
|---|---|
| random folds | 3.48 |
| stratified (`tau_cv`) | ⛔ **5.41** |
| stratified (`argmax_reorder`) | ✅ **2.31** |

⛔ **The range of five numbers is too unstable to be a gate** — it would have passed or
failed depending on which script was read. That is a badly-posed criterion, like
UAVid's gauge-dependent U4, and it is recorded as one rather than quietly relaxed.

What D6 was *proxying for* — can this dataset resolve the effect — is answered by the
effect sizes: noise ÷ effect is **2.3x** here against OpenEarthMap's **25x**, with
5/5 folds positive on both levers and mean−2sd of **+1.11** and **+3.78**. ⚠️ That
reasoning is post hoc, and post hoc is what a pre-registration exists to constrain.
**Stated, not used to overturn the result.**

### ⛔ D4's reasoning held and its size did not

D4 argued thresholds should run high because discarding a wrong pixel is free here.
The median **is** above the global optimum (0.150 > 0.070) ✅ — but well **below**
LoveDA's 0.375 ⛔. The direction was right, the magnitude wrong.

---

## 12. The headline, and how to quote it

| | |
|---|---|
| baseline, our reproduction *(no published number exists)* | **37.27** on the held-out split, **37.89** on all 2100 |
| lever 1 — per-class τ | **+1.77** |
| ⭐ lever 2 — per-class scale | ⭐ **+5.38** |
| **final** | ⭐ **44.42, total +7.15** |
| 5-fold, the figure with an error bar | lever 1 **+2.37 ± 0.63**, lever 2 **+5.84 ± 1.03**, both 5/5 |

⛔ **Quote the 5-fold as the headline and the deployment as the verification** — two
protocols, and the deployment is a single draw from the learning curve's spread.
⛔ **Never quote a DLRSD mean without the per-class table.** Six classes under 2% of
the pixels own **35.3%** of the metric, a 5.5x leverage — `WEEK3 §9h` at its most
extreme in the project.
⚠️ **Do not compare 44.42 against Potsdam's 57.83** and conclude anything. mIoU is
comparable only within a dataset; DLRSD has 17 fine-grained classes against Potsdam's
6 broad ones, and 37.89 is a **high** score here (best published training-free: 26.31; best **trained**: 45.64).

---

## 13. ⛔⭐ THE VOCABULARY ARM — four of six predictions fail, and the failures are the result

`prereg/predict_dlrsd_vocabulary.md` (`122f6e7`), committed before the new vocabulary
was cached. **Three lines changed, arity unchanged, fourteen identical:**
`chaparral` → **`shrubs`** (regional biome term → common word) · `mobile home` →
**`trailer`** (compound → ordinary term) · `field` → **`crop field`** (ambiguous → one
disambiguating modifier).

### ✅ The control is EXACT, and it matters more than anything below

Each class is an independent forward pass with its own text prompt, so the fourteen
untouched channels must be bit-identical. **Over 40 tiles × 14 classes: 560 identical,
0 differing.** ⭐ **Every movement in an untouched class is therefore pure ARGMAX
COMPETITION, not a side effect of re-running the model.**

| | published vocabulary | corrected | Δ |
|---|---|---|---|
| **mIoU** | 37.89 | **39.58** | **+1.69** |
| discard rate | 6.16% | 5.91% | −0.25 pp |
| ⛔ tiles discarded entirely | 22 | ⛔ **29** | **+7** |

### ⭐⭐ THE FINDING: `field` was eating its neighbours, and 75% of the fix lands on THEM

| class | Δ IoU | prompt |
|---|---|---|
| ⭐ **court** | ⭐ **+12.63** | **unchanged** |
| `crop field` | +7.80 | changed |
| ⭐ **bare soil** | **+5.89** | **unchanged** |
| ⭐ **grass** | **+5.40** | **unchanged** |
| `shrubs` | +5.19 | changed |
| ⛔ **trees** | ⛔ **−9.30** | **unchanged** |
| *(the other 11)* | *≤ 0.50 each* | |
| **sum** | **+28.60** → **+1.68 mIoU** | *(measured +1.69)* |

> ⭐⭐ **`crop field` gains 7.80 for itself and releases 23.92 to three classes whose
> prompt never changed — 75% of the benefit lands elsewhere.** `field` at 16.39%
> precision was winning baseball-diamond grass, bare soil and lawns that belonged to
> `court`, `bare soil` and `grass`. **The value of a prompt fix is in the argmax
> competition it stops, not in the class being renamed**, and this is the cleanest
> demonstration of that in the project because the control proves nothing else moved.

⛔ **And the `chaparral` fix is NET NEGATIVE.** `shrubs` grounds where `chaparral` did
not (0.00 → **5.19**) and then takes from `trees` (63.43 → **54.13**): **+5.19 against
−9.30 = −4.11**, or −0.24 mIoU. ⭐ **A prompt that grounds better is not the same as a
prompt that helps** — it has to win the *right* pixels, and `shrubs` wins vegetation
that was already correctly `trees`.

### Predictions, scored

| | prediction | measured | |
|---|---|---|---|
| **W1** | `shrubs` above 5.0 IoU | **5.19** | ✅ *barely* |
| **W2** | `trailer` above 5.0 IoU | ⛔ **0.04** | ⛔ **FAIL** |
| **W3** | `crop field` discard below 30.84% | 30.66% | ✅ *barely* |
| **W4** | baseline rises ≥ +2.0 | ⛔ **+1.69** | ⛔ **FAIL** |
| **W5** | discard below 3.78% | ⛔ **5.91%** | ⛔ **FAIL** |
| **W6** | untouched classes move < 0.5 | ⛔ **4 breach** | ⛔ **FAIL — badly posed** |

⛔ **W2 failed and it is the branch the pre-registration named as most informative.**
`trailer` reaches **0.04**. ⭐ **`mobile home` is a VISUAL confusion a rename cannot
reach, not a naming failure** — the units look like `buildings`. That is
`VOCABULARY_RESULTS`' Potsdam result replicating: renaming `tree`'s competitor left its
recall **38.63 → 38.63, unchanged to a hundredth**. ⭐ **So the vocabulary lever has a
bound, and this is the first dataset where both sides of it are visible at once**:
`chaparral` was a word problem, `mobile home` is not.

⛔ **W6 was MY error, not the run's.** The prereg gave *"the only coupling is the
argmax"* as the reason and then predicted the classes would not move — **the argmax
coupling is precisely the mechanism by which they would.** The prediction contradicted
its own stated justification, and its branch (*"stop, not a controlled comparison"*)
rested on a false premise. The 560/0 check settles it: the run is controlled, the
prediction was wrong. ⚠️ **Third badly-posed prediction in the project**, after UAVid's
gauge-dependent U4 and DLRSD's own D6.

### ⛔⭐ W5 IS THE IMPORTANT FAILURE: D1 STANDS AS STRUCTURAL

The discard rate had to fall **2.38 points** to vindicate the vocabulary. It fell
**0.25 — 11% of the gap** — and tiles discarded entirely got **worse, 22 → 29**.

> ⛔⭐ **The vocabulary is largely exonerated. D1's failure is structural**: a 0%
> catch-all genuinely does NOT produce the smallest residual, and `WEEK3 §7`'s
> surviving half does not survive extrapolation past its measured range. The remaining
> suspects are **17 competing classes** and **3.9x upsampling**, neither tested.

⭐ That is a firmer negative than D1 alone gave, because the cheapest and most
plausible alternative explanation has now been tested and rejected.

### ⛔ What is reported

**The method's DLRSD result stays 37.27 → 44.42 on the PUBLISHED vocabulary**, exactly
as the pre-registration requires. The **+1.69** is its own result. ⚠️ And unlike
UAVid's +3.53, it is **not** a clean win to fold in: it is +1.69 net of a **−4.11**
regression on `trees`/`shrubs`, it raises the annihilated-tile count from 22 to 29, and
two of its three prompt changes did not do what they were chosen to do.

### ✅✅ VERIFIED END-TO-END on the corrected vocabulary — and W7 settles

`reorder_deploy.py` on the v2 cache (399 calibration tiles, same seed and stems), then
three `eval.py` passes on the 1701 held-out tiles. ⭐ **Same held-out set as §1**:
`sea` scores **43.98** and `ship` **15.00** at rung A in both runs, and neither prompt
changed.

| rung | predicted | **`eval.py`** | Δ |
|---|---|---|---|
| A — published τ | 39.05 | **38.98** | 0.07 |
| B — per-class τ | 40.55 | **40.49** | 0.06 |
| ⭐ **C — + scale** | 46.10 | ⭐ **46.12** | ⭐ **0.02** |

**Sixth verified chain in the project.** `aAcc` 60.19 → 65.29 (+5.10), mPrecision
**+10.58** and mRecall **+1.66** — both rise again, and lever 1 trades (recall −4.13)
while lever 2 pays it back (+5.79), exactly as on the published vocabulary and on UAVid.

⭐⭐ **W7 — the lever gains are UNCHANGED by the vocabulary, now on pipeline numbers:**

| vocabulary | A | B | C | **total** |
|---|---|---|---|---|
| published | 37.27 | 39.04 | 44.42 | **+7.15** |
| corrected | 38.98 | 40.49 | 46.12 | **+7.14** |

The vocabulary is worth **+1.71** before the levers and **+1.70** after them. ⭐ **They
add exactly.** On UAVid one word removed **64%** of lever 2 and both routes converged
(63.55 vs 63.62); on DLRSD they are **independent**. ⚠️ Scored against its own bar W7
passes by 0.01 — "total falls below +7.15" — which is noise; **the finding is the
additivity, not the pass**. ⛔ **"The vocabulary and the levers substitute" does not
generalise**, the second substitution reading in the project to fail on a new dataset
after `SUBSTITUTION_RESULTS` refuted "the levers substitute".

⭐ **The class-level picture is subtler than the total, and worth one paragraph.**
Rung C, corrected minus published:

| class | published C | corrected C | Δ |
|---|---|---|---|
| ⭐ `field` → `crop field` | 22.05 | **38.89** | **+16.84** |
| ⭐ `chaparral` → `shrubs` | 0.00 | **13.35** | **+13.35** |
| `sea` | 49.16 | 54.84 | +5.68 |
| `grass` | 52.23 | 55.50 | +3.27 |
| `tanks` | 35.46 | 38.37 | +2.91 |
| ⛔ `court` | 56.43 | 55.85 | **−0.58** |
| ⛔ `water` | 48.07 | 45.59 | −2.48 |
| ⛔ **`trees`** | 67.06 | 54.09 | ⛔ **−12.97** |

- ⭐ **`court` substitutes; `crop field` does not.** At the baseline the word released
  **+12.63** to `court` — and after the levers that gain is **gone (−0.58)**, because lever
  2 had already rescued `court` on the published vocabulary. `crop field` itself gains
  **+16.84 that no lever reached**. So the words and the levers overlap on *some* classes
  and not others, and the totals cancel into exact additivity.
- ⭐ **The levers repair the `shrubs`/`trees` trade to break-even.** At the baseline
  `shrubs` cost `trees` **−4.11 net**; after both levers `trees`+`shrubs` is **67.44**
  against **67.06** — the fit re-divides the vegetation between the two prompts.
- ⛔ **`water` loses under the corrected vocabulary even after the levers** (−2.48), on top
  of losing −2.73 to the levers already. It is the dataset's most consistently harmed class.
- ⛔ **`trailer` stays at 0.29** at every rung and vocabulary. Three interventions, no
  effect: a visual confusion.

### How to quote DLRSD now

| | |
|---|---|
| reproduction, published vocabulary | **37.27** (held-out) |
| **method** on the published vocabulary *(the pre-registered result)* | ⭐ **+7.15 → 44.42** |
| vocabulary arm, its own result | **+1.71** at baseline, **+1.70** after the method |
| best verified configuration | ⭐ **46.12** |

⛔ The method's contribution is **+7.15 on either vocabulary** — the vocabulary does not
inflate or deflate it here, which is the cleanest statement of that separation in the
project. ⚠️ And 46.12 is **training-free**; OVRSISBench's best **training-free** DLRSD
number is 26.31 and its best **trained** one is **45.64** (Pi-Seg, ViT-L, trained on
OVRSIS95K). Different backbone, resolution and unstated taxonomy — **not a controlled
comparison, and most of the margin over CLIP is SAM 3.**

