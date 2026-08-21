# Problem Statement Analysis — Unsupervised Semantic Segmentation of Remote Sensing Images

*Analysis date: August 2026 · last updated 18 Aug 2026 (baseline reproduced; ConInfer added)*

---

## 0. The most important thing to know first

**Your second PDF is not the SAM paper.** The file named `sam_research papaer (1).pdf` is:

> **SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images**
> Li, Zhang, Wang, Deng, Wang, Meng, Cao — Xi'an Jiaotong University
> arXiv:2512.08730v2 (v2 dated 22 Apr 2026) · [code](https://github.com/earth-insights/SegEarth-OV-3) · 205 stars

This matters enormously, because **that paper already does a large part of what your problem statement proposes**, by the same group that wrote your first PDF (SegEarth-OV, CVPR 2025). You have accidentally been handed your own most direct competitor.

> ### ⚠️ Second competitor, found later: ConInfer (arXiv:2603.29271, ~Mar 2026)
>
> **ConInfer: Context-Aware Inference for Training-Free Open-Vocabulary Remote Sensing
> Segmentation** (Chen et al., Yunnan Normal University) claims to be the *first work to
> explicitly incorporate contextual information at inference* for OVRSS. It fits a GMM over
> DINOv3 patch features and fuses the resulting posterior with a VLM prior via a joint
> KL-divergence objective, reporting **+2.80 mIoU** over SegEarth-OV on the same eight
> benchmarks you plan to use.
>
> This overlaps your Step 4. **Any reviewer in this area will know it.** But it does not
> close your gap:
>
> | | ConInfer | Yours |
> |---|---|---|
> | Backbone | CLIP / MaskCLIP | **SAM 3** (presence + semantic + instance heads) |
> | Context type | **Visual only** (GMM over DINOv3 features) | **Semantic** — class-pair adjacency prior |
> | Granularity | **Patch** level | **Region** level (SAM 3 masks) |
> | Prior source | Per-scene, fitted at inference | **Corpus-level**, mined across the dataset |
> | Exclusion evidence | Not representable | **Signed PMI** — negative evidence suppresses classes |
>
> ConInfer's own conclusion states it operates at patch level, which limits fine-grained
> boundary delineation, and names pixel-level contextual modelling as future work.
> **That is precisely your opening.** Cite it as the nearest neighbour, position against it
> explicitly, and — if at all possible — run it as a baseline row.
>
> **Revised one-sentence contribution:** *a corpus-level semantic co-occurrence prior over
> SAM 3 region proposals, where ConInfer uses a per-scene visual prior over CLIP patches.*

This is good news, not bad. It means:

- The baseline is published, open-source, and reproducible — you don't have to build from zero.
- The "does SAM 3 work on remote sensing?" question is answered, so you don't burn a month on it.
- You now know exactly what your contribution has to be to not be a duplicate.

But you must **reposition your framing**, or a reviewer/examiner will open arXiv on day one and ask why your project is different. Section 2 explains how.

---

## 1. What each paper actually does

### 1.1 SegEarth-OV (CVPR 2025) — your first PDF

**Problem:** CLIP-based training-free OVSS produces distorted shapes and ill-fitting boundaries on remote sensing imagery, because ViT-B/16 features are downsampled to 1/16 of input resolution. On aerial images full of small objects, that is fatal.

**Two contributions:**

1. **SimFeatUp** — a universal, lightweight feature upsampler trained on a *few unlabeled images* that restores spatial detail in arbitrary RS features. (This is why they footnote that "training-free" really means "annotation-free" — SimFeatUp *is* trained, just without labels.)
2. **Global bias subtraction** — they observed CLIP patch tokens respond abnormally to the `[CLS]` token, and simply subtract it out.

**Results:** 17 RS datasets, 4 tasks (semantic seg, building extraction, road extraction, flood detection). +5.8 / +8.2 / +4.0 / +15.3 % over SOTA.

**Why you read it:** it defines the problem space, the evaluation protocol, and the dataset suite you will inherit. Its `dataset_prepare.md` is the practical dataset guide for your project.

### 1.2 SegEarth-OV3 (arXiv 2512.08730) — your second PDF

**Problem:** CLIP-based pipelines are complex multi-model ensembles with poor localization. SAM 3 unifies segmentation + recognition, so can it just replace the whole stack?

**SAM 3 architecture (memorise this — it is the spine of your project):**

```
Image I ──► Vision Encoder ──┐
                             ├──► Fusion Encoder ──► F_cond ──┬──► Presence Head      → S_pres ∈ [0,1]  (scalar: does concept t exist?)
Text t ───► Text Encoder ────┘                                ├──► Semantic Seg Head  → P_sem ∈ [0,1]^(H×W)  (FCN-style, dense)
                                                              └──► Transformer Decoder→ {(P_inst^k, s_conf^k)}_{k=1..N}  (DETR/MaskFormer, instance queries)
```

**Their two strategies:**

1. **Dual-Head Mask Fusion.** The instance head is sharp on countable "things" (cars, buildings) but fragments amorphous "stuff" (roads, bareland). The semantic head is continuous on "stuff" but blurs small "things." So:

   ```
   P_inst_agg(h,w) = max_k [ P_inst^k(h,w) · s_conf^k ]        (instance aggregation)
   P_fused(h,w)    = max( P_sem(h,w), P_inst_agg(h,w) )        (dual-head fusion)
   ```

2. **Presence-Guided Filtering.** A global land-cover vocabulary queried against a 200-metre image patch means most classes are physically absent → hallucinations ("low vegetation" vs "sports field"). So gate by the presence score:

   ```
   P_final^(c) = P_fused^(c) · S_pres^(c)
   M(h,w)      = argmax_c P_final^(c)          ... below threshold τ → "background"
   ```

Plus an extension to open-vocabulary **change detection** via joint instance- and pixel-level verification.

**Results:** 20 segmentation datasets, 3 change-detection datasets, 1 3D point-cloud dataset.

---

## 2. Your pipeline vs. SegEarth-OV3 — the honest diff

| Your step | What SegEarth-OV3 does | Verdict |
|---|---|---|
| **1.** SAM 3 → identified vs. unidentified via threshold τ_high | `argmax` over `P_final`, below τ → background | **Already done.** Theirs is better — you propose a raw-confidence threshold, they gate by presence score first |
| **2.** Group identified patches by predicted class | Instance aggregation (max over queries × conf) | **Already done**, and more principled |
| **3.** Co-occurrence Matrix **M** + **INFO** | *Nothing comparable* | **This is yours.** No co-occurrence modelling anywhere in the paper |
| **3b.** Preprocessing module (SLIC / Felzenszwalb) on unidentified regions | Handled implicitly by dual-head fusion | **Different approach**, arguably yours is more explicit |
| **4.** Contextual clustering: label unidentified patches using M + INFO | *Nothing comparable* — they simply drop low-confidence pixels to "background" | **This is yours, and it's the real contribution** |
| **5.** Fuse identified + newly-labelled → final mask | Dual-head max-fusion | **Overlaps**, but yours fuses different inputs |

### The one-sentence version of your contribution

> SegEarth-OV3 **throws away** low-confidence pixels by dumping them into "background." Your method **recovers** them, by exploiting the fact that land cover has extremely strong spatial co-occurrence structure — roads touch buildings, bareland touches agriculture, water touches vegetation — and using that statistical prior to resolve what SAM 3 alone could not.

That is a legitimate, defensible, publishable framing. Remote sensing is arguably the *best* domain for this argument, because aerial scenes have far more rigid spatial-semantic grammar than natural images. A cat can be anywhere in a photo; a parking lot is essentially always adjacent to a building or a road.

### How to reposition the project

Stop describing it as "unsupervised segmentation with SAM 3." Describe it as:

> **Context-aware label propagation for training-free open-vocabulary remote sensing segmentation** — recovering the ambiguous-region residual that presence-gated SAM 3 pipelines discard.

And make SegEarth-OV3 your **explicit baseline**. Your headline number becomes "+X% mIoU over SegEarth-OV3, particularly on the pixels it assigns to background." That is a much stronger story than an unanchored number.

---

## 2.5 Baseline reproduced — 18 Aug 2026

`python eval.py ./configs/cfg_loveda.py` → **mIoU 47.38** against the paper's **47.4**
(Δ 0.02). 1669 LoveDA val images, 0.85 s/image, 6115 MB peak on a 16 GB RTX 2000 Ada.
Environment and full per-class table in `WEEK1_RESULTS.md`.

**The per-class breakdown is evidence for the premise, not just a sanity check:**

| Class | IoU | Precision | Recall | P−R gap |
|---|---|---|---|---|
| building | 63.81 | 77.2 | 78.6 | −1.4 |
| road | 53.89 | 69.6 | 70.5 | −1.0 |
| **water** | 51.44 | **89.5** | **54.7** | **+34.8** |
| agricultural | 47.47 | 66.9 | 62.0 | +4.9 |
| **background** | 45.50 | **56.9** | **69.4** | **−12.5** |
| barren | 35.73 | 51.5 | 53.9 | −2.4 |
| forest | 33.78 | 57.9 | 44.8 | +13.1 |

Three readings:

1. **Water: 89.5 precision / 54.7 recall.** SAM 3 is right nine times in ten when it fires,
   but finds half the water present. Those pixels are not *misclassified* — they fall below
   τ = 0.5 and are discarded to background.
2. **Background: 56.9 precision / 69.4 recall — the inverse.** Background is over-predicted
   and impure: it is absorbing pixels that belong to real classes. This is the discard
   bucket, directly visible in the metrics.
3. **Building (−1.4) and road (−1.0) are balanced.** The weakness is confined to amorphous
   "stuff", matching SegEarth-OV3's own things/stuff duality. Your method should be evaluated
   *per class*, and the gain should concentrate in water, forest, and agricultural.

This is indirect evidence — §6's discard-rate instrumentation converts it into a direct
number for the paper's introduction.

---

## 3. Technical critique of your approach

These are the issues that will bite you during implementation. Address them in your design *now*, not in month three.

### 3.1 A per-image co-occurrence matrix is statistically far too sparse

Your Step 3 builds M from the identified patches **of a single image**. A typical aerial tile yields maybe 10–40 confident patches spanning 3–6 classes. An n×n matrix estimated from ~30 samples is noise, not statistics. You will be conditioning on garbage.

**Fix — build M hierarchically:**

```
M_global   ← accumulated over the entire unlabeled corpus (thousands of tiles)   [strong, stable prior]
M_image    ← this image's identified patches only                                [weak, scene-specific]
M_eff      = λ · M_global + (1 − λ) · M_image
```

This is a Bayesian shrinkage / hierarchical smoothing argument, and it is also *the thing that makes your method genuinely "unsupervised"* — you mine statistics from the unlabeled dataset itself, which is exactly the setting your problem statement describes. Frame it that way and it becomes a feature, not a patch.

A Dirichlet prior with pseudo-counts (add-α smoothing) on M is the clean formulation. λ can even be adaptive: trust `M_image` more when the image has many confident patches.

### 3.2 Circularity: M inherits SAM 3's biases

M is estimated from SAM 3's own confident predictions. If SAM 3 is systematically confident about "building" and systematically unsure about "bareland," then M under-represents every bareland relationship — and you then use M to label the bareland regions SAM 3 missed. The prior actively works against the classes you most need help with.

**Mitigations:**

- Compute a *reference* M from ground-truth annotations on a held-out set and report the divergence (KL or Frobenius) against your estimated M. This is a diagnostic experiment, not a supervision leak, and it makes a great figure.
- Weight contributions to M by patch confidence, not uniformly.
- Consider a class-frequency correction so rare classes aren't crushed.

### 3.3 Your visual-similarity term is weakest exactly where you need it

Step 4 scores unidentified patches by comparing their embedding against known class prototypes. But a region is unidentified *because* SAM 3 found it visually ambiguous. The embedding is, almost by construction, uninformative there.

Expect the co-occurrence and neighbour terms to carry nearly all the signal. **Run this ablation early** (week 8, not week 11) — if the embedding term contributes ~0, cut it and simplify the method. Knowing this early saves weeks.

### 3.4 The preprocessing step contains a conceptual muddle

Your statement says the preprocessing module "solves the over-segmentation problem" and then proposes SLIC / Felzenszwalb. But **SLIC deliberately creates over-segmentation** — that is its entire purpose. You cannot use an over-segmentation algorithm to fix over-segmentation.

What you actually want is a two-stage design, and you should say so explicitly:

1. **Atomise** — SLIC/Felzenszwalb break noisy unidentified pixels into small, colour/texture-homogeneous superpixels that respect real edges.
2. **Agglomerate** — merge adjacent superpixels using a Region Adjacency Graph, with a merge criterion combining feature similarity *and* co-occurrence compatibility.

Stage 2 is what fixes over-segmentation. Stage 1 just gives you clean atoms to work with.

**Also consider skipping SLIC entirely.** SAM 3's instance head already gives you class-agnostic mask proposals with far better boundary quality than any classical superpixel algorithm. Using SAM 3's own low-confidence masks as your patch proposals is likely both simpler and stronger. Test both — it's a cheap experiment and a good ablation row.

### 3.5 Use the presence head in Step 1 — but do not inherit it uncritically

Your Step 1 thresholds raw prediction confidence. SegEarth-OV3's Figure 3 demonstrates that with
a large vocabulary this produces severe noise. Gate by `S_pres` *before* thresholding, exactly as
they do — that part stands.

**But the earlier claim in this section, that inheriting their fix "costs you nothing", was
wrong, and measurement has now disproved it.**

Because `P_final = P_fused · S_pres`, the presence score is a hard ceiling on every pixel in the
tile for that class. When the presence head is miscalibrated low, it destroys dense evidence
that is otherwise confident. Probed on LoveDA val tile `3487` (`WEEK1_RESULTS.md` §9.2):

| Class | S_pres | `semantic_seg` max logit | → sigmoid | Ceiling on P_final |
|---|---|---|---|---|
| road | **0.0757** | **+10.13** | 1.000 | **0.076** |
| building | 0.1309 | +6.44 | 0.998 | 0.131 |
| forest | 0.0094 | +5.03 | 0.993 | 0.009 |
| agricultural | 0.0200 | +5.44 | 0.996 | 0.020 |
| barren | 0.0481 | +2.77 | 0.941 | 0.045 |
| water | 0.0298 | +2.28 | 0.907 | 0.027 |

The semantic head is effectively certain road is present (+10.13). The presence gate reduces the
tile's best achievable score to 0.076 — below τ=0.1, let alone τ=0.5. Zero instance masks are
returned for all six classes. The tile reports **100% discard**, and 54 other tiles behave the
same way.

**This is a second, distinct failure mode.** The low-confidence residual (§2, §4) is one problem;
presence-head collapse is another, and it accounts specifically for the catastrophic tail
identified in `WEEK1_RESULTS.md` §7.4 — the 55 tiles at 100% discard against 958 tiles below 1%.

Three implications for the method:

1. **It is favourable, not fatal.** Had these tiles failed because `P_sem` was genuinely
   uncertain, there would be no signal left to recover. Instead the dense evidence is intact and
   being suppressed by a single global scalar. Recoverable information is the precondition for
   this entire project.

2. **A local prior is the natural correction for a bad global scalar.** `S_pres` is one number
   per class per image. The co-occurrence prior aggregates evidence from *neighbouring regions*.
   These are different information sources, so the prior can in principle override a presence
   veto that the dense heads already contradict. That is a stronger argument for the method than
   the residual-recovery framing alone, and it is worth stating explicitly in the paper.

3. **But the seeding problem is real, and must be resolved before Week 8.** The method labels
   unidentified regions by conditioning on the labels of *identified* neighbours. On a
   100%-discard tile there are no identified patches at all — no seeds, an empty `M_image`, no
   neighbour labels to condition on. `M_global` alone cannot place a label without an anchor.
   Options, to be decided by measurement:
   - Bootstrap from `P_fused` *before* presence gating on tiles where all classes are suppressed
     (a presence-bypass fallback, triggered by a detectable condition rather than a hand-tuned
     rule).
   - Use per-class relative ordering of `S_pres` rather than its absolute value — on 3487 the
     ranking (road < forest < agricultural < water < barren < building) may still be informative
     even where the magnitudes are useless.
   - Accept the 55 tiles as out of scope and report them as an inherited baseline limitation.
     Honest, but it forfeits the most dramatic gains available.

**Caveat: n = 1 tile.** This is currently a single well-characterised anecdote. Before it appears
in the paper, dump per-class `S_pres` across all 1669 val tiles and compare the distribution on
the catastrophic set against the healthy set. If catastrophic tiles are systematically
presence-suppressed, this is a figure and a finding that SegEarth-OV3's paper does not report.
If not, it is a curiosity about one tile and must be dropped. See `INSTRUMENTATION_PATCH.md`.

### 3.6 Terminology: "unsupervised" is the wrong word

Your title says *Unsupervised*, but your input includes "a list of all the class names that are possible." That is not unsupervised — it is **zero-shot / open-vocabulary / training-free**. SegEarth-OV explicitly footnotes that even "training-free" is a stretch for them (SimFeatUp is trained on unlabeled data), and settles on **annotation-free**.

Get this right in your writeup. Examiners notice. Recommended phrasing: *"training-free, annotation-free open-vocabulary semantic segmentation."*

### 3.7 The original repo was built on SAM 1 + CLIP, not SAM 3 — ✅ removed 21 Aug

*(Retained for the record; the code it describes no longer exists in the working tree.)*

The original scaffold (`src/`, `pipeline.py`, `configs/config.yaml`, `tests/`) implemented the
pipeline against SAM 1's `SamAutomaticMaskGenerator` plus CLIP crop-and-classify. That
architecture has no presence head, no semantic head, and no text conditioning inside the
segmenter — the three things your method depends on. It also carried a live bug directly relevant
to Step 3:

> `Preprocessor._greedy_merge` merged patches by mask **IoU ≥ 0.25**, but over-segmented fragments
> are *disjoint*, so IoU ≈ 0 and nothing ever merged. Adjacency-based merging (dilate one mask,
> test boundary overlap) is what you need — and it is exactly the Region Adjacency Graph from §3.4.

Two further defects worth remembering, because the *concepts* migrate into Phase 3 even though
the code does not:

- Morphological opening could erode a small patch to **zero area** without dropping it; the
  centroid helper then returned `(0,0)`, silently anchoring the patch at the image origin and
  poisoning every nearest-neighbour lookup downstream. Silent corruption, not a crash.
- `_find_k_nearest` recomputed centroids with `np.where` over the full mask for every identified
  patch, for every unidentified patch — O(U × I × H × W). Unusable at 1024².

Every design decision the scaffold encoded (centroid-distance adjacency, raw symmetric counts,
additive-positive scoring, per-image M) has since been overturned by §4. It was **removed on
21 Aug** — it is in git history, and its `requirements.txt` was an active hazard to the pinned
environment. Phase 3 builds by forking `segearthov3_segmentor.py` instead.

---

## 4. Empirical findings — measured, not assumed

*Run: 12 Aug 2026 · LoveDA Val (677 urban + 992 rural masks, 1024×1024 @ 0.3 m) · `scripts/cooccurrence_gt.py`*

Sections 1–3 argued from intuition. This section replaces three of those arguments with measurements. **Everything below is reproducible from this repo** — that matters, because these numbers are the empirical foundation of your method section.

**Method.** Adjacency is measured as *shared boundary length*: every horizontally or vertically neighbouring pixel pair carrying two different class labels counts once. Observed adjacency is then compared against an independence baseline via pointwise mutual information:

```
PMI(i,j) = log2( P_observed(i,j) / (P(i) · P(j)) )
```

PMI > 0 means the two classes border each other more than chance; < 0 means they avoid each other; ≈ 0 means adjacency is indistinguishable from random. Mean |PMI| over off-diagonal pairs is the headline statistic.

**Control.** On synthetic masks with classes scattered uniformly at random, the pipeline reports **0.004 bits**. That is the noise floor, and it is what "no signal" looks like.

> ⚠️ **Caveat on the null model — added 21 Aug, affects per-pair values only.**
> `P_obs` is a **boundary**-frequency distribution, but `P_exp = outer(p, p)` is built from
> **area** marginals (`scripts/cooccurrence_gt.py:118-131`). These are different measures. A
> class fragmented into thin ribbons has far more boundary per unit area than one in large
> contiguous blocks, so **PMI is systematically inflated for high-perimeter classes and deflated
> for compact ones, independent of any semantic affinity.**
>
> The control above does *not* catch this: scattering classes uniformly destroys blob geometry
> altogether, so it tests "is there any signal at all", not "is this pair's signal geometric or
> semantic".
>
> **What is and is not at risk.** The headline premise is not — 1.3–1.7 bits against a 0.004
> floor is far too large a margin to be a geometric artefact, and §4.1's conclusion stands. What
> *is* at risk is the **per-pair** structure, and specifically §4.3: `road` is the thinnest,
> highest-perimeter class in LoveDA, its entire PMI row is uniformly elevated, and that row is
> exactly what the "roads are hubs" finding and the discriminability weighting are derived from.
>
> **Fix, before §4.3 is relied on:** a structure-preserving permutation null — keep each image's
> mask geometry, permute the class labels across regions, ~100 draws. That yields per-pair
> z-scores with confidence intervals instead of raw bits, which is strictly better for the paper.
> CPU-only and cheap. Tracked in `WEEK1_RESULTS.md` §10.

### 4.1 The co-occurrence prior carries strong signal — and it is not an artefact

| Domain | Mean \|PMI\| with `background` | Without `background` | Change |
|---|---|---|---|
| Urban | 1.658 bits | 1.326 bits | −20% |
| Rural | 1.538 bits | **1.662 bits** | **+8%** |
| Random control | — | 0.004 bits | — |

LoveDA's `background` is a catch-all (26% of urban area, 43% of rural) that borders nearly everything — building→background adjacency is 0.933. The obvious worry was that the whole signal was just *"things touch the leftover class"*, which cannot disambiguate anything.

It isn't. Removing background **increased** the rural signal, meaning background was diluting genuine structure rather than creating it. Urban lost a fifth — consistent with urban background being pavement acting as connective tissue — but 1.326 bits against a 0.004 noise floor is decisive.

**Conclusion: the central premise of this project is sound.** Land cover has strong adjacency structure among real semantic classes.

Once background is excluded, the attractions become semantically interpretable rather than bookkeeping:

| Pair | Urban | Rural | Reading |
|---|---|---|---|
| road – barren | +1.37 (2.6×) | **+3.40 (10.5×)** | unpaved shoulders and margins |
| road – forest | +1.44 (2.7×) | +2.44 (5.4×) | roads cut through woodland |
| water – barren | −0.22 | +1.70 (3.2×) | exposed banks, seasonal margins |
| forest – agriculture | +1.17 (2.3×) | −1.62 | field margins vs. cleared blocks |
| **building – water** | **−4.45 (0.05×)** | **−5.38 (0.02×)** | near-hard constraint |

### 4.2 The signal is exclusion-dominated → use signed PMI, not raw counts

The largest magnitudes are negative. Building–water sits at −5.38 (0.02× chance) in rural, against a best attraction of +3.40. Buildings essentially never border water in either domain.

This **invalidates the additive-positive-evidence formulation** in the original problem statement:

```python
scores += w_coc * M[neighbour_class]      # attraction only - discards the strongest signal
```

Raw row-normalised counts can only ever *add* evidence. The most reliable information available — "this patch borders a building, therefore it is almost certainly not water" — is unrepresentable in that form.

**Use signed PMI as the co-occurrence term** so negative evidence actively suppresses candidate classes. This is a stronger method than originally proposed, and now has a measured justification rather than an intuition behind it.

### 4.3 Neighbours are not equally informative — weight by discriminative power

*(New finding — not anticipated in §3.)*

Rural `road` attracts almost everything: barren +3.40, forest +2.44, building +1.34, agriculture +1.13. Roads are **connectors**, so "this patch borders a road" barely narrows the vocabulary. `building`, by contrast, is strongly negative against water, agriculture and forest — "borders a building" eliminates half the candidates.

So a uniform sum over neighbours dilutes strong evidence with weak evidence. Weight each neighbour's contribution by the spread of its PMI row:

```
w(n) ∝ Var_c[ PMI(label(n), c) ]        # or negative entropy of the row
```

Hub classes contribute little, exclusive classes contribute a lot. Small change, clear justification, and a natural ablation row.

### 4.4 The prior is domain-specific → the hierarchical M in §3.1 is required

| | With background | Without |
|---|---|---|
| Mean \|PMI difference\|, urban vs rural | 1.137 bits | 1.310 bits |
| Max \|PMI difference\| | 3.064 bits | 2.791 bits |

**Six of fifteen class pairs flip sign** between domains: building–road, building–barren, water–barren, barren–agriculture, forest–agriculture, water–forest.

The clearest case is building–road: **−1.14 in urban** (they avoid — separated by pavement that LoveDA labels background) but **+1.34 in rural** (buildings sit directly on roads). A single global M averages these into mush and would actively mislead on both domains.

Note that removing background made the domains look *more* different, not less — the catch-all class was a shared denominator masking domain-specific structure.

**This settles §3.1.** `M_eff = λ·M_global + (1−λ)·M_image` is not a refinement, it is a requirement. λ becomes a first-class hyperparameter and the λ-sweep is a mandatory ablation.

### 4.5 SAM 3's fragmentation asymmetry, confirmed on LoveDA

Single 1024×1024 LoveDA tile, via `scripts/sam3_smoke_test.py --raw`:

| Prompt | Instance masks returned | Score range | Presence score |
|---|---|---|---|
| `building` | **14** | 0.51 – 0.77 | high |
| `road` | **2** | 0.81 – 0.85 | **0.957** |

The presence head is confident roads exist (0.957), yet the Transformer decoder emits only two instances. It resolves countable "things" and collapses on amorphous "stuff" — exactly SegEarth-OV3 Figure 1, reproduced independently on our own data. Road coverage has to come from `semantic_seg`, which is what makes dual-head fusion necessary rather than optional.

**All required tensors are reachable** via `model.forward_grounding()` (the public `Sam3Processor` API discards most of them):

| Needed for | Tensor | Shape |
|---|---|---|
| `S_pres` — Step 1 gating | `presence_logit_dec.sigmoid()` | `(1,1)` |
| `P_sem` — "stuff" coverage | `semantic_seg` | `(1,1,288,288)` |
| `P_inst`, `s_conf` — "things" | `pred_masks`, `pred_logits` | `(1,200,288,288)`, `(1,200,1)` |
| `INFO` — patch embeddings | `encoder_hidden_states` | `(5184,1,256)` = 72×72×256 |

Two implementation notes: SAM 3 resizes all input to 1008×1008, so **tiles must be square** or aspect ratio is distorted; and inference **must** run inside `torch.autocast("cuda", dtype=torch.bfloat16)` — see `SETUP_SAM3.md`.

### 4.6 What this changes

| Original design | Revised, and why |
|---|---|
| Per-image M | **Hierarchical** `λ·M_global + (1−λ)·M_image` — §4.4, six sign flips |
| Raw row-normalised counts | **Signed PMI** — §4.2, exclusion carries the largest magnitudes |
| Uniform sum over neighbours | **Discriminability-weighted** — §4.3, hub vs exclusive classes |
| Centroid-distance adjacency | **Shared boundary length** — already used throughout this analysis |
| "Does the premise hold?" — open | **Settled.** 1.3–1.7 bits vs a 0.004 noise floor |

Still open, and the next things to measure:

1. **Does the embedding-similarity term earn its place?** (§3.3). Run that ablation in week 8.
2. **Is §4.3's "road is a hub" semantic or geometric?** Run the permutation null in the §4
   caveat box. It is cheap, and it is the only measured finding here whose null model has not
   itself been checked.
3. **How much of the 29.68% residual is region-shaped rather than boundary-shaped?**
   `WEEK1_RESULTS.md` §9.1a. A region-level prior can only address the former, so this converts
   the headline into an *addressable* number.

---

## 5. What is genuinely novel, ranked

*Recalibrated after finding ConInfer (arXiv:2603.29271), which published context-at-inference
for OVRSS in March 2026.*

| Idea | Novelty | Risk | Worth doing? |
|---|---|---|---|
| Corpus-level **semantic** co-occurrence prior (signed PMI over class pairs) | **High** — ConInfer's context is purely visual; no published work uses a class-adjacency prior here | **Low** (§4.1 measured the premise) | **Yes — this is the thesis** |
| Context at **region** level via SAM 3 masks | **Medium-High** — ConInfer is patch-level and names this as future work | Low | **Yes — this is the differentiator** |
| Recovering background-assigned pixels as the headline evaluation | **Medium** — the framing is yours; §2.5 measured the opportunity | Medium | **Yes — this is the experiment** |
| Context modelling at inference *in general* | ~~High~~ → **None** (ConInfer, Mar 2026) | — | Cite, position against, don't claim |
| RAG-based agglomeration of SAM 3 low-confidence masks | Medium | Low | Yes — solid supporting contribution |
| Dual-head fusion | None (published) | — | Reuse, cite, don't claim |
| Presence-guided filtering | None (published) | — | Reuse, cite, don't claim |
| Applying SAM 3 to remote sensing at all | None (published) | — | Reuse, cite, don't claim |

**Stretch goal if things go well:** co-occurrence is a *directed, asymmetric, scale-dependent* relation in remote sensing that nobody has modelled properly. "Road adjacent to building" and "building adjacent to road" carry different conditional probabilities, and both change with ground sampling distance. A GSD-conditioned co-occurrence prior would be a genuinely original angle.

---

## 6. Evaluation plan — decide this now, not later

Your project lives or dies on whether you can show a number beating SegEarth-OV3.

**Metrics:** mIoU (primary), per-class IoU, pixel accuracy. Report mIoU **restricted to pixels SegEarth-OV3 labels as background** — that isolates your contribution and will be your strongest table.

**Datasets** (all reachable via SegEarth-OV's `dataset_prepare.md`):

- **Start:** OpenEarthMap (8 classes + background, 0.25–0.5 m GSD, global coverage) — the cleanest fit for co-occurrence modelling.
- **Then:** LoveDA (7 classes, urban/rural split — lets you test whether M transfers across domains, a *great* experiment).
- **Then:** Potsdam / Vaihingen (6 classes, ultra-high 5 cm / 9 cm resolution).
- **Optional:** iSAID (15 classes, instance-heavy — stress-tests the "things" side).

**Baselines to run, in order:**

1. SAM 3 instance-head only
2. **SegEarth-OV3** (dual-head + presence) — ✅ reproduced, LoveDA **47.38**
3. SegEarth-OV3 with τ lowered — *the trivial alternative a reviewer will raise. If simply
   dropping τ recovers the same pixels, your method has no reason to exist. Run this early.*
4. DenseCRF over SAM 3 logits — the classical "use spatial context" answer
5. **ConInfer** ([code](https://github.com/Dog-Yang/ConInfer)) — the nearest published competitor
6. SegEarth-OV3 + your co-occurrence module

Rows 3 and 5 are not optional. Row 3 is the cheapest possible refutation of your method; row 5
is the reviewer's "how is this different from ConInfer?" answered with a number.

**Ablations:** M_global only / M_image only / hierarchical · embedding term on-off · neighbour term on-off · SLIC atoms vs. SAM 3 mask atoms · λ sweep.

Also look at **OVRSISBenchV1** ([arXiv:2604.15652](https://arxiv.org/html/2604.15652)), a 2026 benchmark that reformulates DLRSD, iSAID, Potsdam, Vaihingen, UAVid, LoveDA, VDD and UDD5 under one unified open-vocabulary protocol. Evaluating on a standardised protocol is a credibility multiplier.

---

## 7. Bottom line

Your instinct is good and the co-occurrence idea is real. Three things have to change:

1. **Reframe** around recovering the ambiguous residual, with SegEarth-OV3 as your named
   baseline — not as a from-scratch SAM 3 pipeline. ✅ Baseline reproduced (47.38 vs 47.4),
   and §2.5 shows the residual is measurable in the per-class metrics.
   **Position explicitly against ConInfer**: semantic vs. visual context, region vs. patch
   granularity, corpus-level vs. per-scene prior.
2. **Make M corpus-level and hierarchically smoothed**, which fixes the statistics *and* earns you the "unsupervised" claim honestly.
3. **Migrate off SAM 1 + CLIP to SAM 3**, because your method needs the presence head and the semantic head.

The roadmap in `ROADMAP.md` sequences all of this across 12 weeks.
