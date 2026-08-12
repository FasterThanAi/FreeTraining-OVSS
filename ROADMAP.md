# 12-Week Roadmap — Training-Free Open-Vocabulary Segmentation for Remote Sensing

**Your profile:** Python + basic ML · ~25 hrs/week · 3 months · local GPU
**Total budget:** ~300 hours
**Read `ANALYSIS.md` first** — it explains why the project needs repositioning.

---

## The one honest caveat

You cannot learn all of computer vision, deep learning and vision-language modelling *and* produce original research in 300 hours. Anyone selling you a "learn CV then do the project" curriculum is setting you up to spend 10 weeks on foundations and 2 weeks panicking.

So this roadmap uses **just-in-time learning**. You learn each concept in the week you need it, driven by the code you're about to touch. Foundations are ~25% of your time, not 70%. The deep theory you skip now, you can fill in afterwards — the project is the forcing function.

**The rule:** if a concept doesn't change a line of code you'll write in the next 3 weeks, defer it.

---

## Phase overview

| Phase | Weeks | Goal | Exit criterion |
|---|---|---|---|
| **0. Ignition** | 1 | SAM 3 running on your GPU | You've segmented an aerial image with a text prompt |
| **1. Foundations** | 2–3 | Enough DL/CV to read the code | You can explain what a ViT patch token is |
| **2. Baseline** | 4–5 | Reproduce SegEarth-OV3 | Your mIoU on OpenEarthMap matches their paper ±2% |
| **3. Build** | 6–9 | Implement the co-occurrence method | End-to-end pipeline producing masks |
| **4. Prove** | 10–11 | Evaluation + ablations | Tables + figures showing you beat the baseline |
| **5. Write** | 12 | Report / paper | Submitted draft |

Weeks 2–5 have **parallel tracks** — theory in the morning, code in the evening. Don't serialise them.

---

# PHASE 0 — Ignition (Week 1, ~25 hrs)

The single highest-value thing you can do in week one is get SAM 3 producing a mask on a satellite image. It de-risks everything and it makes the rest of the project concrete instead of abstract.

## 0.1 Environment (Day 1–2)

**See `SETUP_SAM3.md` for the full Linux / lab-PC walkthrough.** Short version — note the hard requirements from the official README:

> Python **3.12+** · PyTorch **2.7+** · CUDA-compatible GPU with **CUDA 12.6 or higher**

```bash
conda create -n sam3 python=3.12 -y && conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git && cd sam3
pip install -e ".[notebooks]"
```

**Checkpoints are gated.** Request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) — do this on **day one**, approval is not instant. Then `hf auth login`. Mirror: [ModelScope](https://modelscope.cn/models/facebook/sam3).

**Use SAM 3.1** if starting fresh — improved checkpoints released 27 Mar 2026, separate repo: [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1). Requires latest repo code.

**VRAM:** the model is 848M parameters. Image inference is comfortable on 16 GB and workable on 8–12 GB with FP16 + tiling. Check `nvidia-smi` — it constrains your tile size.

**Optional speedups:** `flash-attn-3`, `einops`, `ninja`, `cc_torch`. Skip until something is actually slow.

## 0.2 First contact (Day 3–4)

Run the official notebooks. Then, critically, **prompt it with remote sensing imagery** — grab a few OpenEarthMap tiles and try `"building"`, `"road"`, `"water"`, `"bareland"`.

Your goal is to *see with your own eyes* the phenomenon SegEarth-OV3 Figure 1 describes: the instance head fragments roads, the semantic head blurs cars. Everything in your project follows from that observation.

## 0.3 Get inside the outputs (Day 5–7)

Write a scratch script that, for one image and one text prompt, prints and visualises:

- `S_pres` — the scalar presence score
- `P_sem` — the dense semantic map
- `{(P_inst^k, s_conf^k)}` — every instance query and its confidence

**This is the most important exercise of week one.** Your entire method operates on these three tensors. Know their shapes, ranges, and dtypes cold.

```bash
git clone https://github.com/earth-insights/SegEarth-OV-3.git
# read demo.py and segearthov3_segmentor.py line by line
```

> **Deliverable:** a notebook showing SAM 3 masks on 5 aerial tiles, with presence scores printed per class.

---

# PHASE 1 — Foundations (Weeks 2–3, ~50 hrs)

Theory track ~15 hrs/wk, code track ~10 hrs/wk, running concurrently.

## What you actually need (and what you don't)

**Need:** convolutions · encoder-decoder segmentation · ViT and patch tokens · attention · contrastive vision-language training · what a "query" is in DETR · mIoU · superpixels · basic probability (conditional probability, Bayes, Dirichlet smoothing)

**Defer:** RNNs/LSTMs · GANs · diffusion · RL · NAS · most of classical CV · training tricks (you're training-free!) · distributed training

**Note what's missing from that list: backprop internals and optimiser theory.** You are not training a model. You need to *read* architectures, not train them. This is a huge saving — spend the time on inference, tensors and evaluation instead.

## Theory track — pick ONE per row, don't collect them all

| Topic | Primary (free) | Alternative | Hours |
|---|---|---|---|
| **Neural nets, ground up** | [Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) (first 3 videos) | [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) for intuition | 8 |
| **Practical DL** | [fast.ai — Practical Deep Learning](https://course.fast.ai/) (lessons 1–3) | [MIT 6.S191](http://introtodeeplearning.com/) | 10 |
| **CV / segmentation** | [Michigan EECS 498-007 — Deep Learning for CV, Justin Johnson](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) — **lec 1–8** (fundamentals + CNNs), **lec 13** (attention), **lec 16 "Detection and Segmentation"** ← the critical one. Skip 12 (RNNs), 19–22. | [Stanford CS231n](http://cs231n.stanford.edu/) — same lineage, older | 15 |
| **Transformers / ViT** | [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) + [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | [Umar Jamil](https://www.youtube.com/@umarjamilai) — codes papers line by line | 10 |
| **Classical CV (superpixels)** | [First Principles of CV — Shree Nayar, Columbia](https://www.youtube.com/@firstprinciplesofcomputerv3258) (segmentation lectures only; [companion PDFs](https://fpcv.cs.columbia.edu/)) | [Szeliski's book](https://szeliski.org/Book/), ch. 7 | 4 |
| **Reference, not linear reading** | [Dive into Deep Learning (d2l.ai)](https://d2l.ai/) — free, PyTorch code inline | — | as needed |

**Books** — treat as lookup, not cover-to-cover:

- **[Dive into Deep Learning](https://d2l.ai/)** — best single free reference; every concept has runnable PyTorch.
- **[Szeliski, *Computer Vision: Algorithms and Applications*](https://szeliski.org/Book/)** — free PDF. Chapter 7 (segmentation) is your SLIC/Felzenszwalb reference.
- **[Mathematics for Machine Learning](https://mml-book.github.io/)** — free PDF. Only if the linear algebra bites you.
- *Deep Learning with PyTorch* (Stevens et al.) — free from Manning. If you want one hands-on PyTorch book.

**Skip Goodfellow's *Deep Learning*.** It's a great book and completely wrong for a 3-month deadline.

## Code track (concurrent)

- Implement mIoU from scratch. Not from a library — by hand, from a confusion matrix. You will debug evaluation bugs all project long and you must trust this function absolutely.
- Load OpenEarthMap. Write a dataloader, visualise images with their GT masks, count class frequencies.
- **Compute the ground-truth co-occurrence matrix on OpenEarthMap.** Which classes actually touch which? This is a 40-line script and it will tell you within two hours whether your entire thesis has signal in it. If land-cover adjacency turns out to be near-uniform, you need to know that in week 2, not week 9.

> **Deliverable:** a heatmap of GT class co-occurrence for OpenEarthMap. This figure probably ends up in your final report.

---

# PHASE 2 — Baseline (Weeks 4–5, ~50 hrs)

## 2.1 Reproduce SegEarth-OV3

```bash
# mmcv + mmsegmentation are the only fussy dependencies — their words, and they're right
python eval.py ./configs/cfg_OpenEarthMap.py
```

Budget real time for mmsegmentation install pain. Match the versions in their repo exactly; do not improvise.

**Success = your mIoU lands within ~2% of their published number.** Until that holds, every later number you produce is meaningless. Do not proceed past this gate.

## 2.2 Papers — tiered reading list

Read in order. **Skim tier 0 for architecture only** — you need to recognise components, not reproduce them.

### Tier 0 — Architectural lineage (~10 hrs, skim)

| # | Paper | Why it matters to *you* |
|---|---|---|
| 1 | **FCN** (Long+ 2015) | Where dense prediction started. 30 min. |
| 2 | **U-Net** (Ronneberger+ 2015) | Still the RS workhorse; you'll see it everywhere in baselines. |
| 3 | **ViT** — *An Image is Worth 16×16 Words* (Dosovitskiy+ 2020) | Patch tokens are the unit SAM 3 and CLIP both operate on. Non-negotiable. |
| 4 | **DETR** (Carion+ 2020) | Object queries + set prediction. **Nicolas Carion is also first author on SAM 3** — the instance head is his design lineage. Read properly, not skim. |
| 5 | **MaskFormer / Mask2Former** (Cheng+ 2021, 2022) | Mask classification. SAM 3's decoder is built on this. |
| 6 | **CLIP** (Radford+ 2021) | The whole open-vocabulary paradigm. Read properly. |
| 7 | **SAM** (Kirillov+ 2023) | *The paper you thought you had.* Actually download it. |
| 8 | **SAM 2** (Ravi+ 2024) | Skim — mostly video/memory, less relevant to you. |
| 9 | **SAM 3** (Carion+ 2025) | Your model. Read three times. |
| 10 | **DINOv2** (Oquab+ 2023) | Self-supervised features used for structural guidance across this literature. |

### Tier 1 — Training-free OVSS lineage (~8 hrs)

| # | Paper | Why |
|---|---|---|
| 11 | **MaskCLIP** (Zhou+ 2022) | First to pull dense features out of CLIP. Origin of the field. |
| 12 | **SCLIP** (Wang+ 2023) | Correlative self-attention. SegEarth-OV3's code is *based on* SCLIP's repo. |
| 13 | **ClearCLIP** (Lan+ 2024) | The "remove pooling" line of attack. |
| 14 | **ProxyCLIP** (Lan+ 2025) | DINO structural guidance — the VFM-assisted paradigm. |
| 15 | **CorrCLIP** (Zhang+ 2025) | Same family, refines attention via SAM. |
| 16 | **GroupViT** (Xu+ 2022) | Weakly-supervised grouping — conceptually closest to your clustering step. |

### Tier 2 — Your direct competition (~10 hrs, read deeply)

| # | Paper | Why |
|---|---|---|
| 17 | **SegEarth-OV** (CVPR 2025) | ✅ You have it. Source of your dataset protocol. |
| 18 | **SegEarth-OV3** (arXiv:2512.08730) | ✅ You have it. **Your baseline. Read until you could re-derive equations 1–3 from memory.** |
| 19 | **Towards Realistic Open-Vocabulary RS Segmentation** ([arXiv:2604.15652](https://arxiv.org/html/2604.15652)) | OVRSISBenchV1 — the unified 2026 eval protocol. Evaluate on it if you can. |
| 20 | **SkySense-O** (Zhu+ 2025) | RS-specific vision-language pretraining — the "just train on RS data" counter-argument you must address. |
| 21 | **AerOSeg** (Dutta+ 2025) | SAM features for structural guidance in aerial imagery. |
| 22 | **RemoteSAM** / **InstructSAM** (Yao+, Zheng+ 2025) | SAM-based unified RS pipelines. |

### Tier 3 — Context modelling (your novelty — thin literature, go mining)

This is where you have to do original library work. Nobody has written the paper you're writing, so read adjacent:

- **SLIC** (Achanta+ 2012) and **Felzenszwalb–Huttenlocher** (2004) — your superpixel options, if you use them at all.
- **DeepLab** CRF post-processing (Chen+ 2017) and **CRF-as-RNN** (Zheng+ 2015) — the classical answer to "use spatial context to fix segmentation." Your co-occurrence prior is a modern, semantic cousin. **You must position against these.**
- **EncNet** — *Context Encoding for Semantic Segmentation* (Zhang+ 2018) — global context as an explicit prior.
- **Search terms for your literature review:** `semantic context prior segmentation`, `label co-occurrence statistics scene parsing`, `region adjacency graph merging segmentation`, `spatial context land cover classification`, `graph label propagation semantic segmentation`, `Markov random field land cover`.

Use [Connected Papers](https://www.connectedpapers.com/) and [Semantic Scholar](https://www.semanticscholar.org/) on SegEarth-OV3 to snowball citations forward from Dec 2025.

> **Deliverable:** annotated bibliography, ~1 paragraph per paper: what it does, what it gets right, what it leaves open for you.

---

# PHASE 3 — Build (Weeks 6–9, ~100 hrs)

Build in this order. Each module is independently testable — **write the test as you go**, because debugging a 5-stage pipeline end-to-end is misery.

## Week 6 — Steps 1 & 2: extraction and grouping

Fork `segearthov3_segmentor.py`. Add an instrumented path that, for each image, emits:

```python
identified   = pixels where P_final > τ_high         # confident, labelled
unidentified = pixels where τ_low < P_final ≤ τ_high # ambiguous → your target
ignored      = pixels where P_final ≤ τ_low          # true background
```

Note the **two** thresholds. SegEarth-OV3 uses one and collapses everything below it into background. Your τ_low/τ_high band *is* your working set — the residual you're recovering. Sweep both later.

Extract per-patch: mask, centroid, area, class label, confidence, and the pooled `F_cond` embedding (your "INFO").

## Week 7 — Step 3: the co-occurrence matrix

Implement per `ANALYSIS.md §3.1`:

```python
M_global = accumulate_over_corpus(all_identified_patches)  # Dirichlet-smoothed
M_image  = build_from_single_image(identified_patches)
M_eff    = lam * M_global + (1 - lam) * M_image
```

Three of these decisions are already **settled by measurement** — see `ANALYSIS.md §4`:

- **Adjacency = shared boundary length.** Not centroid distance (what your old repo used). Already implemented in `scripts/cooccurrence_gt.py`.
- **Store signed PMI, not raw counts.** §4.2 — exclusion carries the largest magnitudes (building–water at 0.02× chance), and additive-positive scoring cannot represent it.
- **λ blend is required, not optional.** §4.4 — six of fifteen class pairs flip sign between urban and rural.

Still open, decide and defend:

- **Directed or symmetric?** Current script symmetrises. Land-cover adjacency may genuinely be asymmetric — worth testing.
- **Discriminability weighting** — §4.3. Weight each neighbour by the variance of its PMI row so hub classes like `road` don't drown out exclusive ones like `building`.
- **Scale sensitivity** — co-occurrence changes with GSD. A GSD-conditioned M is your stretch contribution.

**Validation gate:** compare `M_global` (estimated from SAM 3's confident predictions) against the ground-truth matrices already in `outputs/cooccurrence/`. Large divergence means §3.2's circularity problem is real. You have the reference numbers, so this is a direct comparison, not a guess.

## Week 8 — Steps 4a & 4b: atoms, then labelling

**Atomise + agglomerate** (per `ANALYSIS.md §3.4`): build a Region Adjacency Graph over unidentified regions and merge on combined feature + co-occurrence compatibility. **Run both variants** — SLIC atoms vs. SAM 3 low-confidence mask atoms. My expectation is SAM 3's masks win; measure it rather than assuming.

**Contextual labelling** — score each candidate class per patch:

```
score(c) = w_emb · sim(INFO_patch, prototype_c)
         + w_coc · Σ_{n ∈ N(patch)} M_eff[label(n), c] · w(n)
         + w_nbr · vote(N(patch), c)
```

Two things your current repo gets wrong and you should fix here:

1. **Scale mismatch.** Cosine similarity ranges [−1, 1]; the other two terms are normalised distributions on [0, 1]. Adding them weighted is comparing apples to oranges and makes any confidence threshold arbitrary. Normalise all three to comparable distributions — softmax over classes, or z-score — *before* combining.
2. **Iterate.** Label the highest-confidence unidentified patches first, then treat them as identified and re-run. Confidence-ordered propagation across the RAG will beat one-shot assignment. This is essentially label propagation and it's cheap to add.

**Run the embedding ablation this week** (`ANALYSIS.md §3.3`). If `w_emb → 0` costs you nothing, delete the term and simplify the paper.

## Week 9 — Step 5 + integration

Fuse, resolve conflicts by confidence, optionally smooth boundaries. Then run end-to-end on OpenEarthMap and get your **first real number**.

Two engineering notes that will save you days:

- **Cache SAM 3 outputs to disk.** You'll re-run downstream stages hundreds of times during ablations; re-running the encoder every time is the difference between a 3-minute and a 3-hour experiment loop.
- **Seed everything and log every config.** Use Hydra or just dump the YAML alongside each result. Ablation tables assembled from unlabelled runs are how projects die in week 11.

> **Deliverable:** end-to-end pipeline, one mIoU number on OpenEarthMap, compared against your reproduced baseline.

---

# PHASE 4 — Prove (Weeks 10–11, ~50 hrs)

Per `ANALYSIS.md §6`.

**Week 10 — main results.** Three-row table (SAM 3 instance-only → SegEarth-OV3 → +yours) across OpenEarthMap, LoveDA, Potsdam. Then the table that actually sells the paper: **mIoU restricted to pixels the baseline assigned to background.** That isolates exactly what you added.

**Week 11 — ablations and analysis.** M_global vs M_image vs hierarchical · λ sweep · each scoring term on/off · SLIC vs SAM 3 atoms · τ_low/τ_high sensitivity · the estimated-vs-GT co-occurrence divergence figure.

Then the **cross-domain experiment**, which is more interesting than it sounds: build M on LoveDA-urban, apply to LoveDA-rural. Does the co-occurrence prior transfer? Either answer is publishable — transfer means you've found a general land-cover grammar, failure means the prior is domain-specific and that's a real finding too.

**Qualitative figures.** Side-by-side: image / GT / baseline / yours, with the recovered regions highlighted. Reviewers look at figures before tables. Budget real time here.

---

# PHASE 5 — Write (Week 12, ~25 hrs)

Standard structure: Intro (the residual-discarding problem) → Related Work (your Phase 2 bibliography) → Method (equations for M, the scoring function, propagation) → Experiments → Ablations → Limitations → Conclusion.

**Write the limitations section honestly** — circularity, GSD sensitivity, dependence on SAM 3's calibration. It is the section that most reliably distinguishes a strong project from a mediocre one, and examiners read it closely.

Overleaf + the CVPR or IEEE TGRS template. Start the LaTeX skeleton in **week 9**, not week 12 — write method text while the code is fresh.

---

## Milestones — if you miss these, replan immediately

| End of week | Must be true |
|---|---|
| 1 | SAM 3 segments an aerial tile from a text prompt — **done 12 Aug** |
| 3 | GT co-occurrence heatmap computed — **done 12 Aug, premise validated (`ANALYSIS.md §4`)**; mIoU implemented and trusted |
| 5 | SegEarth-OV3 reproduced within 2% on OpenEarthMap |
| 7 | M_global built and validated against GT |
| 9 | End-to-end pipeline produces a number |
| 11 | Main table + ablations complete |
| 12 | Draft submitted |

## Risks, ranked by likelihood

1. **mmsegmentation dependency hell** (very likely) — budget 2 full days in week 4. Use their exact pinned versions. Docker if you have it.
2. **HF checkpoint access delay** (likely) — request on day 1.
3. **Co-occurrence signal is weak** (possible) — the Phase 1 GT heatmap tells you by week 3, cheaply. If adjacency is near-uniform, pivot to the RAG-agglomeration contribution instead.
4. **Baseline won't reproduce** (possible) — open an issue on their repo early; the authors are active (205 stars, 12 open issues).
5. **Scope creep into change detection / 3D** (likely, and seductive) — **don't**. 2D semantic segmentation on 3 datasets, done properly, beats four half-finished tasks.

## Weekly rhythm that works

- **Mon–Tue:** theory / papers (fresh brain on hard reading)
- **Wed–Fri:** implementation
- **Sat:** experiments running + write down what you learned in a running lab notebook
- **Sun:** off. Three months at 25 hrs/week is a marathon and burnout in week 8 costs more than a rest day.

Keep a `LOGBOOK.md` in this repo — one entry per working day, what you tried, what broke, what the number was. In week 12 it becomes your paper's experiment section, and it turns "I can't remember which config gave 47.2" from a crisis into a `grep`.

---

## Sources

- [SegEarth-OV3 — code](https://github.com/earth-insights/SegEarth-OV-3) · [arXiv:2512.08730](https://arxiv.org/abs/2512.08730)
- [SAM 3 — official repo](https://github.com/facebookresearch/sam3) · [checkpoints](https://huggingface.co/facebook/sam3)
- [SegEarth-OV — dataset preparation](https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md)
- [Meta SAM 3 overview (Roboflow)](https://blog.roboflow.com/what-is-sam3/) · [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3)
- [Towards Realistic Open-Vocabulary RS Segmentation — OVRSISBenchV1](https://arxiv.org/html/2604.15652)
