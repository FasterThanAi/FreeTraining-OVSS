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
| **2. Baseline** | 4–5 | Reproduce SegEarth-OV3 | ✅ **DONE — LoveDA 47.38 vs paper 47.4** |
| **3. Build** | 6–9 | Implement the co-occurrence method | End-to-end pipeline producing masks |
| **4. Prove** | 10–11 | Evaluation + ablations | Tables + figures showing you beat the baseline |
| **5. Write** | 12 | Report / paper | Submitted draft |

Weeks 2–5 have **parallel tracks** — theory in the morning, code in the evening. Don't serialise them.

---

# PHASE 0 — Ignition (Week 1, ~25 hrs)

The single highest-value thing you can do in week one is get SAM 3 producing a mask on a satellite image. It de-risks everything and it makes the rest of the project concrete instead of abstract.

## 0.1 Environment (Day 1–2)

**See `SETUP_SAM3.md` for the full walkthrough, and `scripts/setup_env.sh` to rebuild it.**

> ⚠️ **The SAM 3 README's stated requirements (Python 3.12+, torch 2.7+, CUDA 12.6+) do not
> work for this project.** MMSegmentation is also required — SegEarth-OV3 is built on it —
> and mmcv has no prebuilt wheel above torch 2.4. Three constraints intersect at exactly one
> point:
>
> | Component | Constraint |
> |---|---|
> | SAM 3 | torch **≥ 2.3** (`torch.nn.attention`) |
> | mmcv prebuilt wheels | torch 2.1–2.4 only; **none for 2.5+** |
> | mmsegmentation 1.2.2 | `mmcv >= 2.0.0rc4, < 2.2.0` |
>
> **Verified working (18 Aug 2026): Python 3.11 · torch 2.4.1+cu121 · mmcv 2.2.0 · mmseg 1.2.2
> with `MMCV_MAX` patched to 2.3.0.** Runs fine on a CUDA 13.3 driver — the driver version is
> a ceiling, not a match requirement.

```bash
conda create -n segov3 python=3.11 -y && conda activate segov3
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install "mmsegmentation==1.2.2"
sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/" \
  $CONDA_PREFIX/lib/python3.11/site-packages/mmseg/__init__.py
pip install einops psutil pycocotools hydra-core iopath timm huggingface_hub omegaconf
```

**Do not use `mmcv-lite`, and do not build mmcv from source.** Both were tried and both fail —
see `SETUP_SAM3.md §1` for the failure mode of each.

**Checkpoints are gated.** Request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) — do this on **day one**, approval is not instant. Then `hf auth login`. Mirror: [ModelScope](https://modelscope.cn/models/facebook/sam3).

**Use SAM 3.1** if starting fresh — improved checkpoints released 27 Mar 2026, separate repo: [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1). Requires latest repo code.

**VRAM — measured, not estimated:** LoveDA evaluation at native 1024×1024 peaks at **6115 MB** on an RTX 2000 Ada (16 GB). No OOM, no fp16, no tiling needed. That leaves ~10 GB of headroom for DINOv3 features and cached region embeddings later.

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

## 2.1 Reproduce SegEarth-OV3 — ✅ **DONE (18 Aug 2026)**

```bash
ln -s ~/data/loveda data/LoveDA
python eval.py ./configs/cfg_loveda.py
```

**Result: mIoU 47.38 against the paper's 47.4** — a difference of 0.02, i.e. rounding.
1669 LoveDA val images, 0.85 s/image, ~24 min wall, 6115 MB peak. Full numbers and the
per-class table are in `WEEK1_RESULTS.md`.

The gate is passed. Every subsequent number is now measured against a baseline reproduced on
this machine rather than quoted from a paper.

**What the per-class breakdown showed** (this is the important part, not the headline number):

| Class | Precision | Recall | Gap |
|---|---|---|---|
| water | 89.5 | 54.7 | **+34.8** |
| forest | 57.9 | 44.8 | +13.1 |
| background | 56.9 | 69.4 | **−12.5** |
| building | 77.2 | 78.6 | −1.4 |

SAM 3 is right ~90% of the time it says "water" but finds only half of it. Background shows
the inverse — over-predicted and impure, absorbing pixels from real classes. That asymmetry
is the τ = 0.5 threshold discarding low-confidence pixels, and it is concentrated in
amorphous "stuff"; sharp-boundary "things" are balanced. **The project premise is visible in
the baseline's own metrics.**

> Note: reproduction was done on **LoveDA**, not OpenEarthMap as originally planned. LoveDA
> is the better first target — smaller, urban/rural split for the cross-domain experiment,
> and both SegEarth-OV and SegEarth-OV3 report on it. Run OpenEarthMap next for a second
> reference point.

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
| 19 | **ConInfer** ([arXiv:2603.29271](https://arxiv.org/abs/2603.29271)) | ⚠️ **Your closest competitor — read first.** Claims *first* context-at-inference for OVRSS: DINOv3 features → GMM → KL-consensus fusion with a VLM prior. +2.80 mIoU over SegEarth-OV. Purely **visual** context, **patch**-level, **CLIP**-based. No semantic class-pair prior. Its own limitations section names pixel/region-level contextual modelling as future work — that is your opening. |
| 20 | **Towards Realistic Open-Vocabulary RS Segmentation** ([arXiv:2604.15652](https://arxiv.org/html/2604.15652)) | OVRSISBenchV1 — the unified 2026 eval protocol. Evaluate on it if you can. |
| 21 | **SkySense-O** (Zhu+ 2025) | RS-specific vision-language pretraining — the "just train on RS data" counter-argument you must address. |
| 22 | **AerOSeg** (Dutta+ 2025) | SAM features for structural guidance in aerial imagery. |
| 23 | **RemoteSAM** / **InstructSAM** (Yao+, Zheng+ 2025) | SAM-based unified RS pipelines. |

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
- **M is directed, not symmetric.** ✅ **Closed 21 Aug** — `WEEK1_RESULTS.md` §8.1(b). `water → agricultural` is 19.3M at τ=0.5 with no reverse in any top-8; `forest → agricultural` (23.8M) runs 3× its reverse (7.9M). A symmetric M cannot express "a low-confidence region bordering water is probably not agricultural" independently of the converse. Note `scripts/cooccurrence_gt.py` symmetrises (`M += c + c.T`) — that script is the **GT reference**, and building `M_global` directed means it is no longer directly comparable cell-for-cell. Symmetrise your directed M before the validation-gate comparison below.

Still open, decide and defend:

- **Discriminability weighting** — §4.3. Weight each neighbour by the variance of its PMI row so hub classes like `road` don't drown out exclusive ones like `building`. ⚠️ Note the null-model caveat: §4.3's PMI is computed against an *area*-based independence baseline while the observation is a *boundary* distribution, which systematically inflates high-perimeter classes like `road`. Run a structure-preserving permutation null before relying on this row.
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

---

# PHASE 6 — Publication hardening (added 31 Aug 2026)

**Why this phase exists.** Phases 0–5 are complete and roughly three roadmap-weeks early. The
paper is written (`paper/main.tex`, one `\todo` left). So the question is no longer "can we finish"
but **"what raises the acceptance probability per hour spent"**. Everything below is ranked by
exactly that, with a cost and a hard gate.

**The strategic read.** Both competitor papers come from a group with cluster-scale compute and a
positive method. Competing on "our module beats theirs" from a 70 W-capped RTX 2000 is the losing
game. **Our comparative advantage is rigor** — reproduced baselines, oracle bounds on every
negative, validation gates, negative controls, and a *mechanism* that explains results rather than
reporting them. Every item below plays to that.

| # | item | cost | value | why |
|---|---|---|---|---|
| ~~1~~ | ~~⭐ **Vocabulary intervention**~~ | — | ✅ **done** | §7b/§7c — the claim is causal, the locus is the **label space**, and the effect saturates by ~35% share |
| ~~2~~ | ~~Discard-rate deployment criterion~~ | — | ⛔ **done, NEGATIVE** | no label-free rule exists; §9f, and it unifies with §9d's bound |
| **3** | Catch-all-excluded mIoU as a reported metric | CPU, hours | high | a benchmark recommendation others can adopt |
| **4** | Label-free **coupled** objective via cross-head agreement | CPU, 1 day | high upside | would convert the biggest negative into a method |
| **5** | Pre-registered prediction on a third dataset | data access + 1 GPU h | high | almost nobody pre-registers in CV |
| **6** | ConInfer baseline row | 2–3 days, env risk | medium | the one comparison a reviewer will demand |
| **7** | Bootstrap CIs, cross-dataset transfer, figures | CPU, 1 day | polish | cheap credibility |

---

## 6.1 ⭐ The vocabulary intervention — make the central claim causal

**The problem it fixes.** WEEK3 §7a broke the share/confusability confound by *stratifying* LoveDA,
and states plainly: **"this is a stratification, not a randomised intervention."** Urban and rural
differ in class mix, object scale and scene density too. A reviewer will land on that sentence.
It is currently the weakest joint in the strongest claim.

**The fix.** Catch-all share is a variable **we control** — it is set by the vocabulary handed to
SAM 3. So intervene on it directly instead of finding strata where it happens to vary.

**Design — a dose–response curve with a class-count control.** On OpenEarthMap (catch-all 0.84%),
progressively merge real classes *into* the catch-all, changing **both** the prompt vocabulary and
the ground-truth mapping consistently, and re-measure detection AUC at each dose:

| arm | vocabulary | catch-all share | tests |
|---|---|---|---|
| A₀ | published 8 classes + background | 0.84% | the baseline point |
| A₁ | merge 1 class into `background` | ~10% | dose 1 |
| A₂ | merge 2–3 classes into `background` | ~25% | dose 2 |
| A₃ | merge 4 classes into `background` | ~40% | dose 3 — LoveDA's regime |
| **C** | ⭐ **control**: merge the *same* classes into **each other**, not into the catch-all | stays 0.84% | isolates class-count from share |

**Arm C is the whole experiment.** Merging classes also reduces the class count, which moves mIoU
and the detection base rate on its own. The control changes the class count by exactly the same
amount while leaving catch-all share untouched. If AUC falls along A₀→A₃ and does **not** fall in
C, share is causal and the confound is closed by intervention rather than by argument.

**Also run the reverse on LoveDA:** drop `background` from the prompt vocabulary entirely, so the
vocabulary covers the scene. The mechanism predicts detectability should *rise* toward
OpenEarthMap's regime. A prediction that survives in both directions is very hard to argue with.

> **Gate.** Declare the predicted AUC ordering **in a committed file before the first run**
> (`git log` is the timestamp). Prediction confirmed → §7 is upgraded from "constrains the
> explanation" to "intervention". Prediction fails → that is a genuine finding about the mechanism
> and must be written up as one, not quietly dropped.

⚠️ Cost is real but small: one OEM pass is ~25 min. Five arms is under three hours. **This is by
far the best use of remaining GPU time in the project.**

---

## 6.2 The discard-rate deployment criterion — a label-free rule for when calibration pays

**The observation.** §9e found the calibration gain is +2.77 on rural and +0.10 on urban, and those
strata differ 2× in discard rate (39.3% vs 18.5%). Domain and residual size are confounded there
in precisely the way share and confusability were before §7a broke them.

**The experiment.** Stratify all 1669 LoveDA tiles by **discard rate** into quantiles, *ignoring*
the domain label, and run the same 5-fold within each stratum. Then check whether the gain tracks
the discard rate or the urban/rural tag.

**⭐ Why this is worth more than it looks.** Discard rate is **measurable with no ground truth at
all** — it is just the fraction of pixels the model assigns to background. If the gain tracks it,
the paper ends with a rule a practitioner can apply *before* paying for any annotation:

> measure your discard rate; above *X*%, per-class calibration repays ~200 labelled tiles;
> below it, do not bother.

That converts a scope limitation into a **deployment criterion**, and makes the paper *predictive*
rather than descriptive. Reviewers reward papers that predict. OpenEarthMap (3.78% discard, full
mIoU flat) is a fifth point that either falls on the curve or does not.

CPU-only; reuses `tau_domain.py` almost unchanged — the domain map becomes a quantile split.

---

## 6.3 Catch-all-excluded mIoU — propose the metric, don't just complain about it

We now have the catch-all artefact measured **in both directions**:

| | catch-all | real classes | full mIoU | effect |
|---|---|---|---|---|
| OpenEarthMap recovery (§8.1) | **+22.67** | −2.11 | **+2.28** | mIoU **inflated** |
| LoveDA-urban calibration (§9e) | **−3.51** | +4.18 | **+0.10** | mIoU **deflated** |

Same mechanism, opposite sign, two datasets. That is enough to make a **recommendation** rather
than an observation: open-vocabulary segmentation benchmarks with a catch-all class should report
**mIoU over the real classes** alongside full mIoU, because full mIoU can move substantially in
either direction without land-cover quality changing at all.

**Cost: hours.** Recompute every existing table with the extra column — the histogram cache makes
it arithmetic. **Value: high**, because a concrete, adoptable metric recommendation is a
contribution reviewers can point at, and it generalises past remote sensing.

---

## 6.4 A label-free **coupled** objective — the one attempt left that is not already bounded

**Why the eleven failures don't cover it.** §9d's bound is precise: the optimal per-class τ solves a
**coupled** multi-class IoU objective, so *no per-class scalar can express it*. Every one of the
eleven label-free attempts was a per-class **scalar**. None was a coupled **objective**. That gap
is still open, and it is the intellectually correct next move.

**The concrete version.** The cache now stores each head's own top-1 separately (`iconf/ipred`,
`sconf/spred`, added for §9d). So fit the threshold **vector** by coordinate ascent maximising
**agreement between the semantic and instance heads** — a genuinely multi-class, coupled objective,
computed with **no labels at all** — and evaluate the resulting vector against ground truth.

- **Gate:** beat the random-proxy control's p95 of **+0.58** on held-out tiles. Anything less is a
  twelfth bounded failure.
- **If it clears the gate**, the paper's largest negative becomes a method, and the framing changes
  from *"no label-free rule reaches the oracle"* to *"the reason is coupling, and here is the
  coupled rule that does."* That is a substantially stronger paper.
- **If it fails**, it is still the twelfth attempt and the *first* to test the coupled family, which
  closes the bound properly instead of leaving a reviewer to ask about it.

CPU-only. High upside, and publishable either way — which is the rare combination worth spending a
day on.

---

## 6.5 Pre-register the mechanism's prediction on a third dataset

The mechanism predicts the residual's size **and** detection AUC from catch-all share alone. So:

1. `predict_dataset.py` computes the prediction from **ground-truth masks only** — no GPU.
2. **Commit the prediction**, so `git log` timestamps it before any inference runs.
3. Then run the baseline and compare.

⛔ **iSAID does not qualify** — 97.11% catch-all *and* maximally confusable, so it confirms the
ordering and discriminates nothing. Choose a dataset whose catch-all is **common but visually
distinct**: ISPRS Potsdam / Vaihingen (`clutter`) is the nearest candidate, and its ~5% share lands
in the empty gap between OpenEarthMap's 0.84% and LoveDA-urban's 26%.

⚠️ **ISPRS access needs a registration form — start it on day one of this phase**, because it is the
only slow item in the whole plan and everything else is CPU-bound.

---

## 6.6 The ConInfer row

`github.com/Dog-Yang/ConInfer` — the nearest published competitor, and the one comparison a
reviewer will ask for by name.

⚠️ **It needs its own conda environment. Never touch `segov3`** — the three-way version deadlock in
`WEEK1_RESULTS.md` §2 is the only working combination and rebuilding it costs a day.

**Timebox: 3 days.** If their code does not run in that window, cite it, position against it in
related work (visual vs semantic context, patch vs region, per-scene vs corpus) and say plainly in
limitations that a direct comparison was not run. That is an acceptable outcome; a wrecked
environment is not.

---

## 6.7 Cheap credibility

- **Bootstrap CIs** on the headline gains, not just fold sd.
- **Cross-dataset threshold transfer** (LoveDA → OpenEarthMap). §9e shows it fails *within* LoveDA;
  completing the picture across datasets costs a numpy pass.
- **A qualitative figure of the urban/rural threshold disagreement** — `road` at 0.725 vs 0.225 is
  a picture, not a number.
- **Fill the last `\todo`** in `paper/main.tex` (author block).

---

## Phase 6 ordering

| when | do |
|---|---|
| **day 1** | start the ISPRS registration (slow path); write and **commit** the §6.1 predictions |
| **days 1–2** | §6.1 vocabulary intervention — the GPU work, ~3 h |
| **days 2–3** | §6.2 discard-rate criterion and §6.3 metric column — CPU |
| **day 4** | §6.4 coupled label-free objective — CPU, gated at +0.58 |
| **days 5–7** | §6.5 third dataset if access arrived; else §6.6 ConInfer, timeboxed |
| **ongoing** | §6.7 polish; keep `paper/numbers.tex` the only place a number is typed |

> ⭐ **THE STOP RULE HAS FIRED — 1 Sep.** §6.1 landed (causal, with an arity control and a
> pre-registration) and §6.2 landed as a negative that unifies with §9d rather than weakening it.
> **Write and submit.** §6.3 (the metric column) is hours and worth doing; §6.4–§6.6 are optional
> and must not delay the deadline.

**Stop rule (as written in advance).** If §6.1 and §6.2 both land, the paper is materially stronger than it is today and
**submission should not wait** for §6.5 or §6.6. Shipping a strong workshop paper beats polishing a
better one past the deadline.

---

---

# PHASE 7 — Six months, not six days (added 1 Sep 2026)

⚠️ **The Phase 6 stop rule was written assuming a near deadline. There isn't one.**
EarthVision 2027 runs with CVPR 2027; the submission date is **not yet officially published**, and
past years put it in **early March**. Check the real CFP when it posts (usually with the CVPR
workshop list, Nov–Dec 2026). Planning assumption: **~6 months from 1 Sep 2026.**

**What that changes.** "Ship the strong workshop paper" was right against a two-week horizon. Against
six months it is *under*-ambitious: the two objections a reviewer will actually raise — no ConInfer
comparison, only two datasets — are both fixable in weeks, and fixing them opens venues the paper
cannot currently reach.

## The realistic venue ladder, given six months

| venue | deadline | fit today | fit after §7.1–§7.2 |
|---|---|---|---|
| **IEEE GRSL** | rolling | plausible but 5 pages cannot hold it | — |
| **IGARSS 2027** | ~Jan 2027 | good | good; a fast second option |
| **CVPR EarthVision 2027** | ~Mar 2027 *(verify)* | **good** | **strong** |
| **IEEE TGRS / JSTARS** | rolling | a stretch — wants 4+ datasets, more baselines | **plausible** |

⭐ TGRS is worth naming explicitly: it is a strong journal, it is **rolling** so there is no deadline
to miss, and it is the natural home for a measurement paper with a causal experiment. Reaching it
needs exactly the two things below and nothing conceptually new.

## §7.1 The ConInfer comparison — do it first

The nearest published competitor, code public, cited in our paper **without a number**. It is the
first question a reviewer asks and currently it goes unanswered.

⚠️ **Its own conda environment. Never touch `segov3`** — the three-way version deadlock in
`WEEK1_RESULTS.md` §2 is the only working combination and rebuilding it costs a day.

**Timebox: one week** (was three days under deadline pressure; there is room now). If it will not run,
that is still a result: say so in limitations with what was tried.

## §7.2 Datasets three and four

ISPRS **Potsdam** and **Vaihingen**. Their catch-all is `clutter` at roughly 5% — the empty gap
between OpenEarthMap's 0.84% and LoveDA-urban's 26%.

⚠️ **Registration is the slow path. Submit it this week**, before anything else in this phase.

Two things to run on each, and the order matters:
1. **Pre-register the prediction first** (`predict_dataset.py`, ground-truth masks only, no GPU),
   commit it, *then* run the baseline. §6.5.
2. **Repeat the vocabulary intervention** (§6.1). Establishing the causal claim on a *second*
   dataset is worth more than a third observational point, and the cache work is already written.

## §7.3 The coupled label-free objective — the one idea still open

§9d bounds every *per-class scalar*. It does not bound a **coupled objective**, and the cache now
stores each head separately, so fitting the whole τ vector to maximise cross-head agreement is a day
of CPU. Gate: beat the random control's **+0.58**. Publishable either way — clearing it turns the
paper's largest negative into a method.

## §7.4 The writing pass

75 em-dashes, one per 3.6 sentences, against a healthy one per 10–15. Four sentences exceed 55
words. An hour of work that improves every page.

## ⛔ The content freeze

**1 January 2027.** After that date, no new experiments — only writing, figures and revision. Six
months is enough to do three good things and far too much time to keep finding a fourth. A project
that misses a March deadline because it was still measuring in February has made an avoidable
mistake, and this roadmap has warned about it since Phase 4.

| | |
|---|---|
| **Sep** | Potsdam registration · ConInfer week · writing pass |
| **Oct–Nov** | datasets 3–4: pre-register, baseline, intervention |
| **Dec** | §7.3 coupled objective · assemble the multi-dataset tables |
| **Jan 1** | ⛔ **content freeze** |
| **Jan–Feb** | write, cut to length, internal review |
| **Mar** | submit |

---

## Milestones — if you miss these, replan immediately

| End of week | Must be true |
|---|---|
| 1 | SAM 3 segments an aerial tile from a text prompt — **✅ done 12 Aug** |
| 3 | GT co-occurrence heatmap computed — **✅ done 12 Aug, premise validated (`ANALYSIS.md §4`)**; mIoU implemented and trusted |
| 5 | SegEarth-OV3 reproduced within 2% — **✅ done 18 Aug, LoveDA 47.38 vs 47.4 (Δ 0.02)** |
| 7 | M_global built and validated against GT |
| 9 | End-to-end pipeline produces a number |
| 11 | Main table + ablations complete |
| 12 | Draft submitted |

## Risks, ranked by likelihood

1. ~~**mmsegmentation dependency hell**~~ — **hit, and resolved (18 Aug).** Cost most of a
   day. The repo pins no versions, so the working combination had to be found by elimination:
   torch 2.4.1 + mmcv 2.2.0 (prebuilt wheel) + mmseg 1.2.2 with `MMCV_MAX` patched.
   `mmcv-lite` and source builds both fail. Captured in `scripts/setup_env.sh` — **do not
   rebuild the env by hand.**
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
