# CV study plan — scoped to this project

*Companion to `ROADMAP.md`. Written 12 Aug 2026.*

The organising principle: **you are not training a model.** You are reading tensors out of a frozen foundation model and doing statistics on them. That removes roughly half of a standard CV curriculum. Everything below is scoped to what you will actually touch.

Each concept is tied to the exact place it appears in your work. If you can't point at a line of code or a paper equation, it's not on this list.

---

## Skip list — read this first

You will be tempted by these. Don't. They are ~60% of a normal CV course and none of them touch your project.

| Skip | Why |
|---|---|
| Backprop derivations, autograd internals | You never call `.backward()` |
| Optimizers (SGD/Adam), LR schedules, weight init | Nothing is trained |
| Regularisation, dropout, batchnorm theory | Inference only, model is frozen |
| Data augmentation | No training set |
| GANs, diffusion, NeRF, generative models | Unrelated |
| RNNs, LSTMs, seq2seq | Unrelated |
| SIFT/SURF/ORB, epipolar geometry, camera calibration, stereo, optical flow | Classical CV, unrelated to OVSS |
| Distributed training, mixed-precision *training*, quantisation | Single GPU inference |
| Reinforcement learning | Unrelated |

Two exceptions worth 30 minutes each: **mixed-precision inference** (you already hit the bf16 bug) and **batchnorm vs layernorm** (you'll see `LayerNorm` everywhere in SAM 3).

---

## Tier 0 — needed to read your own output (do this week, ~8 hrs)

You have already printed these tensors. Right now some of them are just shapes. By the end of Tier 0 every number should mean something.

| Concept | Where it appears in *your* work | Best source | Time |
|---|---|---|---|
| Tensor layout, `NCHW` vs `(L,B,C)` | `encoder_hidden_states: (5184,1,256)` — sequence-first, not batch-first | [d2l.ai](https://d2l.ai/) preliminaries chapter | 1h |
| Logits vs probabilities; sigmoid vs softmax | `semantic_seg` range `[-68.5, 14.4]` — these are **logits**. `presence_logit_dec.sigmoid()` → 0.957 | [StatQuest — logistic regression / softmax](https://www.youtube.com/@statquest) | 1h |
| Convolution, stride, receptive field, feature maps | Why `semantic_seg` is `288×288` and not `1024×1024` | Johnson EECS 498 **L7 Convolutional Networks** | 2h |
| Patch embedding, why 72×72 | SAM 3 resizes to 1008, patch size 14 → 1008/14 = **72**. And 72² = **5184** = your sequence length | ViT paper §3 + [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 2h |
| Bilinear interpolation / upsampling | `interpolate(out_masks, (img_h, img_w))` in `sam3_image_processor.py` | Any FCN explainer | 30m |
| mIoU, confusion matrix, per-class IoU | Your only evaluation metric, for 12 weeks | Implement it yourself — see below | 1.5h |

**Do this instead of a lecture on mIoU:** write it from scratch from a confusion matrix, then check against `sklearn` or mmseg on a toy case. You will debug evaluation bugs all term and you must trust this function absolutely.

**Exercise that closes Tier 0:** take your `--raw` dump and write one sentence per tensor explaining what it is and why it has that shape. If you can't explain `(1, 200, 288, 288)`, go to Tier 1.

---

## Tier 1 — needed to reproduce the baseline (weeks 2–5, ~25 hrs)

This is the architecture lineage of SAM 3. You need to *recognise* these designs, not reimplement them.

| Concept | Where it appears | Best source | Time |
|---|---|---|---|
| FCN / encoder-decoder segmentation | SAM 3's semantic head is described as "FCN-style" | Johnson **L16 Detection and Segmentation** | 2h |
| Self-attention, cross-attention, Q/K/V | `TransformerEncoderFusion` fuses image + text features | [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) (first 60 min) | 3h |
| ViT: patches as tokens, CLS token, positional embeddings | `vitdet.py` — the backbone you already stack-traced through | ViT paper + Johnson **L13 Attention** | 3h |
| **DETR: object queries, set prediction, Hungarian matching** | `pred_logits: (1, 200, 1)` — those 200 are **queries**, set in `_create_transformer_decoder(num_queries=200)` | DETR paper (read properly) + [Umar Jamil](https://www.youtube.com/@umarjamilai) | 5h |
| MaskFormer: mask classification vs per-pixel classification | Why SAM 3 predicts 200 masks + labels instead of one label map | MaskFormer paper §3 | 3h |
| CLIP: contrastive image-text, shared embedding space | `language_features: (32,1,256)`; the whole OVSS paradigm | CLIP paper §2–3, then [Umar Jamil — coding a vision-language model from scratch](https://www.youtube.com/@umarjamilai) (he covers CLIP inside that and the Stable Diffusion video; there's no standalone CLIP episode) | 4h |
| Open-vocabulary segmentation paradigm | Your entire field | MaskCLIP → SCLIP → ClearCLIP, in that order | 5h |

**DETR is the one to read properly, not skim.** Nicolas Carion is first author on both DETR and SAM 3; the instance head is directly his lineage. Understanding "what is a query" is the difference between using SAM 3 and understanding it.

---

## Tier 2 — needed to build *your* method (weeks 6–9, ~20 hrs)

This is the part no course teaches, because it's your contribution. Mostly probability and graphs, not deep learning.

| Concept | Where it appears | Source | Time |
|---|---|---|---|
| Conditional probability, joint vs marginal | `P(neighbour = j \| patch = i)` — your row-normalised M | [3Blue1Brown — Bayes](https://www.youtube.com/@3blue1brown) | 1h |
| **Pointwise mutual information** | `ANALYSIS.md §4.2` — already in your code | **Jurafsky & Martin, [Appendix J: PPMI](https://web.stanford.edu/~jurafsky/slp3/J.pdf)** — a whole appendix on exactly this. Context in [Ch 5: Embeddings](https://web.stanford.edu/~jurafsky/slp3/5.pdf) | 2h |
| Add-α / Dirichlet smoothing | Sparse M from few patches — `ANALYSIS.md §3.1` | Jurafsky & Martin, [Ch 3: N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (smoothing sections); [Appendix C: Kneser-Ney](https://web.stanford.edu/~jurafsky/slp3/C.pdf) if you want the sophisticated version | 1.5h |
| Hierarchical / shrinkage estimation | Your `λ·M_global + (1−λ)·M_image` | Search "James–Stein estimator" and "empirical Bayes shrinkage" | 2h |
| Connected components, region adjacency graph | Step 3 agglomeration — the correct replacement for the disjoint-mask IoU-merge bug described in `ANALYSIS.md §3.7` | `skimage.measure.label`, `skimage.graph.RAG` docs — read the source | 3h |
| Superpixels: SLIC, Felzenszwalb | Step 3 atomisation (test against SAM 3 masks) | SLIC paper (2012) + `skimage.segmentation.slic` | 2h |
| Label propagation on graphs | Your iterative confidence-ordered labelling | Zhu & Ghahramani 2002 label propagation | 3h |
| CRFs and why they fell out of favour | You **must** position against these — a reviewer will ask why you didn't just use a CRF | DeepLab paper (CRF post-processing section) + CRF-as-RNN | 3h |
| Cosine similarity, embedding prototypes | Your embedding term, `ANALYSIS.md §3.3` | Any | 30m |

**The CRF question is the one that sinks projects like yours in the viva.** "Isn't your co-occurrence prior just a CRF pairwise potential?" You need a crisp answer. (It isn't — a CRF pairwise potential is typically learned or hand-set and applied at pixel level for smoothing; yours is estimated from corpus statistics at region level and carries semantic class-pair information. But have that ready.)

---

## Tier 3 — remote sensing specifics (~5 hrs, spread out)

Short, but skipping it makes you write things that mark you as an outsider to the field.

- **GSD (ground sample distance)** — LoveDA is 0.3 m/pixel. Co-occurrence structure is scale-dependent; this is your stretch contribution.
- **"Things" vs "stuff"** — the countable/amorphous split. Drives SAM 3's dual-head behaviour and your §4.5 finding.
- **Tiling and edge effects** — 1024×1024 tiles cut objects at borders. Affects your boundary-length adjacency counts.
- **Class imbalance** — LoveDA rural is 53% agriculture. Why mIoU, not pixel accuracy.
- **Why RS ≠ natural images** — overhead view, no canonical orientation, extreme scale variation, stuff-dominated. SegEarth-OV's introduction covers this well; reread it once you know the vocabulary.

---

## Resources, ranked — and honest caveats

**On Reddit:** I could not access it — Anthropic's crawler is blocked from reddit.com. Anything below is my own judgement plus what's verifiable on the open web, not a report of what r/computervision says. Treat rankings accordingly. If you want community signal, search Reddit yourself for `CS231n vs EECS 498 site:reddit.com` — it's a recurring thread.

### Primary — pick these

| Resource | Use it for | Cost |
|---|---|---|
| [Johnson — EECS 498-007 Deep Learning for CV](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) | **Your main course.** L7–8 (CNNs), L13 (attention), L15–16 (detection & segmentation). Skip 12, 17–22. | Free |
| [d2l.ai](https://d2l.ai/) | Reference, not linear reading. Every concept has runnable PyTorch. Look up CNNs, attention/transformers, computer vision chapters. | Free |
| [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Attention, properly, by building it | Free |
| [Umar Jamil](https://www.youtube.com/@umarjamilai) ([code on GitHub](https://github.com/hkproj)) | Transformers and vision-language models built from scratch in PyTorch. Closest thing to reading SAM 3's source with a guide. Start with the Transformer episode, then the vision-language one. | Free |
| [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 30-minute attention intuition before the harder material | Free |
| [Jurafsky & Martin — *Speech and Language Processing* (3rd ed, Jan 2026 draft)](https://web.stanford.edu/~jurafsky/slp3/) | PMI ([App. J](https://web.stanford.edu/~jurafsky/slp3/J.pdf)) and smoothing ([Ch 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf)), explained better than any CV text. PMI is an NLP tool; you are applying it to spatial adjacency, and this is where to learn it properly. | Free |
| [Szeliski — *Computer Vision*](https://szeliski.org/Book/) | Reference for superpixels and classical segmentation. Use the index; don't read linearly. | Free |

**Note on CS231n vs EECS 498:** same lineage, Johnson taught on both. EECS 498 is newer and self-contained on YouTube with clean audio; CS231n's freely available lecture videos are older. I'd take EECS 498. Lecture numbering shifts slightly between years — go by title, not number.

### Secondary — only if a primary isn't landing

- [MIT 6.S191](http://introtodeeplearning.com/) — fast, broad, refreshed yearly. Good if EECS 498 feels slow.
- [fast.ai](https://course.fast.ai/) — excellent, but top-down and training-focused. Somewhat mismatched to a training-free project.
- [First Principles of CV — Shree Nayar](https://fpcv.cs.columbia.edu/) — the segmentation lectures only. Best classical treatment, with companion PDFs.
- [Papers With Code — semantic segmentation](https://paperswithcode.com/task/semantic-segmentation) — for tracking SOTA, not learning.

### Actively avoid, given your deadline

- **Goodfellow, Bengio & Courville, *Deep Learning*** — excellent, wrong tool for 12 weeks. Heavily weighted to training theory you're skipping.
- **Any Udemy "Complete CV Bootcamp"** — OpenCV filters and YOLO fine-tuning. Not your problem space.
- **Full Coursera specialisations** — 4–5 months of pacing you don't have.

---

## Weekly allocation

Total study budget: **~60 hours across 12 weeks**, ~20% of your project time. The other 80% is building and experiments. If study creeps past 25%, you're procrastinating — go run an ablation.

| Weeks | Focus | Hours/wk |
|---|---|---|
| 1 | Tier 0. Close it with the tensor-explanation exercise. | 8 |
| 2–3 | Tier 1 first half: CNNs, attention, ViT. Plus write mIoU. | 8 |
| 4–5 | Tier 1 second half: DETR, MaskFormer, CLIP, OVSS lineage. Concurrent with baseline reproduction. | 8 |
| 6–7 | Tier 2 probability: PMI, smoothing, shrinkage. Directly feeds the M implementation. | 5 |
| 8–9 | Tier 2 graphs: RAG, superpixels, label propagation, CRF positioning. | 5 |
| 10–12 | Tier 3 + targeted reading for the writeup only. | 2 |

**The rule that matters:** learn a concept in the week you need it for code you're about to write. Anything you learn more than three weeks before you use it, you will have forgotten by the time you need it.

---

## Self-check

You're ready to leave each tier when you can answer these without looking:

**Tier 0** — Why is `semantic_seg` 288×288 when the image is 1024×1024? What does 5184 correspond to? Are those numbers probabilities?

**Tier 1** — What are the 200 things in `pred_logits: (1,200,1)`? Why does SAM 3 predict masks-plus-labels rather than a label map? How does a text prompt influence the image features at all?

**Tier 2** — Why PMI instead of raw co-occurrence counts? Why does a per-image M need smoothing? How is your method not just a CRF?

**Tier 3** — Why mIoU and not pixel accuracy on LoveDA rural? Why does GSD affect a co-occurrence matrix?

If you can answer the Tier 2 set clearly, you can defend this project.
