# Application section — plan

**Goal.** Demonstrate both levers on Indian imagery, in a way that (a) showcases the
project's strongest result, (b) is honest about the calibration cost the paper itself
argues is irreducible, and (c) fits in about a week.

⭐ **The demonstration is `tree` and `water`.** Not because they are convenient, but
because they are where the method's two levers do their work:

| class | what the baseline does | what we recover |
|---|---|---|
| `tree` (Potsdam) | recall **38.69%** — finds a third of the trees | recall **66.83%**, IoU **+21.17** |
| `water` (LoveDA) | precision 89.5 / recall 54.7 | **+6.78** from τ, **+4.41** more from the scale |

**A picture of an Indian city where the baseline misses 60% of the trees and our
method finds them is worth more than any table in the paper.**

---

## ⭐ Step 0 — before anything else, check whether labels already exist

OpenEarthMap spans dozens of countries. **If it contains Indian cities, this section
costs almost nothing** — real Indian imagery with real labels, at 0.25–0.5 m, already
in the pipeline's format.

```bash
ls ~/data/openearthmap_mm/images/* | sed 's#.*/##;s/_[0-9]*\.tif$//' | sort -u | head -100
```

Look for Indian city names. Also check the OpenEarthMap paper's region list — the
release covers train (3500) and val (500), and we hold only the val subset, so an
Indian city may exist in train and simply not be downloaded yet.

| outcome | what to do |
|---|---|
| **India present, ≥ 40 tiles** | ⭐ use it. No labelling. Quantitative section. |
| India present, < 40 tiles | use it qualitatively, and hand-label to top up |
| absent | hand-label, per Step 2 |

⛔ **Do this first.** Everything below changes depending on the answer, and it is one
command.

---

## Step 1 — imagery, if labels must be made

⚠️ **Match the resolution.** LoveDA is 0.3 m, Potsdam 0.05 m. Indian imagery must be
in that range. **Sentinel-2 at 10 m is the wrong instrument** — a 30× GSD gap would
confound the entire demonstration, and co-occurrence and threshold behaviour are both
GSD-sensitive.

| source | licence | note |
|---|---|---|
| ⭐ **OpenAerialMap** (openaerialmap.org) | CC-BY, publishable | drone/aerial, sub-metre, patchy India coverage — check first |
| **Bhuvan / Bhoonidhi** (ISRO) | free, registration | ⚠️ read the terms for **redistribution in a publication** before relying on it |
| ⛔ Google Earth / Maps screenshots | **not publishable** | licence forbids it; do not build a paper figure on this |

Target **~100 tiles at 512² or 1024²** from 2–3 cities, so the set is not one
neighbourhood.

---

## Step 2 — labelling, and the honest cost

⚠️ **Dense per-pixel labels are needed.** The fit maximises IoU from a confusion
matrix, and that needs ground truth per pixel. This is the expensive part and the plan
should not pretend otherwise.

| option | cost | what it supports |
|---|---|---|
| **20 tiles, dense** | ~3–4 days in QGIS | 10 calibration / 10 evaluation. A number, with a wide error bar stated. |
| **10 tiles, dense** | ~1.5–2 days | qualitative figure + an indicative number, clearly labelled indicative |
| ⛔ sparse points | ~hours | **not sufficient** — too few pixels to estimate a 6-class confusion matrix |

⭐ **Restrict the vocabulary to 4 classes**: `tree`, `water`, `building`, `road`, plus
a catch-all. Fewer classes means faster labelling, a smaller confusion matrix, and it
still contains both classes the method is strongest on. **Say in the paper that the
vocabulary was reduced for labelling cost**, since catch-all share is a variable the
paper itself shows matters.

---

## Step 3 — pre-register the prediction

⛔ **Do this before running anything**, exactly as for OpenEarthMap and Potsdam. Write
`prereg/predict_india.md`, commit it, and let `git log` timestamp it. Predict:

1. the baseline's `tree` recall (from the Potsdam and LoveDA figures)
2. whether Potsdam's fitted parameters transfer (§9e says **no** — predict a drop)
3. the sign and rough size of the gain after local calibration
4. which of the two levers pays more, and why

⭐ **Prediction 2 is the interesting one.** The paper claims parameters do not transfer
across domains. India is a genuine domain shift from Germany and China. **If the
transfer arm fails as predicted, that is the paper's own scope limit confirmed on a
fourth continent** — which is worth more than the gain itself.

---

## Step 4 — the runs

```
A  baseline, published τ                      the reference
B  + Potsdam-fitted parameters (transfer)     predicted to be WORSE than A
C  + locally fitted τ                         lever 1
D  + locally fitted τ and scale               lever 2
```

All four on the same held-out Indian tiles, with calibration and evaluation disjoint.
`reorder_deploy.py` produces C and D directly; B is a config with the Potsdam vectors
pasted in.

---

## Step 5 — what goes in the paper

**One figure, three columns:** image · baseline · ours, on 3–4 Indian tiles, with
recovered `tree` and `water` pixels highlighted. **Reviewers look at figures first**,
and the tree recall difference is visible without reading a caption.

**One small table:** the four rungs above, per class.

**~400 words**, saying: the method deploys to a new region at the cost of a few dozen
labelled tiles; the parameters from another continent do not transfer, as our own
scope limit predicts; and the gain concentrates in the same classes for the same
reason.

⚠️ **State the limits in the same paragraph**: small evaluation set, reduced
vocabulary, one country. An honest small demonstration is worth more than an
overclaimed one, and this paper's credibility is built on that.

---

# The demo tool — what it is, and why it is worth two days

**A small program someone can click**, rather than a table they have to read.

You select a satellite image, type class names into a box — `tree, water, building,
road` — and see three panels side by side: **the baseline, ours, and the pixels we
recovered**, highlighted.

### Why it is worth building

| | |
|---|---|
| ⭐ **It makes the contribution visible in ten seconds** | your supervisor and examiners see the trees the baseline missed, without reading a number |
| ⭐ **It demonstrates the open-vocabulary property** | type `solar panel` and it segments solar panels — no retraining. That is the part of "training-free" people do not believe until they see it |
| **It is viva-proof** | a live demonstration answers "does this actually work?" better than any table |
| **No research risk** | it exhibits results you already have |

### Two forms

**(a) Live app — Gradio, on the workstation.** ~50 lines around the existing
segmentor: upload an image, enter a vocabulary, get the three panels. Runs on the GPU,
so it is genuinely interactive and accepts any image and any class names.
⚠️ Only runs where the model and GPU are — fine for a demonstration, not shareable.

**(b) Static page — pre-computed.** Run 8–12 interesting tiles offline, save the
panels, and build one HTML page with a slider or tabs. **Shareable, works on a phone,
survives the GPU being busy**, and can be sent to an examiner in advance.
⚠️ Fixed examples only; nobody can type their own class name.

⭐ **Build (a) first** — it is the smaller job and it is what a live viva needs. Add
(b) later from its outputs if a shareable version is wanted.

### Cost

| | |
|---|---|
| Gradio app | 1 day |
| Static page from its outputs | half a day |

⛔ **It is a deliverable, not a contribution.** It goes in the project report and the
viva, not in the paper's contribution list. Do not let it consume time that belongs to
the application experiments above.
