# ConInfer comparison — runbook

**Why.** ConInfer (arXiv:2603.29271) is the nearest published competitor. We cite it
and position against it, **without a number**. It is the first thing a reviewer will
ask about, and right now the question goes unanswered. One table row closes it.

**Timebox: one week.** If it will not run in a week, stop. A stated "we could not
run it, here is exactly what we tried" is an acceptable limitation; a wrecked
environment is not.

---

## ⛔ The one rule

**Nothing for ConInfer is installed into `segov3`. Ever.**

`segov3` is torch 2.4.1+cu121 · mmcv 2.2.0 (prebuilt wheel) · mmsegmentation 1.2.2
with `MMCV_MAX` patched to `'2.3.0'`. It is the only working combination found after
five failed attempts (`WEEK1_RESULTS.md` §2), and **every number in this project
rests on it** — including the 47.38 reproduction that makes the paper credible.

ConInfer is CLIP/DINOv3-based and will almost certainly want different pins. **That
is fine and expected.** Separate environments is the entire point. If their README
demands a torch version that conflicts, you do nothing to `segov3`.

`scripts/setup_coninfer.sh` enforces this: it **refuses to run** if `segov3` is the
active environment, and it snapshots `segov3`'s package list so a leak is caught by
a test rather than by a mIoU that quietly moved.

---

## Day 1 — environment and clone

```bash
conda deactivate                     # the script refuses to run from inside segov3
cd ~/FreeTraining-OVSS && git pull
bash scripts/setup_coninfer.sh
```

⛔ **Do not run their README's install lines verbatim.** They read:

```bash
conda create -n ConInfer python=3.11
pip install -r requirements.txt          # ← no `conda activate` between them
```

There is no `conda activate`, so `pip` installs into whatever environment is
**currently active**. With `segov3` active that command destroys the environment
every number in this project rests on. Their env name is also `ConInfer` where ours
is `coninfer`, and conda names are case-sensitive.

### Their pins do not build on this machine, and the reason is already documented

`requirements.txt` asks for `torch==2.7.1` with `mmcv>=2.1.0`. **No prebuilt mmcv
wheel exists for torch 2.7**, so pip falls back to a source build of
`mmcv-2.2.0.tar.gz`, which fails twice over:

1. `ModuleNotFoundError: No module named 'pkg_resources'` — mmcv's `setup.py`
   imports it and setuptools ≥ 70 removed it;
2. even patched, torch refuses to compile CUDA extensions because system nvcc is
   **13.3** against torch's 12.x.

Both are recorded in `WEEK1_RESULTS.md` §2 (lines 95–99) from week one. Their pin
works on *their* machine because their nvcc matches their torch.

⭐ **ConInfer needs `mmcv` and `mmsegmentation 1.2.2` — the same constraints as
SegEarth-OV3.** It is CLIP/DINOv3-based and will not need torch-2.7-specific APIs,
so install the combination this project already proved works:

```bash
conda activate coninfer
which pip                                # MUST show .../envs/coninfer/...
bash ~/FreeTraining-OVSS/scripts/install_coninfer_deps.sh
```

That installs torch 2.4.1+cu121, mmcv 2.2.0 from the **prebuilt** index, mmseg
1.2.2 with `MMCV_MAX` patched, then everything else from their `requirements.txt`
unchanged. It refuses to run outside `coninfer` and checks `pip`'s path first.

⚠️ **This is a deviation from their published pin and must be reported.** If their
numbers do not reproduce, the torch version is the first suspect — which is exactly
why the next step is to reproduce *their* number before touching our splits.

## What a clean install actually required — for the reproducibility note

Three things stop `python eval.py` from running on a fresh machine. None is
exotic, and all three are worth one factual sentence in the paper, because they
are what a reader would hit too:

| problem | detail |
|---|---|
| `segearth_segmentor.py` **absent from their repo** | `eval.py` imports it at line 4. Copied from upstream `likyoo/SegEarth-OV`. |
| `ftfy` undeclared | mmsegmentation 1.2.2 does `from .tokenizer import tokenize` in `mmseg/utils/__init__.py`, and that CLIP tokenizer imports `ftfy` and `regex`. mmseg does not list them; neither does ConInfer. |
| `psutil` undeclared | `fast-pytorch-kmeans` imports it without declaring it. |

Plus two that are properties of *this* machine rather than their repo:

- **`torch==2.7.1` is unusable here.** No prebuilt mmcv wheel exists for torch 2.7,
  and the source build fails on `pkg_resources` and then on nvcc 13.3 against
  torch's 12.x (`WEEK1_RESULTS.md` §2). We run their code on torch 2.4.1+cu121
  with mmcv 2.2.0 from the prebuilt index. ⚠️ **This deviation must be reported**,
  and is the reason their own published number is reproduced first.
- **`pydensecrf==1.0rc3` does not build against Cython 3** and is excluded. Nothing
  in their tree imports it.

State it neutrally: *"reproducing the comparison required one file from the
upstream repository, two undeclared dependencies, and a torch version change; we
verified the environment by reproducing their reported baseline before measuring."*
That is a reproducibility note, not a complaint, and it is evidence the run is real.

## Day 2–3 — reproduce *their* number first

⚠️ **Do not evaluate on our splits until their own reported number reproduces.**
This is the same discipline that made our baseline credible: we reproduced 47.38
against a published 47.4 *before* measuring anything. A ConInfer row that is 3 points
low because of a preparation bug is worse than no row, because it looks like a result.

Run whatever dataset they report that we also have. Record:

- the number they report, the number you get, and the gap
- if the gap is > 1 mIoU, **stop and debug the preparation**, not the method

## Day 4 — evaluate on our splits

```bash
# LoveDA val, the same 1669 tiles
# OpenEarthMap val, the same 384 tiles
```

Both must be the **identical tile lists** we used, or the comparison is not a
comparison. For OpenEarthMap ours is 384 of the official 500; hand ConInfer the same
subset.

## Day 5 — the row, in both metrics

Report ConInfer the way we report everything else:

| method | full mIoU | catch-all-excluded mIoU |
|---|---|---|
| SegEarth-OV3 (baseline) | 47.37 | 47.68 |
| ConInfer | ? | ? |
| **+ per-class $\tau$ (ours)** | **48.53** | **49.04** |

⭐ **Both columns.** If ConInfer's gain is concentrated in the catch-all class the way
recovery's was on OpenEarthMap (§8.1), that is itself a finding, and it is exactly
what our reporting recommendation predicts should be checked. **Do not assume it;
measure it and report whichever way it comes out.**

⚠️ Our method and ConInfer are **not mutually exclusive** — ours re-fits a threshold,
theirs modifies the scores. If both run, "ConInfer + per-class τ" is a cheap and
interesting third row.

## Every day — verify nothing leaked

```bash
bash scripts/setup_coninfer.sh --verify
```

A matching package list is **necessary, not sufficient**. Before trusting any further
measurement from this project, re-run the behavioural gate:

```bash
cd ~/SegEarth-OV-3 && python eval.py ./configs/cfg_loveda.py    # must be 47.38
```

---

## If it does not run

Write it up honestly and move on to Phase 7.2. State: the version attempted, the
commit hash, the specific failure, and how long was spent. That is a real limitation
paragraph and reviewers accept it. What they do not accept is a competitor cited with
no number and no explanation.


---

# ⏸ STATUS — 1 Sep 2026: blocked on gated DINOv3 weights

Everything except one file is working. Resume from here.

| | |
|---|---|
| `coninfer` env | ✅ torch 2.4.1+cu121 · mmcv 2.2.0 · mmseg 1.2.2 · ops OK |
| undeclared deps | ✅ `ftfy`, `regex`, `psutil`, `openpyxl` |
| `segearth_segmentor.py` | ✅ copied from upstream `likyoo/SegEarth-OV` |
| LoveDA symlink | ✅ `~/ConInfer/data/LoveDA` → 1669 tiles |
| single-GPU configs | ✅ `cfg_loveda_1gpu.py` in both config dirs (batch 8, workers 4) |
| mmengine builds the runner | ✅ config parses, model construction begins |
| `segov3` | ✅ byte-identical, and behaviourally re-gated at **47.37** |
| **DINOv3 SAT-493M weights** | ⛔ **HTTP 403 — licence-gated** |

**The blocker, precisely.** `ConInfer_segmentor.py:180` calls
`torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=WEIGHT_DIR)`.
The vendored `dinov3/hub/utils.py` sets
`DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"`, and
`dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` there returns **403**. Hugging Face
carries only the `transformers` conversion (`config.json` + `model.safetensors`),
which `torch.hub.load` cannot consume.

⛔ **Do NOT convert the safetensors.** The HF conversion renames every parameter,
so a hand-written remap can load under `strict=False` with half the network still
randomly initialised — producing a plausible number from a broken model, with no
way to detect it. That is the precise failure "reproduce their published figure
first" exists to prevent.

⚠️ **DINOv3 is required for their BASELINE config too.** `base_config1.py` also
builds `ConInferSegmentation`, and the `if/else` at line ~171 selects *which*
backbone, never whether to load one. There is no lighter path.

## To resume

1. Accept the DINOv3 licence (`facebookresearch/dinov3`, pretrained-models
   section) and obtain the signed URL.
2. `curl -fL -o ~/weights/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth "<url>"`
3. `bash scripts/patch_coninfer_paths.sh ~/weights/dinov3/<that file>`
4. `cd ~/ConInfer && python eval.py --config configs_baseline/cfg_loveda_1gpu.py --work-dir ./out_base_loveda`
5. Check it against **SegEarth-OV's** published LoveDA mIoU — ⚠️ *not* 47.4, which
   is SegEarth-OV**3** on SAM 3 at 1024². This baseline is CLIP at 448².

## If the licence never comes through

The limitation paragraph writes itself, and it is specific enough to be credible:

> *We attempted a direct comparison with ConInfer. Their released code omits
> `segearth_segmentor.py`, requires three dependencies declared by neither their
> requirements nor the packages importing them, and hardcodes absolute paths from
> the authors' filesystem; all were resolved. It further requires a
> satellite-pretrained DINOv3 ViT-L/16 checkpoint whose distribution is
> licence-gated (HTTP 403), and which is required by their baseline configuration
> as well as their method. We therefore report the comparison qualitatively.*

⭐ **And one finding stands regardless of whether the run happens:** ConInfer
requires a second large pretrained backbone with gated weights; our method
requires ~200 labelled tiles and no additional weights. That is a real difference
in deployment cost, discovered only by attempting the reproduction, and it belongs
in the related-work positioning either way.


---

# §7.1a — does per-class τ transfer to a CLIP backbone?

**The question the comparison row cannot answer:** *is per-class thresholding a
SAM 3 quirk?* ConInfer thresholds per-class scores with a single global `prob_thd`
— precisely the design our method replaces — so it is the natural test.

⭐ **The reproduction gap does not block this.** We measure a **delta on our own
run**, and a delta is robust to a constant offset. If per-class τ adds +X to their
scores as we measured them, that is a valid statement about transfer whether their
absolute is 36.99 or 39.33.

## Step 1 — cache their scores (LoveDA only; OpenEarthMap is dropped)

```bash
cd ~/ConInfer
python ~/FreeTraining-OVSS/scripts/coninfer_cache.py \
  --config configs_ConInfer/cfg_loveda_1gpu.py \
  --out ~/outputs/coninfer_loveda/cache
```

⭐ **No edits to their source.** mmseg leaves `seg_logits`, `pred_sem_seg` and
`gt_sem_seg` on every `SegDataSample` at original resolution, so the script wraps
`predict()`, calls it unmodified, and reads the result. Observation-only is
structural here, not something to verify by diffing.

⚠️ **Two gates before the cache is used:**

1. **mIoU must print 36.99**, matching the un-instrumented run. A read-only wrapper
   cannot change it; if it moved, something is wrong.
2. **`conf` must lie in [0, 1]** — the script reports the observed range. Our
   threshold grid is over [0, 1], so a raw-logit score would be binned wrongly and
   every fitted τ would be meaningless. If it warns, stop and check whether their
   `postprocess_result` applies a sigmoid before `prob_thd`.

## Step 2 — fit per-class τ with the existing scripts, unchanged

```bash
cd ~/FreeTraining-OVSS
python scripts/tau_oracle.py    --cache ~/outputs/coninfer_loveda/cache --tau 0.8 \
  --md ~/outputs/week4/coninfer_tau_oracle.md
python scripts/tau_cv.py        --cache ~/outputs/coninfer_loveda/cache --tau 0.8 \
  --objective real --md ~/outputs/week4/coninfer_tau_cv.md
python scripts/metric_report.py --cache ~/outputs/coninfer_loveda/cache --tau 0.8 \
  --md ~/outputs/week4/coninfer_metric.md
```

⚠️ **`--tau 0.8`, not 0.5.** That is ConInfer's own published LoveDA threshold. The
baseline every gain is measured against must be *their* operating point, or the
comparison is against a strawman.

## What each outcome means

| outcome | what the paper says |
|---|---|
| **clearly positive** | ⭐ per-class τ is a property of **thresholded per-class scores**, not of a backbone. Demonstrated on two architectures (SAM 3, CLIP ViT-B/16) and two methods. The strongest form of the claim. |
| **near zero** | the effect needs SAM 3's particular per-class calibration asymmetry. **Report it** — it sharpens the mechanism and pre-empts the reviewer who assumes generality we did not test. |
| **negative** | ConInfer's GMM already equalises per-class calibration, so there is nothing left to correct. That is an *interesting* negative and a compliment to their method. |

**Every outcome is publishable**, which is why this is worth a day and a third
dataset is not, until it is done.
