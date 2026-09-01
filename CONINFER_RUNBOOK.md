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
