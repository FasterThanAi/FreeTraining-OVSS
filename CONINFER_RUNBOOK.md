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

Then, by hand, **after reading their README** — do not guess their pins:

```bash
conda activate coninfer
cd ~/ConInfer && less README.md
# install only inside `coninfer`
```

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
