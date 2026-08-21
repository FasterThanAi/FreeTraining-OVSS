# Patch guide — `measure_discard_rate.py`: per-class `S_pres` + `.npz` cache

⚠️ **I have not seen your `measure_discard_rate.py`.** The fragments below are written against
the tensor names confirmed by `sam3_smoke_test.py --raw` on 2026-08-20 and against the CSV schema
your script already emits (`image, valid_px, real_px, discarded_px, discard_pct_of_real`). Adapt
variable names to match your actual loop rather than pasting blindly.

Both changes go in **one edit, one re-run at τ=0.5** (~25 min).

---

## 0. Before you start

```bash
cd ~/FreeTraining-OVSS
git add -A && git commit -m "chore: checkpoint before instrumentation patch"
cp scripts/measure_discard_rate.py scripts/measure_discard_rate.py.bak
```

Find where the model outputs are produced:

```bash
grep -n 'forward_grounding\|presence\|semantic_seg\|pred_masks\|argmax\|tau' \
  scripts/measure_discard_rate.py
```

---

## 1. Capture per-class `S_pres`

Confirmed tensor (from the raw dump): `presence_logit_dec`, shape `(1,1)`, dtype `bfloat16`.
It is a **logit** — apply sigmoid.

Inside the per-class loop, wherever the forward pass happens:

```python
# --- per class, inside the existing forward pass ---
s_pres = out["presence_logit_dec"].sigmoid().float().item()
pres_vec[ci] = s_pres          # ci = class index, pres_vec = np.zeros(n_classes, np.float32)
```

Allocate `pres_vec` once per image, before the class loop:

```python
pres_vec = np.zeros(len(CLASSES), dtype=np.float32)
```

If your script calls a wrapper that discards `presence_logit_dec` (the way `Sam3Processor` does),
you need the raw path — `model.forward_grounding(...)` — as used in `sam3_smoke_test.py`.

### Emit it

Add one row per image to a new CSV:

```python
pres_rows.append({
    "image": stem,
    **{f"spres_{c}": float(pres_vec[i]) for i, c in enumerate(CLASSES)},
    "spres_max": float(pres_vec.max()),
    "spres_mean": float(pres_vec.mean()),
})
```

and at the end, alongside your existing writes:

```python
pd.DataFrame(pres_rows).to_csv(out_dir / "per_image_presence.csv", index=False)
```

`spres_max` is the field that matters most — a tile where **no** class clears ~0.2 is the
presence-collapse signature from `WEEK1_RESULTS.md` §8.1.

---

## 2. Add the `.npz` cache

The point: τ is a post-hoc comparison on `P_final`. Caching the per-pixel max-confidence and
argmax means every future threshold, confusion matrix, and ablation becomes a numpy pass instead
of a 25-minute encoder run.

After `P_final` is assembled for all classes and **before** thresholding:

```python
cache_dir = out_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    cache_dir / f"{stem}.npz",
    conf=P_final.max(axis=0).astype(np.float16),    # H×W  best score at each pixel
    pred=P_final.argmax(axis=0).astype(np.uint8),   # H×W  which class won
    gt=gt.astype(np.uint8),                         # H×W  ground truth
    spres=pres_vec,                                 # (n_classes,) float32
)
```

**Size:** ~1.5 MB/image compressed → **~2.5 GB** for the 1669-tile val split. Check headroom
first (`df -h ~`); if home is tight, put `cache_dir` on scratch and symlink.

**Important:** `conf` as float16 has ~3 decimal digits of precision near 0.1 — fine for τ
comparisons at 0.05 granularity, marginal below that. If you plan a fine sweep near zero, use
float32 and accept ~5 GB.

**Caveat on `pred`:** `argmax` discards the full per-class distribution, so the cache supports
threshold sweeps and confusion matrices but **not** anything needing runner-up scores (e.g.
margin-based analysis, or your Week 8 scoring function). If you want those later, also cache the
top-2 class indices and scores — cheap, and saves a third full re-run.

---

## 3. Re-run

```bash
cd ~/SegEarth-OV-3
nohup python ~/FreeTraining-OVSS/scripts/measure_discard_rate.py \
  --tau 0.5 --out ~/outputs/week2_tau0.5_instrumented \
  > ~/logs/week2_tau0.5_instrumented.log 2>&1 &
```

Note the trailing `&`. Write to a **new** output dir so the verified τ=0.5 results stay intact.

**Validation gate:** the new run must reproduce **mIoU 47.37** and **29.68%** discard. If either
moves, the patch changed behaviour and must be fixed before its numbers are used.

---

## 4. Then — the analysis the patch exists for

```python
import pandas as pd, numpy as np

p = pd.read_csv('~/outputs/week2_tau0.5_instrumented/per_image_presence.csv')
d = pd.read_csv('~/outputs/week2_tau0.1/per_image_discard.csv')
m = p.merge(d[['image', 'discard_pct_of_real']], on='image')

catastrophic = m[m.discard_pct_of_real >= 99.99]
healthy      = m[m.discard_pct_of_real < 1]

print(f"catastrophic n={len(catastrophic)}  spres_max mean={catastrophic.spres_max.mean():.4f}")
print(f"healthy      n={len(healthy)}  spres_max mean={healthy.spres_max.mean():.4f}")
print(catastrophic.filter(like='spres_').describe())
print(healthy.filter(like='spres_').describe())
```

**Read it as follows.** If the catastrophic set's `spres_max` clusters low (≲0.2) while the
healthy set's sits high, presence collapse is confirmed as a systematic failure mode and §8.1
generalises — that is a figure, a paragraph, and a claim SegEarth-OV3's paper does not make. If
the two distributions overlap, tile 3487 was an anecdote and must be dropped from the writeup.

Either outcome is worth the 25 minutes. The second one is worth more than it looks: finding out
now costs one run, finding out in Week 11 costs a rewritten results section.

A scatter of `spres_max` against `discard_pct_of_real` across all 1669 tiles is the figure to
draw first — it shows the relationship and the bimodality in one panel.
