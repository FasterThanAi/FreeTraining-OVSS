"""
Week 2 diagnostic — how many labelled pixels does SegEarth-OV3 throw into "background"?

Runs SegEarth-OV3 over a LoveDA split and accumulates:
  * a 7x7 confusion matrix (true class x predicted class)
  * per-class counts of pixels that carry a real GT label but are predicted background
  * per-image discard rates

Outputs to  outputs/week2/  :
  confusion_matrix.npy / .csv     raw counts
  discard_per_class.csv           the headline table
  per_image_discard.csv           distribution across images
  discard_summary.md              paste-ready summary
  confusion_matrix.png            figure
  discard_by_class.png            figure
  per_image_presence.csv          per-class S_pres per image        [instrumented]
  cache/<image>.npz               (conf, pred, conf2, pred2, gt, spres)  [instrumented]

MUST be run from inside the SegEarth-OV-3 clone (it imports their custom classes):

    conda activate segov3
    cd ~/SegEarth-OV-3
    python ~/FreeTraining-OVSS/scripts/measure_discard_rate.py --limit 20
    python ~/FreeTraining-OVSS/scripts/measure_discard_rate.py

Instrumentation (added 21 Aug) — see INSTRUMENTATION_PATCH.md
------------------------------------------------------------
1. per-class S_pres.  P_final = P_fused * S_pres, so the presence score is a
   hard ceiling on every pixel of a tile for that class. It is a local variable
   inside the segmentor, so capturing it needs the observe-only patch in
   reference/segearthov3_segmentor.py (marked "<<< INSTRUMENTATION").
   Copy that file over ~/SegEarth-OV-3/segearthov3_segmentor.py first, or this
   script falls back to writing NaNs and warns once.

   NOTE queries != classes. cls_loveda.txt declares 7 classes but 11 queries
   (building,house / barren,bareland,soil / forest,tree). We collapse queries
   to classes with max-over-synonyms, mirroring the .max(1) the segmentor does
   in predict(). And under sliding-window inference there is one S_pres per
   *crop*, not per image -- the CSV reports max and mean across crops, and the
   .npz keeps the full (n_views, n_cls) array.

2. .npz cache.  tau is only ever compared against the max of seg_logits, so
   caching (conf, pred) makes every future threshold, confusion matrix and
   ablation a sub-minute numpy pass instead of a 25-minute encoder run.
   conf2/pred2 (runner-up) are included so margin-based analysis and the Week 8
   scoring function do not force a third full re-run.

VALIDATION GATE: an instrumented run at tau=0.5 must still report
mIoU 47.37 and 29.68% discard. Neither is computed from anything added here --
the confusion-matrix path is untouched -- so a move means something broke.

Label convention (LoveDA):
    GT files hold 0..7 where 0 = no-data (ignored), 1 = background, 2..7 = real classes.
    mmseg is configured with reduce_zero_label=True, so predictions come back 0-indexed.
    We add 1 to predictions to put both on the same 1..7 scale. Getting this wrong
    silently invalidates every number, so --limit runs print an mIoU you can check
    against the known baseline of 47.38.
"""
import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# An uninstrumented segmentor, or a class absent from a crop, yields all-NaN
# slices. That is expected and handled; without this the log gets 1669 copies.
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Mean of empty slice')

# --- register the authors' custom segmentor/dataset with mmseg's registry ---
sys.path.insert(0, os.getcwd())
try:
    import segearthov3_segmentor  # noqa: F401
    import custom_datasets        # noqa: F401
    import custom_transforms      # noqa: F401
except ImportError as e:
    sys.exit(f"ERROR: run this from inside the SegEarth-OV-3 clone.\n  {e}")

from mmseg.apis import init_model, inference_model  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels  # noqa: E402

# Resolved from the model/config at runtime -- see labels.py. Hardcoding LoveDA
# here meant that pointing this at OpenEarthMap would not crash, it would compute
# nonsense against valid-looking array indices.
CLASSES = None
N = None
BACKGROUND = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/cfg_loveda.py')
    ap.add_argument('--img-dir', default='data/LoveDA/img_dir/val')
    ap.add_argument('--ann-dir', default='data/LoveDA/ann_dir/val')
    ap.add_argument('--out', default=os.path.expanduser('~/outputs/week2'))
    ap.add_argument('--limit', type=int, default=0, help='0 = all images')
    ap.add_argument('--tau', type=float, default=None,
                    help='override prob_thd, e.g. 0.3 or 0.1 for the sweep')
    ap.add_argument('--no-presence', action='store_true',
                    help='disable presence gating (P_final = P_fused, no S_pres multiply). '
                         'The counterfactual for WEEK1_RESULTS 9.2: if the catastrophic '
                         'tiles recover with this off, presence gating CAUSED their collapse; '
                         'if they stay bad, low S_pres was a symptom of genuinely hard tiles. '
                         'Expect overall mIoU to DROP -- gating helps on average.')
    ap.add_argument('--no-cache', action='store_true',
                    help='skip the per-image .npz cache (~2.5-4 GB for 1669 tiles)')
    ap.add_argument('--cache-dir', default=None,
                    help='where .npz files go (default: <out>/cache)')
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in Path(args.img_dir).glob('*.png'))
    if args.limit:
        names = names[:args.limit]
    print(f'{len(names)} images | config {args.config} | out {out}')

    if args.tau is not None:
        print(f'  overriding prob_thd -> {args.tau}')

    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    model = init_model(args.config, device='cuda')
    if args.tau is not None:
        model.prob_thd = args.tau   # segmentor reads this attribute at inference
    if args.no_presence:
        model.use_presence_score = False   # same trick: read at inference, not init
        print('  PRESENCE GATING DISABLED -- P_final = P_fused. '
              'Counterfactual run; S_pres will be NaN because it is never applied.')

    # ---- instrumentation setup -------------------------------------------
    cache_dir = None
    if not args.no_cache:
        cache_dir = Path(args.cache_dir) if args.cache_dir else out / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(cache_dir).free / 2**30
        need_gb = 0.0025 * len(names)          # ~2.5 MB/image compressed
        print(f'  cache -> {cache_dir}  (need ~{need_gb:.1f} GB, free {free_gb:.1f} GB)')
        if free_gb < need_gb * 1.2:
            sys.exit(f'ERROR: not enough disk for the cache. Use --no-cache, or '
                     f'--cache-dir on a bigger volume.')

    global CLASSES, N, BACKGROUND
    _lab = labels.from_model(model, cfg.get('model', cfg))
    CLASSES, N, BACKGROUND = _lab.names, _lab.n, _lab.bg
    print(f'  classes: {N} -- {", ".join(CLASSES)}')
    print(f'  background is mask value {BACKGROUND} ({CLASSES[BACKGROUND - 1]})')

    # query -> class map, collapsing synonym queries onto classes
    qidx = model.query_idx.cpu().numpy() if hasattr(model, 'query_idx') else None
    has_presence = hasattr(model, 'last_presence')
    if not has_presence:
        print('  !! segmentor is NOT instrumented -- S_pres will be NaN.\n'
              '     Copy reference/segearthov3_segmentor.py over\n'
              '     ~/SegEarth-OV-3/segearthov3_segmentor.py and re-run.')

    conf = np.zeros((N, N), dtype=np.int64)      # conf[true, pred], 0-indexed
    per_image = []
    pres_rows = []

    for i, name in enumerate(names, 1):
        img_p = f'{args.img_dir}/{name}.png'
        gt = np.array(Image.open(f'{args.ann_dir}/{name}.png'))
        result = inference_model(model, img_p)
        pred = result.pred_sem_seg.data.cpu().numpy().squeeze()
        pred = pred.astype(np.int64) + 1          # 0-indexed -> 1..7

        valid = gt > 0                            # drop no-data
        g, p = gt[valid], pred[valid]
        np.add.at(conf, (g - 1, p - 1), 1)

        real = g > BACKGROUND                     # carries a real land-cover label
        discarded = real & (p == BACKGROUND)
        per_image.append({
            'image': name,
            'valid_px': int(valid.sum()),
            'real_px': int(real.sum()),
            'discarded_px': int(discarded.sum()),
            'discard_pct_of_real': round(100 * discarded.sum() / max(real.sum(), 1), 2),
        })

        # ---- instrumentation: S_pres, collapsed queries -> classes --------
        # pres_views: (n_views, N). n_views = 1 for whole-image inference, or
        # one row per crop under sliding window.
        pres_views = np.full((1, N), np.nan, dtype=np.float32)
        if has_presence and qidx is not None:
            raw = model.last_presence                      # (n_views, num_queries)
            if raw.size:
                pres_views = np.full((raw.shape[0], N), np.nan, dtype=np.float32)
                with np.errstate(all='ignore'):
                    for c in range(N):                     # max over synonyms
                        cols = raw[:, qidx == c]
                        if cols.size:
                            pres_views[:, c] = np.nanmax(cols, axis=1)

        with np.errstate(all='ignore'):
            pres_max = np.nanmax(pres_views, axis=0)       # across views, per class
            pres_mean = np.nanmean(pres_views, axis=0)
        row = {'image': name, 'n_views': int(pres_views.shape[0])}
        row.update({f'spres_{c}': float(pres_max[k]) for k, c in enumerate(CLASSES)})
        row.update({f'spresmean_{c}': float(pres_mean[k]) for k, c in enumerate(CLASSES)})
        # over REAL classes only -- background presence is not informative here
        row['spres_max'] = float(np.nanmax(pres_max[1:])) if N > 1 else float('nan')
        row['spres_mean'] = float(np.nanmean(pres_max[1:])) if N > 1 else float('nan')
        pres_rows.append(row)

        # ---- instrumentation: .npz cache ----------------------------------
        # tau is only ever compared against seg_logits.max(0), so caching the
        # top-2 makes any future threshold / ablation a numpy pass.
        if cache_dir is not None:
            lg = result.seg_logits.data.float()             # (N, H, W)
            # <<< INSTRUMENTATION: P_fused = max(P_sem, P_inst_agg), captured
            # BEFORE the presence multiply. `conf` is P_final = P_fused * S_pres,
            # so the two factors are entangled in it and cannot be recovered
            # afterwards. Recorded as its own top-1 so the recoverability AUC can
            # be re-run on the ungated score. Absent on an unpatched segmentor,
            # in which case the keys are simply omitted and readers fall back.
            fused_arrays = {}
            fl = getattr(model, 'last_fused', None)
            if fl is not None:
                try:
                    fl = fl.float()
                    if fl.shape[-2:] != lg.shape[-2:]:
                        fl = torch.nn.functional.interpolate(
                            fl[None], size=lg.shape[-2:], mode='bilinear',
                            align_corners=False)[0]
                    ftop = torch.topk(fl, k=min(2, fl.shape[0]), dim=0)
                    fv = ftop.values.cpu().numpy()
                    fi = ftop.indices.cpu().numpy().astype(np.uint8)
                    fused_arrays = dict(fconf=fv[0].astype(np.float16), fpred=fi[0])
                except Exception as e:
                    print(f'    (P_fused capture failed on {name}: {e})')

            k = min(2, lg.shape[0])
            top = torch.topk(lg, k=k, dim=0)
            vals = top.values.cpu().numpy()
            idxs = top.indices.cpu().numpy().astype(np.uint8)
            np.savez_compressed(
                cache_dir / f'{name}.npz',
                conf=vals[0].astype(np.float16),            # best score  (== max_vals)
                pred=idxs[0],                               # argmax, 0-indexed, PRE-threshold
                conf2=(vals[1] if k > 1 else vals[0]).astype(np.float16),
                pred2=(idxs[1] if k > 1 else idxs[0]),
                gt=gt.astype(np.uint8),
                spres=pres_views,                           # (n_views, N) full fidelity
                classes=np.array(CLASSES),
                **fused_arrays,                             # <<< P_fused, pre-gating
            )

        if i % 100 == 0 or i == len(names):
            print(f'  {i}/{len(names)}')

    # ---- derived numbers -------------------------------------------------
    total_valid = conf.sum()
    total_real = conf[1:, :].sum()
    total_discarded = conf[1:, BACKGROUND - 1].sum()

    rows = []
    for c in range(1, N):                          # skip background as a true class
        tot = conf[c, :].sum()
        lost = conf[c, BACKGROUND - 1]
        rows.append({
            'class': CLASSES[c],
            'gt_pixels': int(tot),
            'lost_to_background': int(lost),
            'pct_lost': round(100 * lost / max(tot, 1), 2),
        })
    rows.sort(key=lambda r: -r['pct_lost'])

    # mIoU from the confusion matrix — sanity check against the known 47.38
    inter = np.diag(conf).astype(float)
    union = conf.sum(1) + conf.sum(0) - np.diag(conf)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    miou = float(np.nanmean(iou) * 100)

    # ---- write ------------------------------------------------------------
    np.save(out / 'confusion_matrix.npy', conf)
    with open(out / 'confusion_matrix.csv', 'w') as f:
        f.write('true\\pred,' + ','.join(CLASSES) + '\n')
        for c in range(N):
            f.write(CLASSES[c] + ',' + ','.join(map(str, conf[c])) + '\n')
    with open(out / 'discard_per_class.csv', 'w') as f:
        f.write('class,gt_pixels,lost_to_background,pct_lost\n')
        for r in rows:
            f.write(f"{r['class']},{r['gt_pixels']},{r['lost_to_background']},{r['pct_lost']}\n")
    with open(out / 'per_image_discard.csv', 'w') as f:
        f.write('image,valid_px,real_px,discarded_px,discard_pct_of_real\n')
        for r in per_image:
            f.write(f"{r['image']},{r['valid_px']},{r['real_px']},"
                    f"{r['discarded_px']},{r['discard_pct_of_real']}\n")

    # instrumentation: per-image presence table
    if pres_rows:
        cols = list(pres_rows[0].keys())
        with open(out / 'per_image_presence.csv', 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in pres_rows:
                f.write(','.join(
                    r[c] if isinstance(r[c], str) else f'{r[c]}' for c in cols) + '\n')

    pcts = np.array([r['discard_pct_of_real'] for r in per_image])
    tau_str = args.tau if args.tau is not None else 'config default (0.5)'
    summary = [
        '# Week 2 — Discard-Rate Diagnostic\n',
        f'- Images: **{len(names)}**  |  τ: **{tau_str}**',
        f'- Presence gating: **{"DISABLED (counterfactual)" if args.no_presence else "on (baseline)"}**',
        f'- mIoU recomputed from confusion matrix: **{miou:.2f}** '
        (f'(baseline reference: 47.38 — if these disagree, the label alignment is '
         f'wrong)\n' if 'loveda' in str(args.config).lower() else
         f'(no published reference for this config — record it as the new baseline)\n'),
        '## Headline\n',
        f'- Labelled (non-no-data) pixels: **{total_valid:,}**',
        f'- Pixels with a real class (excl. background): **{total_real:,}** '
        f'({100*total_real/max(total_valid,1):.1f}%)',
        f'- **Of those, discarded to background: {total_discarded:,} '
        f'({100*total_discarded/max(total_real,1):.2f}%)**\n',
        f'- Per-image discard rate: mean **{pcts.mean():.2f}%**, '
        f'median {np.median(pcts):.2f}%, max {pcts.max():.2f}%\n',
        '## Loss by class\n',
        '| Class | GT pixels | Lost to background | % lost |',
        '|---|---|---|---|',
    ]
    for r in rows:
        summary.append(f"| {r['class']} | {r['gt_pixels']:,} | "
                       f"{r['lost_to_background']:,} | **{r['pct_lost']}%** |")
    # ---- instrumentation: does presence collapse explain the bad tiles? ----
    # WEEK1_RESULTS 9.2 rests on ONE tile (3487). This generalises it.
    if pres_rows and has_presence:
        sp = np.array([r['spres_max'] for r in pres_rows], dtype=float)
        dp = np.array([r['discard_pct_of_real'] for r in per_image], dtype=float)
        ok = np.isfinite(sp)
        cat, hea = ok & (dp >= 99.0), ok & (dp < 1.0)
        summary += [
            '\n## Presence-head collapse — catastrophic vs healthy tiles\n',
            '`spres_max` = highest presence score over the six real classes '
            '(max across sliding-window crops).\n',
            '| Tile set | n | mean spres_max | median | p90 |',
            '|---|---|---|---|---|',
        ]
        for lbl, m in (('catastrophic (>=99% discard)', cat), ('healthy (<1% discard)', hea)):
            if m.sum():
                summary.append(f'| {lbl} | {int(m.sum())} | {sp[m].mean():.4f} | '
                               f'{np.median(sp[m]):.4f} | {np.percentile(sp[m], 90):.4f} |')
            else:
                summary.append(f'| {lbl} | 0 | - | - | - |')
        if ok.sum() > 2:
            r = float(np.corrcoef(sp[ok], dp[ok])[0, 1])
            summary.append(f'\n- Correlation(spres_max, discard%) = **{r:+.3f}** '
                           f'over {int(ok.sum())} tiles.')
        summary += [
            '\n**How to read it.** If the catastrophic set clusters low (~<0.2) while the '
            'healthy set sits high, presence collapse is a systematic failure mode and '
            'WEEK1_RESULTS 9.2 generalises from n=1 — a figure, and a claim SegEarth-OV3 '
            'does not make. If the two distributions overlap, tile 3487 was an anecdote '
            'and must be dropped from the writeup.\n',
        ]

    summary += [
        '\n## Read this\n',
        '- **> 15% of real-class pixels lost** → premise confirmed, proceed with the '
        'co-occurrence prior.',
        '- **5–15%** → real but modest; the gain ceiling is limited, say so explicitly.',
        '- **< 5%** → premise weak. Pivot to the medium-resolution domain gap '
        '(GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).\n',
        'Compare against the τ-sweep before concluding: if τ=0.1 recovers these pixels '
        'without hurting precision, the trivial fix suffices and the method needs a '
        'sharper justification.\n',
    ]
    (out / 'discard_summary.md').write_text('\n'.join(summary))

    # ---- figures ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(N)); ax.set_xticklabels(CLASSES, rotation=45, ha='right')
        ax.set_yticks(range(N)); ax.set_yticklabels(CLASSES)
        ax.set_xlabel('predicted'); ax.set_ylabel('true')
        ax.set_title('Row-normalised confusion matrix')
        for a in range(N):
            for b in range(N):
                if cm[a, b] > 0.01:
                    ax.text(b, a, f'{cm[a,b]*100:.0f}', ha='center', va='center',
                            fontsize=8, color='white' if cm[a, b] > 0.5 else 'black')
        fig.colorbar(im); plt.tight_layout()
        plt.savefig(out / 'confusion_matrix.png', dpi=140); plt.close()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh([r['class'] for r in rows][::-1],
                [r['pct_lost'] for r in rows][::-1], color='#c0392b')
        ax.set_xlabel('% of class pixels discarded to background')
        ax.set_title(f'What SegEarth-OV3 throws away (τ={tau_str})')
        plt.tight_layout(); plt.savefig(out / 'discard_by_class.png', dpi=140); plt.close()
    except Exception as e:
        print(f'(figures skipped: {e})')

    print('\n' + '\n'.join(summary[:14]))
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()



