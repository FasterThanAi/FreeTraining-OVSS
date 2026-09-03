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
    ap.add_argument('--limit', type=int, default=0,
                    help='0 = all images. ⚠️ Takes the FIRST n in FILENAME order, '
                         'which is NOT a representative subset when a split merges '
                         'domains with disjoint ID ranges — on LoveDA val (Rural '
                         '992 + Urban 677) --limit 500 is essentially rural-only. '
                         'Use --sample for a subset you intend to generalise from.')
    ap.add_argument('--sample', type=int, default=0,
                    help='draw n images at RANDOM instead of taking the first n. '
                         'Use this whenever the subset stands in for the split.')
    ap.add_argument('--seed', type=int, default=0, help='--sample seed')
    ap.add_argument('--ext', default=None,
                    help='image extension; auto-detected (.png / .tif) if unset')
    ap.add_argument('--reduce-zero-label', type=lambda v: v.lower() == 'true',
                    default=None,
                    help='override the config. True = raw 0 is no-data (LoveDA); '
                         'False = raw 0 is a real class (OpenEarthMap).')
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
    # <<< VOCABULARY INTERVENTION (ROADMAP §6.1). Every class is an INDEPENDENT
    # forward pass with its own text prompt (segmentor:141-189) and the only
    # cross-class operation in the pipeline is the argmax in predict(). So
    # dropping a class from the vocabulary is EXACTLY equivalent to dropping its
    # channel here. Caching the full (N, H, W) stack therefore makes every
    # vocabulary arm a CPU pass over ONE set of model outputs -- which is not
    # merely cheaper than re-running, it is a cleaner intervention, because the
    # arms then differ in the vocabulary and in nothing else at all.
    ap.add_argument('--cache-full', action='store_true',
                    help='also store the full per-class score stack `logits` '
                         '(N, H, W) float16. ~10x the cache size; required by '
                         'vocab_intervention.py and by nothing else.')
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # LoveDA ships .png, OpenEarthMap .tif. Detect rather than assume.
    exts = [args.ext] if args.ext else ['.png', '.tif', '.tiff', '.jpg']
    ext = next((e for e in exts if any(Path(args.img_dir).glob(f'*{e}'))), None)
    if ext is None:
        raise SystemExit(
            f'no images matching {exts} in {args.img_dir!r}\n'
            f'  cwd is {Path.cwd()}\n'
            f'  exists: {Path(args.img_dir).exists()}\n'
            '  If this is OpenEarthMap, the config expects val/images and '
            'val/labels; the Kaggle archive ships images/val and label/val. '
            'Create the symlinks first.')
    ann_ext = next((e for e in ['.png', '.tif', '.tiff']
                    if any(Path(args.ann_dir).glob(f'*{e}'))), ext)
    names = sorted(p.stem for p in Path(args.img_dir).glob(f'*{ext}'))
    print(f'  image ext {ext}, annotation ext {ann_ext}')
    if args.limit and args.sample:
        sys.exit('ERROR: --limit and --sample both given; they mean different '
                 'things. --limit takes the first n (a smoke test), --sample '
                 'draws n at random (a representative subset). Pick one.')
    if args.limit:
        names = names[:args.limit]
        # LoveDA val is Rural 992 + Urban 677 merged, with disjoint ID ranges, so
        # the first 500 filenames are one domain. That produced a 40.94 mIoU /
        # 41.15% discard run that looked like a broken baseline and was actually
        # a rural-only sample. Say so at the top rather than in the numbers.
        print(f'  ⚠️  --limit takes the FIRST {args.limit} in filename order, which '
              f'is not a random subset. If this split merges domains, expect one '
              f'of them. Use --sample to generalise from the result.')
    elif args.sample:
        rng = np.random.default_rng(args.seed)
        names = sorted(rng.choice(names, min(args.sample, len(names)),
                                  replace=False).tolist())
        print(f'  --sample {args.sample} at seed {args.seed} '
              f'(random, representative of the split)')
    print(f'{len(names)} images | config {args.config} | out {out}')

    if args.tau is not None:
        print(f'  overriding prob_thd -> {args.tau}')

    from mmengine.config import Config
    cfg = Config.fromfile(args.config)

    # ---- label convention. THIS IS LOAD-BEARING AND DATASET-SPECIFIC.
    #
    # LoveDA sets reduce_zero_label=True: raw mask 0 is no-data and the classes
    # occupy 1..N. OpenEarthMap sets it False: raw mask 0 IS `background`, a real
    # scored class, and the classes occupy 0..N-1.
    #
    # Everything downstream (labels.py, the .npz cache, every Week 3 script)
    # assumes "0 = ignore, class i at mask value i+1". Feeding OEM in raw would
    # make `valid = gt > 0` silently delete every background pixel -- the exact
    # class this project is about -- and nothing would crash. So normalise here,
    # once, at the only place that reads the raw file.
    try:
        rzl = cfg.test_dataloader['dataset'].get('reduce_zero_label', True)
    except Exception:
        rzl = True
    if args.reduce_zero_label is not None:
        rzl = args.reduce_zero_label
    if rzl:
        shift_gt = lambda g: g                      # already 0=ignore, 1..N
    else:
        def shift_gt(g):                            # 0..N-1  ->  1..N
            g = g.astype(np.int32)
            return np.where(g == 255, 0, g + 1)     # 255 is mmseg's ignore
    print(f'  reduce_zero_label={rzl} -> '
          + ('raw masks used as-is (0 = no-data)' if rzl
             else 'raw masks shifted +1 (raw 0 = background, no no-data value)'))
    model = init_model(args.config, device='cuda')
    if args.tau is not None:
        # set_prob_thd, not a bare attribute write: the segmentor now also carries
        # `prob_thd_vec` for per-class thresholds, and assigning `prob_thd`
        # directly would leave a vector from the config in place and silently
        # ignore --tau. The setter clears it.
        if hasattr(model, 'set_prob_thd'):
            model.set_prob_thd(args.tau)
        else:
            model.prob_thd = args.tau
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
        # ~2.5 MB/image compressed; the full stack adds ~2 bytes per class-pixel,
        # which for 7 classes at 1024^2 is ~15 MB before compression.
        need_gb = (0.025 if args.cache_full else 0.0025) * len(names)
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
        img_p = f'{args.img_dir}/{name}{ext}'
        gt = np.array(Image.open(f'{args.ann_dir}/{name}{ann_ext}'))
        gt = shift_gt(gt)
        result = inference_model(model, img_p)
        pred = result.pred_sem_seg.data.cpu().numpy().squeeze()
        pred = pred.astype(np.int64) + 1          # 0-indexed -> 1..7

        valid = gt > 0                            # drop no-data
        g, p = gt[valid], pred[valid]
        np.add.at(conf, (g - 1, p - 1), 1)

        # ⚠️ NOT `g > BACKGROUND`. That assumed the catch-all is the LOWEST class
        # value, which holds for LoveDA and OpenEarthMap (both mask value 1) and
        # FAILS on Potsdam, where `clutter` is mask value 6 -- the highest. There
        # `g > 6` matches nothing, every tile reports 0.00% discard, and the
        # catch-all is counted as a real class losing pixels to itself.
        real = (g > 0) & (g != BACKGROUND)        # carries a real land-cover label
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

            # <<< CROSS-HEAD: the two SAM 3 heads SEPARATELY. `fused` is their
            # elementwise max, so after the fact there is no way to ask whether
            # they agreed -- and cross-head agreement is the leading label-free
            # candidate for per-class precision, which WEEK3 §9a shows is what
            # sets the right threshold. Stored as top-1 per head: which class
            # each head would pick alone, and how strongly.
            for key, attr in (('i', 'last_inst'), ('s', 'last_sem')):
                hl = getattr(model, attr, None)
                if hl is None:
                    continue
                try:
                    hl = hl.float()
                    if hl.shape[-2:] != lg.shape[-2:]:
                        hl = torch.nn.functional.interpolate(
                            hl[None], size=lg.shape[-2:], mode='bilinear',
                            align_corners=False)[0]
                    htop = torch.topk(hl, k=1, dim=0)
                    fused_arrays[f'{key}conf'] = \
                        htop.values[0].cpu().numpy().astype(np.float16)
                    fused_arrays[f'{key}pred'] = \
                        htop.indices[0].cpu().numpy().astype(np.uint8)
                except Exception as e:
                    print(f'    ({attr} capture failed on {name}: {e})')

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
                **({'logits': lg.cpu().numpy().astype(np.float16)}
                   if args.cache_full else {}),             # <<< VOCABULARY INTERVENTION
            )

        if i % 100 == 0 or i == len(names):
            print(f'  {i}/{len(names)}')

    # ---- derived numbers -------------------------------------------------
    total_valid = conf.sum()
    # ⚠️ Rows are 0-indexed classes, so the catch-all is row BACKGROUND-1 -- NOT
    # row 0. `conf[1:]` skipped row 0 unconditionally, which silently kept the
    # catch-all in the real-class total on any dataset whose catch-all is not
    # first. Same bug as the `g > BACKGROUND` line above.
    real_rows = [c for c in range(N) if c != BACKGROUND - 1]
    total_real = conf[real_rows, :].sum()
    total_discarded = conf[real_rows, BACKGROUND - 1].sum()

    rows = []
    for c in real_rows:                            # skip the catch-all as a true class
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
        # NOTE the leading `+`. Without it this is implicit string concatenation
        # followed by a parenthesised expression, i.e. CALLING the string --
        # which is exactly how this broke: TypeError, 'str' object is not callable.
        + (f'(baseline reference: 47.38 — if these disagree, the label alignment '
           f'is wrong)\n' if 'loveda' in str(args.config).lower() else
           f'(no published reference for this config — record it as the new '
           f'baseline)\n'),
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



