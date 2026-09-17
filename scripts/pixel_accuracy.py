"""Pixel accuracy -- overall and per class -- for rungs A / B / C, from a cache.

WHAT. For ONE dataset per run, on the SAME held-out tiles eval.py was run on:

    A  published global tau, no scale        (the baseline)
    B  per-class tau, no scale                (lever 1)
    C  per-class scale + per-class tau        (lever 1 + lever 2)

it reports, exactly as mmseg's IoUMetric defines them:

    aAcc            correct labelled pixels / all labelled pixels
    per-class Acc   correct pixels of class c / all pixels of class c  (= recall)
    precision, IoU, mIoU   -- mIoU is printed ONLY as a check against eval.py

⭐ NO GPU. The cache holds the full score stack, and each rung is arithmetic on
it -- the same arithmetic that predicted eval.py to within 0.07 mIoU. Tiles are
streamed one at a time, so memory stays small and one CPU core is used.

⛔ THE GATE. mIoU is printed beside eval.py's measured value for every rung. If
any rung is off by more than ~0.1, the accuracy numbers are not trustworthy
either -- stop and find out why before plotting anything.

Discarded pixels go to the segmentor's `bg_idx`:
  DLRSD    bg_idx = 17, outside the 17 classes: an UNSCORED sink, never correct.
  Potsdam  bg_idx = 5 = `clutter`: a discard IS a clutter prediction, and is
           correct where the ground truth is clutter.

    python scripts/pixel_accuracy.py --preset dlrsd
    python scripts/pixel_accuracy.py --preset potsdam
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as LB  # noqa: E402

PRESETS = {
    'dlrsd': dict(
        cache='~/outputs/dlrsd_full/cache', tau=0.1,
        split='~/splits/dlrsd_heldout.txt',
        tau_cfg='~/SegEarth-OV-3/configs/cfg_dlrsd_tau.py',
        scale_cfg='~/SegEarth-OV-3/configs/cfg_dlrsd_perclass.py',
        ref_miou='37.27,39.04,44.42', ref_aacc='58.94,,65.78',
        expect_tiles=1701,
        json='results/dlrsd/pixel_accuracy.json',
        md='results/dlrsd/pixel_accuracy.md'),
    'potsdam': dict(
        cache='~/outputs/potsdam_full/cache', tau=0.1,
        split='~/splits/potsdam_reorder_heldout.txt',
        tau_cfg='~/SegEarth-OV-3/configs/cfg_potsdam_tauonly.py',
        scale_cfg='~/SegEarth-OV-3/configs/cfg_potsdam_reorder.py',
        ref_miou='57.60,58.35,63.27', ref_aacc=',,',
        expect_tiles=1816,
        json='results/potsdam/pixel_accuracy.json',
        md='results/potsdam/pixel_accuracy.md'),
}
RUNGS = ['A', 'B', 'C']
RUNG_NAME = {'A': 'baseline (published τ)', 'B': '+ per-class τ',
             'C': '+ per-class scale'}


def read_vector(cfg_path, key, n):
    """A list out of a GENERATED config. Read, never retyped: the segmentor checks
    the length of these vectors and cannot detect a permutation."""
    p = Path(cfg_path).expanduser()
    if not p.is_file():
        raise SystemExit(f'⛔ config not found: {p}')
    m = re.search(rf'^\s*{key}\s*=\s*\[(.*?)\]', p.read_text(), re.S | re.M)
    if not m:
        return None
    vals = [float(x) for x in m.group(1).replace('\n', ' ').split(',') if x.strip()]
    if len(vals) != n:
        raise SystemExit(f'⛔ {key} in {p} has {len(vals)} values, the cache has '
                         f'{n} classes.')
    return np.asarray(vals, np.float32)


def parse_refs(s):
    out = []
    for x in (s or '').split(','):
        out.append(float(x) if x.strip() else None)
    return (out + [None, None, None])[:3]


def decide(flat, w, thr, bg):
    """The segmentor's rule: scale the argmax, threshold the RAW winning score.
    `thr` is a scalar (rung A) or a per-class vector indexed by the argmax."""
    s = flat if w is None else flat * w[:, None]
    pred = np.argmax(s, 0)
    raw = flat[pred, np.arange(flat.shape[1])]
    t = thr if np.ndim(thr) == 0 else thr[pred]
    fired = raw < t
    out = pred.copy()
    out[fired] = bg
    return out, fired


def metrics(C, n):
    """mmseg IoUMetric, from a (n x n+1) confusion matrix, rows = truth."""
    inter = np.diag(C[:, :n]).astype(np.float64)
    label = C.sum(1).astype(np.float64)
    pred = C[:, :n].sum(0).astype(np.float64)
    union = label + pred - inter
    with np.errstate(invalid='ignore', divide='ignore'):
        acc = 100 * inter / label
        iou = 100 * inter / union
        prec = 100 * inter / pred
    return dict(aAcc=float(100 * inter.sum() / label.sum()),
                mIoU=float(np.nanmean(iou)), mAcc=float(np.nanmean(acc)),
                acc=acc, iou=iou, prec=prec)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=sorted(PRESETS))
    ap.add_argument('--dataset-name')
    ap.add_argument('--cache')
    ap.add_argument('--tau', type=float)
    ap.add_argument('--split')
    ap.add_argument('--tau-cfg')
    ap.add_argument('--scale-cfg')
    ap.add_argument('--ref-miou', help='eval.py mIoU for A,B,C')
    ap.add_argument('--ref-aacc', help='eval.py aAcc for A,B,C (blank = unknown)')
    ap.add_argument('--expect-tiles', type=int)
    ap.add_argument('--json')
    ap.add_argument('--md')
    args = ap.parse_args()

    # A preset fills only what was NOT given explicitly.
    if args.preset:
        for k, v in PRESETS[args.preset].items():
            if getattr(args, k) is None:
                setattr(args, k, v)
        args.dataset_name = args.dataset_name or args.preset
    for k in ('cache', 'tau', 'split', 'tau_cfg', 'scale_cfg', 'json', 'md'):
        if getattr(args, k) is None:
            raise SystemExit(f'⛔ --{k.replace("_", "-")} is required (or use --preset)')

    cache = Path(args.cache).expanduser()
    lab = LB.from_cache(cache)
    n, names = lab.n, lab.names
    if lab.bg is None:
        raise SystemExit('⛔ could not determine where discarded pixels go (bg_idx)')
    bg = lab.bg - 1                                  # 0-indexed segmentor bg_idx
    print(f'  {args.dataset_name}: {n} classes, discarded pixels -> index {bg} '
          f'({"unscored sink" if bg >= n else names[bg]})')

    tau_b = read_vector(args.tau_cfg, 'prob_thd', n)
    w_c = read_vector(args.scale_cfg, 'class_scale', n)
    tau_c = read_vector(args.scale_cfg, 'prob_thd', n)
    if tau_b is None or tau_c is None or w_c is None:
        raise SystemExit('⛔ prob_thd missing from --tau-cfg, or class_scale/prob_thd '
                         'missing from --scale-cfg')

    stems = [s.strip() for s in Path(args.split).expanduser().read_text().splitlines()
             if s.strip()]
    missing = [s for s in stems if not (cache / f'{s}.npz').is_file()]
    if missing:
        raise SystemExit(f'⛔ {len(missing)} split tiles have no cache file, e.g. '
                         f'{missing[:3]}')
    if args.expect_tiles and len(stems) != args.expect_tiles:
        raise SystemExit(f'⛔ split has {len(stems)} tiles, expected '
                         f'{args.expect_tiles} (the eval.py held-out set)')
    print(f'  {len(stems)} held-out tiles from {args.split}')

    conf = {r: np.zeros((n, n + 1), np.int64) for r in RUNGS}
    fired_px = {r: 0 for r in RUNGS}
    labelled = 0
    for i, s in enumerate(stems, 1):
        z = np.load(cache / f'{s}.npz')
        if 'logits' not in z.files:
            raise SystemExit(f'⛔ {s}.npz has no `logits`: this needs a --cache-full cache')
        L = z['logits'].astype(np.float32)
        gt = z['gt'].astype(np.int64)
        if L.shape[0] != n or L.shape[1:] != gt.shape:
            raise SystemExit(f'⛔ {s}: logits {L.shape} vs gt {gt.shape}, {n} classes')
        m = gt > 0                                   # 0 = no-data, ignored
        if not m.any():
            continue
        g = gt[m] - 1
        flat = L[:, m]
        labelled += g.size
        for r, (w, thr) in {'A': (None, np.float32(args.tau)),
                            'B': (None, tau_b), 'C': (w_c, tau_c)}.items():
            out, fired = decide(flat, w, thr, bg)
            conf[r] += np.bincount(g * (n + 1) + out,
                                   minlength=n * (n + 1)).reshape(n, n + 1)
            fired_px[r] += int(fired.sum())
        if i % 200 == 0 or i == len(stems):
            print(f'    {i}/{len(stems)}')

    ref_m, ref_a = parse_refs(args.ref_miou), parse_refs(args.ref_aacc)
    gt_px = conf['A'].sum(1)
    res = dict(dataset=args.dataset_name, n_tiles=len(stems), classes=names,
               bg_idx=int(bg), sink=bool(bg >= n), labelled_px=int(labelled),
               gt_share=[float(100 * x / gt_px.sum()) for x in gt_px], rungs={})
    lines = [f'# Pixel accuracy — {args.dataset_name}', '',
             f'- {len(stems)} held-out tiles (`{args.split}`), {labelled:,} labelled pixels',
             f'- discarded pixels go to index {bg} '
             f'({"an unscored sink — never correct" if bg >= n else "`" + names[bg] + "`, scored"})',
             '- computed from the cache; mIoU is shown only to check against eval.py', '',
             '| rung | pixel accuracy (aAcc) | eval.py aAcc | mIoU | eval.py mIoU | gate | discarded |',
             '|---|---|---|---|---|---|---|']
    worst = 0.0
    for k, r in enumerate(RUNGS):
        M = metrics(conf[r], n)
        dm = None if ref_m[k] is None else M['mIoU'] - ref_m[k]
        da = None if ref_a[k] is None else M['aAcc'] - ref_a[k]
        for d in (dm, da):
            if d is not None:
                worst = max(worst, abs(d))
        ok = all(d is None or abs(d) <= 0.15 for d in (dm, da))
        gate = '—' if dm is None and da is None else ('✅' if ok else '⛔')
        res['rungs'][r] = dict(
            name=RUNG_NAME[r], aAcc=M['aAcc'], mIoU=M['mIoU'], mAcc=M['mAcc'],
            discarded_pct=100 * fired_px[r] / labelled,
            ref_mIoU=ref_m[k], ref_aAcc=ref_a[k],
            acc=[None if np.isnan(x) else float(x) for x in M['acc']],
            prec=[None if np.isnan(x) else float(x) for x in M['prec']],
            iou=[None if np.isnan(x) else float(x) for x in M['iou']])
        fa = '' if ref_a[k] is None else f'{ref_a[k]:.2f} ({da:+.2f})'
        fm = '' if ref_m[k] is None else f'{ref_m[k]:.2f} ({dm:+.2f})'
        lines.append(f'| **{r}** {RUNG_NAME[r]} | **{M["aAcc"]:.2f}** | {fa} | '
                     f'{M["mIoU"]:.2f} | {fm} | {gate} | '
                     f'{100 * fired_px[r] / labelled:.2f}% |')
        print(f'  {r}: aAcc {M["aAcc"]:.2f}  mIoU {M["mIoU"]:.2f}  '
              f'(eval.py mIoU {ref_m[k]})  discarded {100 * fired_px[r] / labelled:.2f}%')

    checked = any(v is not None for v in ref_m + ref_a)
    passed = checked and worst <= 0.15
    verdict = ('⛔ NO eval.py REFERENCE GIVEN — nothing was checked, so these numbers '
               'are unverified. Pass --ref-miou (and --ref-aacc if known).' if not checked else
               '✅ GATE PASSED — every rung within 0.15 of eval.py (largest gap '
               f'{worst:.2f}), so the accuracy numbers can be trusted.' if passed else
               f'⛔ GATE FAILED — a rung is {worst:.2f} away from eval.py. Do NOT '
               'plot or quote these numbers until the cause is found.')
    res['gate_passed'] = bool(passed)
    res['gate_worst_diff'] = worst
    lines += ['', verdict, '', '## Per-class pixel accuracy (% of that class\'s pixels labelled correctly)', '',
              '| class | share of pixels | A | B | C | C − A |', '|---|---|---|---|---|---|']
    ra, rb, rc = (res['rungs'][r]['acc'] for r in RUNGS)
    order = sorted(range(n), key=lambda c: -((rc[c] or 0) - (ra[c] or 0)))
    for c in order:
        f = lambda v: '—' if v is None else f'{v:.2f}'  # noqa: E731
        d = '—' if ra[c] is None or rc[c] is None else f'{rc[c] - ra[c]:+.2f}'
        lines.append(f'| {names[c]} | {res["gt_share"][c]:.2f}% | {f(ra[c])} | '
                     f'{f(rb[c])} | {f(rc[c])} | **{d}** |')
    lines += ['', '⚠️ Per-class accuracy is RECALL: it rises whenever a class is '
              'predicted more, even wrongly. Read it beside precision '
              '(in the JSON) or IoU, never alone.']

    for p, text in ((args.json, json.dumps(res, indent=1)), (args.md, '\n'.join(lines) + '\n')):
        p = Path(p).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'  wrote {p}')
    print('\n' + verdict)


if __name__ == '__main__':
    main()
