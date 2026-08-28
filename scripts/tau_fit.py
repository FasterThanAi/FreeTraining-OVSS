"""
Per-class thresholds fitted on TRAIN, evaluated on VAL. A method, not a bound.

WHY THIS IS FAIR, WHICH IS THE WHOLE POINT. tau_oracle.py found per-class
thresholds worth +1.46 mIoU on LoveDA -- the only result in this project where
land cover genuinely improves, six real classes gaining 8.63 IoU in aggregate.
But it chose those thresholds ON the evaluation labels, so it is a ceiling, not
something anyone could deploy.

SegEarth-OV3 does not use a label-free rule either. It tunes tau PER DATASET
using labels: 0.5 for LoveDA, 0.1 for OpenEarthMap. So fitting per-class
thresholds on a TRAIN split and evaluating on val is the same protocol the
baseline already uses, with 6 free parameters instead of 1. No model weights are
trained, so the pipeline stays training-free. That is a fair comparison rather
than a leak, and it is the difference between "+1.46 exists in principle" and
"+X is what you get".

THE ROWS, and the honesty lives in the gap between the last two:

    published tau            what the baseline ships          -> val
    global tau, fit on TRAIN  1 parameter, their protocol      -> val
    per-class tau, fit on TRAIN  N parameters, our proposal    -> val   <- THE RESULT
    per-class tau, fit on VAL    the oracle ceiling            -> val   <- NOT a result

Train mIoU is printed beside val mIoU for every fitted row, so overfitting is
visible rather than inferred. Six parameters on thousands of images should
generalise, but "should" is not a measurement.

    python scripts/tau_fit.py \
        --train-cache ~/outputs/loveda_train/cache \
        --val-cache   ~/outputs/week3_fused/cache --tau 0.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                             # noqa: E402
from tau_oracle import confusion_at, miou, per_class_iou, NBINS   # noqa: E402


def build(files, nc, nbins, tag):
    H = np.zeros((nc, nc, nbins), np.int64)
    for i, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.int32)
        conf = z['conf'].astype(np.float32)
        pred = z['pred'].astype(np.int32)
        m = gt > 0
        if m.any():
            b = np.clip((conf[m] * nbins).astype(np.int32), 0, nbins - 1)
            np.add.at(H, (gt[m] - 1, np.clip(pred[m], 0, nc - 1), b), 1)
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  {tag} {i + 1}/{len(files)}')
    return H


def fit(H, bg, nbins, rounds, start=None):
    """Coordinate ascent over per-class thresholds, maximising mIoU on H."""
    nc = H.shape[0]
    grid = np.arange(nbins + 1) / nbins
    if start is None:
        gl = [(miou(confusion_at(H, np.full(nc, t), bg, nbins)), t) for t in grid]
        best, t0 = max(gl)
        taus = np.full(nc, t0)
    else:
        taus = np.asarray(start, float).copy()
        best = miou(confusion_at(H, taus, bg, nbins))
    for _ in range(rounds):
        moved = False
        for c in range(nc):
            if c == bg:
                continue                  # tau[bg] cannot change any assignment
            cand = [(miou(confusion_at(H, np.where(np.arange(nc) == c, t, taus),
                                       bg, nbins)), t) for t in grid]
            sc, st = max(cand)
            if sc > best + 1e-9:
                best, taus[c], moved = sc, st, True
        if not moved:
            break
    return taus, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-cache', required=True)
    ap.add_argument('--val-cache', required=True)
    ap.add_argument('--tau', type=float, required=True, help='the published τ')
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LT = labels.from_cache(args.train_cache)
    LV = labels.from_cache(args.val_cache)
    if LT.names != LV.names:
        raise SystemExit(f'class lists differ!\n  train {LT.names}\n  val   {LV.names}\n'
                         'Fitting on one and evaluating on the other would be meaningless.')
    nc, bg = LV.n, LV.bg - 1
    print(f'  classes: {LV}')

    tr = sorted(Path(args.train_cache).expanduser().glob('*.npz'))
    va = sorted(Path(args.val_cache).expanduser().glob('*.npz'))
    if args.limit:
        tr, va = tr[:args.limit], va[:args.limit]
    if not tr or not va:
        raise SystemExit('empty cache')
    inter = {p.stem for p in tr} & {p.stem for p in va}
    if inter:
        raise SystemExit(
            f'{len(inter)} tile ids appear in BOTH caches, e.g. {sorted(inter)[:5]}. '
            'Fitting and evaluating on the same tiles is exactly the leak this script '
            'exists to avoid.')
    print(f'train {len(tr)} tiles | val {len(va)} tiles | published τ = {args.tau}\n')

    Htr = build(tr, nc, NBINS, 'train')
    Hva = build(va, nc, NBINS, 'val  ')

    def ev(H, t):
        return miou(confusion_at(H, np.asarray(t, float), bg, NBINS))

    pub = np.full(nc, args.tau)
    m_pub_v = ev(Hva, pub)
    m_pub_t = ev(Htr, pub)

    # 1 parameter, fitted on train -- the baseline's own protocol
    grid = np.arange(NBINS + 1) / NBINS
    m_g_t, t_g = max((ev(Htr, np.full(nc, t)), t) for t in grid)
    g_taus = np.full(nc, t_g)
    m_g_v = ev(Hva, g_taus)

    # N parameters, fitted on train -- the proposal
    f_taus, m_f_t = fit(Htr, bg, NBINS, args.rounds)
    m_f_v = ev(Hva, f_taus)

    # N parameters, fitted on VAL -- the ceiling, never a result
    o_taus, m_o_v = fit(Hva, bg, NBINS, args.rounds)

    md = ['# Per-class τ, fitted on train and evaluated on val\n',
          f'- train: **{len(tr)}** tiles (`{args.train_cache}`)',
          f'- val: **{len(va)}** tiles (`{args.val_cache}`)',
          f'- classes: **{nc}**  |  published τ: **{args.tau}**  |  no tile id in both\n',
          'SegEarth-OV3 tunes τ per dataset using labels, so fitting thresholds on a train '
          'split and evaluating on val is **the same protocol with more parameters**. No '
          'model weights are trained; the pipeline stays training-free.\n',
          '| rule | params | fitted on | train mIoU | **val mIoU** | Δ val |',
          '|---|---|---|---|---|---|',
          f'| published τ = {args.tau} | 1 | (theirs) | {m_pub_t:.2f} | **{m_pub_v:.2f}** | — |',
          f'| global τ = {t_g:.3f} | 1 | train | {m_g_t:.2f} | **{m_g_v:.2f}** | '
          f'{m_g_v - m_pub_v:+.2f} |',
          f'| **per-class τ** | {nc - 1} | **train** | {m_f_t:.2f} | **{m_f_v:.2f}** | '
          f'**{m_f_v - m_pub_v:+.2f}** |',
          f'| _per-class τ (oracle)_ | {nc - 1} | _val_ | — | _{m_o_v:.2f}_ | '
          f'_{m_o_v - m_pub_v:+.2f}_ |\n',
          f'Generalisation gap on the fitted row: train {m_f_t:.2f} → val {m_f_v:.2f} '
          f'(**{m_f_v - m_f_t:+.2f}**). The oracle row is fitted on val and is a ceiling, '
          'never a result.\n',
          '## Thresholds\n',
          '| class | fitted on train | oracle (val) | published |', '|---|---|---|---|']
    for k in range(nc):
        if k == bg:
            md.append(f'| {LV.names[k]} | — *(no effect)* | — | — |')
            continue
        md.append(f'| {LV.names[k]} | **{f_taus[k]:.3f}** | {o_taus[k]:.3f} | {args.tau} |')

    v0 = per_class_iou(confusion_at(Hva, pub, bg, NBINS))
    v1 = per_class_iou(confusion_at(Hva, f_taus, bg, NBINS))
    d = v1 - v0
    present = Hva.sum(axis=(1, 2))
    md += ['\n## Per-class IoU on val, published τ vs fitted\n',
           '| class | published | fitted | Δ |', '|---|---|---|---|']
    for k in range(nc):
        if present[k] == 0:
            md.append(f'| {LV.names[k]} | *absent* | | |')
            continue
        md.append(f'| {LV.names[k]} | {v0[k]:.2f} | {v1[k]:.2f} | **{d[k]:+.2f}** |')
    realk = [k for k in range(nc) if k != bg and present[k] > 0]
    bgd = float(d[bg]) if present[bg] > 0 and np.isfinite(d[bg]) else 0.0
    reald = float(np.nansum(d[realk]))
    md.append(f'\n`background` **{bgd:+.2f}**, the {len(realk)} real classes '
              f'**{reald:+.2f}** in aggregate.\n')

    gain = m_f_v - m_pub_v
    capt = (gain / (m_o_v - m_pub_v) * 100) if m_o_v - m_pub_v > 1e-9 else float('nan')
    md += ['## Verdict\n']
    if gain <= 0.05:
        md.append(f'⛔ **The fitted thresholds do not transfer** ({gain:+.2f} on val against '
                  f'{m_f_t - m_pub_t:+.2f} on train). The oracle gain is real but only '
                  'reachable by fitting on the evaluation set, so per-class thresholding '
                  'is a property of the split rather than a method. Report the bound, not '
                  'a method.')
    elif reald <= 0:
        md.append(f'⚠️ **{gain:+.2f} mIoU on val, but the real classes lose {reald:+.2f}** — '
                  'the background-unwinding pattern again. A true mIoU gain that is not '
                  'better land-cover classification. Never report it without the per-class '
                  'table.')
    else:
        md.append(f'✅ **{gain:+.2f} mIoU on val, with the real classes gaining '
                  f'{reald:+.2f} in aggregate** — {capt:.0f}% of the oracle bound, from '
                  'thresholds fitted on a disjoint train split. Same protocol the baseline '
                  'uses for its own τ, more parameters, no weights trained.\n\n'
                  'Before claiming it: re-run on the second dataset, report the '
                  'generalisation gap above, and give the per-class thresholds so a reader '
                  'can see the spread a single τ was forced to average over.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
