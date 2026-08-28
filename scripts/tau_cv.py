"""
Per-class τ: cross-validated, and how much calibration data it needs.

WHY. A single 50/50 split of LoveDA val gave +1.44 mIoU with the real classes
gaining 8.06 -- the first positive method result in this project. One split is
one draw. Before that number is quoted anywhere it needs an error bar, and the
obvious follow-up needs an answer: how many labelled tiles does the calibration
actually take? If it is 800, the method is impractical. If it is 50, it is
something a practitioner would do.

HOW IT IS CHEAP. The confusion matrix at any threshold vector is a function of a
(gt, pred, conf-bin) histogram, and histograms ADD. So each tile's own 7x7x200
histogram is built once -- about 130 MB for LoveDA val -- and every split
thereafter is a subset sum. Hundreds of fits cost seconds, and no tile is ever
re-read.

TWO EXPERIMENTS

  k-fold        fit on k-1 folds, evaluate on the held-out one, rotate. Reports
                mean and spread of the gain, so a lucky partition cannot be
                mistaken for a result.

  learning curve  fit on n randomly drawn tiles, evaluate on everything else,
                repeat. Answers "how many labels does this cost?", which decides
                whether it is a method or a curiosity.

⚠️ Calibration and evaluation tiles are always disjoint. The published-τ baseline
is recomputed on the SAME held-out tiles as the fitted rule, so the two are
compared on identical pixels rather than against a global figure.

    python scripts/tau_cv.py --cache ~/outputs/week3_fused/cache --tau 0.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                             # noqa: E402
from tau_oracle import confusion_at, miou, per_class_iou, NBINS   # noqa: E402


def per_tile_hists(files, nc, nbins):
    """One (gt, pred, conf-bin) histogram per tile. Subset sums give any split."""
    H = np.zeros((len(files), nc, nc, nbins), np.int32)
    for i, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.int32)
        conf = z['conf'].astype(np.float32)
        pred = z['pred'].astype(np.int32)
        m = gt > 0
        if m.any():
            b = np.clip((conf[m] * nbins).astype(np.int32), 0, nbins - 1)
            h = np.zeros((nc, nc, nbins), np.int64)
            np.add.at(h, (gt[m] - 1, np.clip(pred[m], 0, nc - 1), b), 1)
            H[i] = h
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}')
    return H


def obj_miou(C, bg, objective):
    """The quantity the fit maximises.

    'all'  -- mIoU over every class, the reported metric.
    'real' -- mIoU over the real classes only, excluding the catch-all.

    This distinction decides the OpenEarthMap result. There, `background` sits at
    17.13 IoU with 17.33% precision, so roughly 54 points are available from that
    single class -- and an optimiser maximising full mIoU will trade away
    `road` and `building` to collect them. It is optimising exactly what it was
    asked for; the ask was wrong. Improving the catch-all is not the goal, so the
    fit should not be paid for it. Full mIoU is still what gets REPORTED.
    """
    v = per_class_iou(C)
    if objective == 'real':
        v = np.array([v[c] for c in range(len(v)) if c != bg])
    return float(np.nanmean(v)) if np.isfinite(v).any() else 0.0


def fit(H, bg, nbins, rounds=6, objective='all'):
    nc = H.shape[0]
    grid = np.arange(nbins + 1) / nbins

    def sc_of(t):
        return obj_miou(confusion_at(H, t, bg, nbins), bg, objective)

    best, t0 = max((sc_of(np.full(nc, t)), t) for t in grid)
    taus = np.full(nc, t0)
    for _ in range(rounds):
        moved = False
        for c in range(nc):
            if c == bg:
                continue
            sc, st = max((sc_of(np.where(np.arange(nc) == c, t, taus)), t) for t in grid)
            if sc > best + 1e-9:
                best, taus[c], moved = sc, st, True
        if not moved:
            break
    return taus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--sizes', type=int, nargs='+',
                    default=[10, 25, 50, 100, 200, 400, 800])
    ap.add_argument('--repeats', type=int, default=5,
                    help='random draws per calibration size')
    ap.add_argument('--objective', choices=['all', 'real'], default='all',
                    help="what the FIT maximises. 'real' excludes the catch-all class, "
                         "so the optimiser cannot buy mIoU by fixing an over-predicted "
                         "background at the expense of land cover. Reporting is always "
                         "full mIoU either way.")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    print(f'  classes: {LB}')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    n = len(files)
    print(f'{n} tiles | published τ = {args.tau} | {args.folds}-fold\n')
    PT = per_tile_hists(files, nc, NBINS)
    rng = np.random.default_rng(args.seed)

    def ev(idx, taus):
        return miou(confusion_at(PT[idx].sum(0).astype(np.int64), taus, bg, NBINS))

    # ---------------- k-fold
    order = rng.permutation(n)
    folds = np.array_split(order, args.folds)
    rows, gains, pc = [], [], []
    for k in range(args.folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        taus = fit(PT[tr].sum(0).astype(np.int64), bg, NBINS,
                       objective=args.objective)
        base = ev(te, np.full(nc, args.tau))
        got = ev(te, taus)
        rows.append((k + 1, len(tr), len(te), base, got, got - base))
        gains.append(got - base)
        Cb = confusion_at(PT[te].sum(0).astype(np.int64), np.full(nc, args.tau), bg, NBINS)
        Cf = confusion_at(PT[te].sum(0).astype(np.int64), taus, bg, NBINS)
        pc.append(per_class_iou(Cf) - per_class_iou(Cb))
        print(f'  fold {k + 1}: {base:.2f} -> {got:.2f}  ({got - base:+.2f})')
    gains = np.array(gains)
    pc = np.nanmean(np.array(pc), axis=0)

    md = [f'# Per-class τ — cross-validated\n',
          f'- cache: `{args.cache}`  |  tiles: **{n}**  |  classes: **{nc}**',
          f'- published τ: **{args.tau}**  |  folds: **{args.folds}**',
          f'- fit objective: **`{args.objective}`**'
          + ('  — the catch-all is excluded from what the fit maximises; reporting is '
             'still full mIoU' if args.objective == 'real' else
             '  — full mIoU, including the catch-all') + '\n',
          'Calibration and evaluation tiles are always disjoint, and the published-τ '
          'baseline is recomputed on the same held-out tiles, so both are measured on '
          'identical pixels.\n',
          '| fold | calib tiles | eval tiles | published τ | fitted | Δ |',
          '|---|---|---|---|---|---|']
    for k, ntr, nte, b, g, d in rows:
        md.append(f'| {k} | {ntr} | {nte} | {b:.2f} | **{g:.2f}** | **{d:+.2f}** |')
    md.append(f'\n**Mean Δ = {gains.mean():+.2f} mIoU, sd {gains.std(ddof=1):.2f}, '
              f'range {gains.min():+.2f} to {gains.max():+.2f}** over {args.folds} folds.\n')

    md += ['## Mean per-class Δ IoU across folds\n', '| class | Δ |', '|---|---|']
    for c in range(nc):
        if np.isfinite(pc[c]):
            md.append(f'| {LB.names[c]}{" *(catch-all)*" if c == bg else ""} | '
                      f'**{pc[c]:+.2f}** |')
    realsum = float(np.nansum([pc[c] for c in range(nc) if c != bg]))
    md.append(f'\n`{LB.names[bg]}` **{pc[bg]:+.2f}**, the {nc - 1} real classes '
              f'**{realsum:+.2f}** in aggregate.\n')

    # ---------------- learning curve
    md += ['## How many labelled tiles does calibration need?\n',
           f'Fit on *n* randomly drawn tiles, evaluate on the rest, '
           f'{args.repeats} draws each.\n',
           '| calib tiles | mean Δ | sd | worst draw |', '|---|---|---|---|']
    curve = []
    for sz in args.sizes:
        if sz >= n:
            continue
        ds = []
        for r in range(args.repeats):
            idx = rng.permutation(n)
            tr, te = idx[:sz], idx[sz:]
            taus = fit(PT[tr].sum(0).astype(np.int64), bg, NBINS,
                       objective=args.objective)
            ds.append(ev(te, taus) - ev(te, np.full(nc, args.tau)))
        ds = np.array(ds)
        curve.append((sz, ds.mean(), ds.std(ddof=1) if len(ds) > 1 else 0.0, ds.min()))
        md.append(f'| {sz} | **{ds.mean():+.2f}** | {ds.std(ddof=1) if len(ds) > 1 else 0:.2f} '
                  f'| {ds.min():+.2f} |')
        print(f'  n={sz}: {ds.mean():+.2f} ± {ds.std(ddof=1) if len(ds)>1 else 0:.2f}')

    md += ['\n## Verdict\n']
    lo = gains.mean() - 2 * gains.std(ddof=1) if len(gains) > 1 else gains.mean()
    if lo <= 0:
        md.append(f'⛔ **Not distinguishable from zero.** Mean {gains.mean():+.2f} with sd '
                  f'{gains.std(ddof=1):.2f} over {args.folds} folds, so the spread covers '
                  'no gain at all. The single-split +1.44 was a favourable draw. Report '
                  'the oracle bound only.')
    elif realsum <= 0:
        md.append(f'⚠️ **{gains.mean():+.2f} mIoU, but the real classes lose '
                  f'{realsum:+.2f}** — background unwinding again, not better land cover.')
    else:
        cheap = [c for c in curve if c[1] >= 0.6 * gains.mean() and c[3] > 0]
        note = (f' **{cheap[0][0]} labelled tiles already reach '
                f'{cheap[0][1]:+.2f}**, so the calibration cost is small.'
                if cheap else
                ' The learning curve shows the gain needs most of the split, which limits '
                'how practical this is.')
        md.append(f'✅ **{gains.mean():+.2f} ± {gains.std(ddof=1):.2f} mIoU across '
                  f'{args.folds} folds** (worst {gains.min():+.2f}), with the real classes '
                  f'gaining {realsum:+.2f}.{note}\n\n'
                  '⚠️ Calibration tiles must come from the SAME distribution as the '
                  'evaluation tiles. Fitting on LoveDA *train* and evaluating on val gives '
                  '−0.12, because those splits differ sharply (discard 14.54% vs 29.68% at '
                  'identical background share). State that limitation beside the gain — it '
                  'is the honest scope of the result.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
