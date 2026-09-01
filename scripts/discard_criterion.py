"""
When is per-class calibration worth the labelling? A label-free rule. ROADMAP §6.2.

THE PROBLEM. §9b's calibration gain is +1.18 on LoveDA, but §9e showed it is really
+2.77 on rural and +0.10 on urban. A practitioner cannot act on that: "it works on
rural imagery" is not a rule, and they do not know which stratum they are in until
they have already paid for the labels.

Those strata differ 2x in DISCARD RATE (39.3% vs 18.5%), so domain and residual
size are confounded there in exactly the way share and confusability were before
§7a broke them. This asks which one the gain actually tracks.

⭐ WHY IT MATTERS THAT IT IS THE DISCARD RATE. The fraction of pixels a model
assigns to the catch-all is computable FROM THE PREDICTIONS ALONE -- no ground
truth, no annotation, nothing but a forward pass over unlabelled tiles. If the
calibration gain tracks it, the paper ends with a rule that can be applied BEFORE
spending money:

    measure your catch-all fraction; above X, per-class calibration repays
    ~200 labelled tiles; below it, do not bother.

That converts §9e's scope limitation into a deployment criterion, and makes the
result predictive rather than descriptive.

⚠️ THE CONTROL IS THE EXPERIMENT, AGAIN. Strata defined by ANY per-tile statistic
are more internally homogeneous than the whole split, and a homogeneous stratum
may calibrate better for that reason alone -- nothing to do with the discard rate.
So every run also splits the SAME tiles into random strata OF THE SAME SIZES and
fits identically. A gradient across discard strata means something only if the
random strata do not show one.

⚠️ The statistic is deliberately the LABEL-FREE one: the fraction of ALL pixels
assigned to the catch-all. The familiar 29.68% is a different quantity -- it needs
ground truth to know which pixels had a real class. The two are reported side by
side so the substitution is visible rather than assumed.

    python scripts/discard_criterion.py --cache ~/outputs/week3_fused/cache \\
        --tau 0.5 --strata 4 --map ~/splits/loveda_domain.txt \\
        --md ~/outputs/week3/discard_criterion.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, miou, per_class_iou, NBINS   # noqa: E402
from tau_cv import fit                                            # noqa: E402


def scan(files, nc, bg, tau, nbins):
    """Per-tile histogram, plus the label-free statistic and its labelled twin."""
    H = np.zeros((len(files), nc, nc, nbins), np.int32)
    free = np.zeros(len(files))          # catch-all fraction, PREDICTIONS ONLY
    lab = np.zeros(len(files))           # real-class pixels sent to catch-all (needs GT)
    for i, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.int32)
        conf = z['conf'].astype(np.float32)
        pred = np.clip(z['pred'].astype(np.int32), 0, nc - 1)
        final = np.where(conf < tau, bg, pred)
        free[i] = float((final == bg).mean())          # no ground truth used
        m = gt > 0
        real = m & (gt - 1 != bg)
        lab[i] = float((real & (final == bg)).sum() / max(real.sum(), 1))
        if m.any():
            b = np.clip((conf[m] * nbins).astype(np.int32), 0, nbins - 1)
            h = np.zeros((nc, nc, nbins), np.int64)
            np.add.at(h, (gt[m] - 1, pred[m], b), 1)
            H[i] = h
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}')
    return H, free, lab


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float('nan')


def cv(PT, idx, nc, bg, tau, folds, objective, rng):
    """k-fold within one stratum. Calibration and evaluation always disjoint, and
    the published-tau baseline recomputed on the same held-out tiles."""
    if len(idx) < folds * 2:
        return np.array([np.nan])
    order = rng.permutation(len(idx))
    parts = np.array_split(order, folds)
    out = []
    for k in range(folds):
        te = idx[parts[k]]
        tr = idx[np.concatenate([parts[j] for j in range(folds) if j != k])]
        taus = fit(PT[tr].sum(0).astype(np.int64), bg, NBINS, objective=objective)
        Hte = PT[te].sum(0).astype(np.int64)
        out.append(miou(confusion_at(Hte, taus, bg, NBINS))
                   - miou(confusion_at(Hte, np.full(nc, tau), bg, NBINS)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--strata', type=int, default=4)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--map', default=None,
                    help='optional tile\\tdomain map; reports how much of the '
                         'stratification is just the domain in disguise')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')
    print(f'  classes: {LB}\n  {len(files)} tiles | τ = {args.tau} | '
          f'{args.strata} strata | {args.folds}-fold')

    PT, free, lab = scan(files, nc, bg, args.tau, NBINS)
    rho = spearman(free, lab)
    print(f'\n  label-free catch-all fraction: mean {free.mean():.3f}, '
          f'median {np.median(free):.3f}')
    print(f'  ρ(label-free, label-based discard) = {rho:+.3f}')

    rng = np.random.default_rng(args.seed)
    order = np.argsort(free)
    strata = np.array_split(order, args.strata)          # equal-count quantiles
    rnd = np.array_split(rng.permutation(len(files)), args.strata)   # SAME sizes

    dom = None
    if args.map:
        m = {}
        for line in Path(args.map).expanduser().read_text().splitlines():
            if line.strip():
                k, _, v = line.partition('\t')
                m[k.strip()] = v.strip().lower()
        dom = [m.get(f.stem, '?') for f in files]

    rows, ctrl = [], []
    for j, ix in enumerate(strata):
        g = cv(PT, ix, nc, bg, args.tau, args.folds, args.objective, rng)
        rows.append((j, ix, g))
        print(f'  stratum {j + 1}: catch-all {free[ix].min():.3f}–{free[ix].max():.3f} '
              f'| Δ {np.nanmean(g):+.2f}')
    for j, ix in enumerate(rnd):
        ctrl.append((j, ix, cv(PT, ix, nc, bg, args.tau, args.folds,
                               args.objective, rng)))

    md = [f'# When does per-class calibration pay? A label-free criterion\n',
          f'- cache: `{args.cache}` | tiles: **{len(files)}** | τ = **{args.tau}** | '
          f'{args.strata} equal-count strata | {args.folds}-fold | objective '
          f'**`{args.objective}`**\n',
          '⭐ **The stratifying statistic uses no ground truth**: it is the fraction of '
          'all pixels the model assigns to the catch-all, computable from a forward pass '
          'over unlabelled tiles. The familiar discard rate is a different quantity — it '
          'needs labels to know which pixels held a real class. Both are shown so the '
          f'substitution is visible: **Spearman ρ = {rho:+.3f}** between them.\n',
          '⚠️ **The random strata are the control.** Any per-tile statistic makes strata '
          'more internally homogeneous, and a homogeneous stratum may calibrate better for '
          'that reason alone. The control splits the same tiles into random strata of the '
          '**same sizes** and fits identically, so a gradient means something only if the '
          'control does not show one.\n',
          '| stratum | tiles | catch-all fraction | labelled discard | **Δ mIoU** | sd | '
          'worst fold |' + (' domain mix |' if dom else ''),
          '|---|---|---|---|---|---|---|' + ('---|' if dom else '')]
    for j, ix, g in rows:
        mix = ''
        if dom:
            c = {}
            for i in ix:
                c[dom[i]] = c.get(dom[i], 0) + 1
            mix = ' ' + ', '.join(f'{k} {v * 100 // len(ix)}%'
                                  for k, v in sorted(c.items())) + ' |'
        md.append(f'| {j + 1} | {len(ix)} | {free[ix].min():.3f}–{free[ix].max():.3f} | '
                  f'{lab[ix].mean() * 100:.1f}% | **{np.nanmean(g):+.2f}** | '
                  f'{np.nanstd(g, ddof=1):.2f} | {np.nanmin(g):+.2f} |' + mix)

    md += ['\n## Control — random strata of identical sizes\n',
           '| stratum | tiles | **Δ mIoU** | sd |', '|---|---|---|---|']
    for j, ix, g in ctrl:
        md.append(f'| {j + 1} | {len(ix)} | **{np.nanmean(g):+.2f}** | '
                  f'{np.nanstd(g, ddof=1):.2f} |')

    gains = np.array([np.nanmean(g) for _, _, g in rows])
    cgains = np.array([np.nanmean(g) for _, _, g in ctrl])
    spread, cspread = gains.max() - gains.min(), cgains.max() - cgains.min()
    mids = np.array([free[ix].mean() for _, ix, _ in rows])
    rho_g = spearman(mids, gains)

    md += ['\n## Verdict\n',
           f'| | spread across strata | ρ(catch-all fraction, Δ mIoU) |',
           '|---|---|---|',
           f'| **discard-stratified** | **{spread:.2f}** | **{rho_g:+.3f}** |',
           f'| random control | {cspread:.2f} | — |', '']
    if spread > 2 * cspread and rho_g > 0.5:
        # A cutoff is only a RULE if some stratum below it fails. When every
        # stratum is positive the "cutoff" is just the dataset's minimum, which
        # says nothing -- an earlier version reported exactly that.
        ok = [np.nanmin(g) > 0 for _, _, g in rows]
        first = next((j for j in range(len(ok)) if all(ok[j:])), None)
        md.append(f'✅ **The gain tracks the label-free catch-all fraction.** It spans '
                  f'{spread:.2f} mIoU across the strata against {cspread:.2f} for random '
                  f'strata of the same sizes, with ρ = {rho_g:+.3f}.')
        if first is None:
            md.append('\n⚠️ No suffix of the strata is uniformly positive, so the gradient '
                      'is real but no cutoff can be stated. Report the table.')
        elif first == 0:
            md.append(f'\n⚠️ **Every stratum is positive, so no cutoff is identified '
                      f'within the observed range** ({free.min():.3f}–{free.max():.3f}). '
                      f'The gradient gives the *size* of the expected gain, not a '
                      f'go/no-go rule; a dataset below {free.min():.3f} would be needed to '
                      f'find the floor — OpenEarthMap, at a far lower catch-all fraction, '
                      f'is the natural test.')
        else:
            md.append(f'\n⭐ **Deployment rule: calibrate when the catch-all fraction '
                      f'exceeds {free[rows[first][1]].min():.3f}.** At and above it every '
                      f'fold of every stratum is positive; below it stratum '
                      f'{first} gives {np.nanmean(rows[first - 1][2]):+.2f} with a worst '
                      f'fold of {np.nanmin(rows[first - 1][2]):+.2f}. **The quantity needs '
                      f'no labels**, so it can be checked before any annotation is '
                      f'commissioned.')
    elif spread <= 2 * cspread:
        md.append(f'⚠️ **Inconclusive — the control moves nearly as much.** Discard strata '
                  f'span {spread:.2f} against {cspread:.2f} for random strata of the same '
                  f'sizes. Splitting the data at all produces most of this variation, so '
                  f'the catch-all fraction is not shown to be what matters. Report the '
                  f'per-stratum table and no rule.')
    else:
        md.append(f'⛔ **The gain does not track the catch-all fraction** (ρ = {rho_g:+.3f} '
                  f'over {args.strata} strata). §9e\'s rural/urban difference is therefore '
                  f'not explained by residual size, and no label-free deployment rule '
                  f'follows from this statistic.')
    if dom:
        md.append('\n⚠️ Check the domain-mix column. If the strata are simply urban and '
                  'rural re-labelled, this restates §9e rather than adding to it — the '
                  'claim is only new insofar as the strata cut across the domains.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
