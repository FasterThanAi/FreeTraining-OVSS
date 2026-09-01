"""
Every headline number in BOTH metrics: full mIoU and catch-all-excluded mIoU.

WHY THIS IS A CONTRIBUTION AND NOT BOOKKEEPING. This project has now measured the
same artefact twice, in opposite directions:

    OpenEarthMap, recovery (§8.1)   `background` +22.67, real classes −2.11,
                                    full mIoU **+2.28**   -> INFLATED
    LoveDA urban, calibration (§9e) `background` −3.51,  real classes +4.18,
                                    full mIoU **+0.10**   -> DEFLATED

Same mechanism, opposite sign, two datasets. Full mIoU moved substantially in both
cases while land-cover quality moved the other way. That is enough to make a
RECOMMENDATION rather than a complaint: an open-vocabulary benchmark with a
catch-all class should report mIoU over the real classes alongside the headline.

THE ARITHMETIC BEHIND IT, which is worth stating because it is not obvious how
large the leverage is. mIoU is an unweighted mean over N classes, so the catch-all
owns exactly 1/N of the metric no matter how meaningful that class is. On LoveDA
that is 14.3% of the score for a class the annotation guide defines as "everything
else"; on OpenEarthMap 11.1% for a class covering 0.84% of the pixels. A 22.67
point move in one class is 3.24 mIoU on a 7-class benchmark before anything real
has changed.

⚠️ Full mIoU stays the headline -- it is what the literature reports and what makes
this comparable to the baseline. The second column is reported BESIDE it, never
instead of it.

    python scripts/metric_report.py --cache ~/outputs/week3_fused/cache --tau 0.5 \\
        --md ~/outputs/week3/metric_report.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, per_class_iou, NBINS         # noqa: E402
from tau_cv import per_tile_hists, fit                            # noqa: E402


def both(C, bg):
    """(full mIoU, catch-all-excluded mIoU, catch-all IoU) from one matrix."""
    v = per_class_iou(C)
    real = [v[c] for c in range(len(v)) if c != bg]
    return (float(np.nanmean(v)) , float(np.nanmean(real)), float(v[bg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
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
    print(f'  classes: {LB}\n  {len(files)} tiles | τ = {args.tau}')

    PT = per_tile_hists(files, nc, NBINS)
    rng = np.random.default_rng(args.seed)
    pub = np.full(nc, args.tau)

    # Held-out confusion matrices, accumulated across folds, so both metrics come
    # from exactly the same pixels and the same protocol as §9b.
    Cb = np.zeros((nc, nc), np.int64)
    Cf = np.zeros((nc, nc), np.int64)
    order = rng.permutation(len(files))
    parts = np.array_split(order, args.folds)
    for k in range(args.folds):
        te, tr = parts[k], np.concatenate([parts[j] for j in range(args.folds) if j != k])
        taus = fit(PT[tr].sum(0).astype(np.int64), bg, NBINS, objective=args.objective)
        H = PT[te].sum(0).astype(np.int64)
        Cb += confusion_at(H, pub, bg, NBINS)
        Cf += confusion_at(H, taus, bg, NBINS)
        print(f'  fold {k + 1} done')

    fb, rb, gb = both(Cb, bg)
    ff, rf, gf = both(Cf, bg)
    lever = 100.0 / nc

    md = [f'# Both metrics: full mIoU and catch-all-excluded mIoU\n',
          f'- cache: `{args.cache}` | tiles: **{len(files)}** | published τ = '
          f'**{args.tau}** | {args.folds}-fold | fit objective **`{args.objective}`**',
          f'- catch-all class: **`{LB.names[bg]}`**, {nc} classes total\n',
          '⚠️ **Full mIoU stays the headline** — it is what the literature reports and what '
          'makes this comparable to the baseline. The second column is reported *beside* '
          'it, never instead of it.\n',
          f'⭐ **The leverage, stated plainly.** mIoU is an unweighted mean over {nc} '
          f'classes, so the catch-all owns exactly **{lever:.1f}%** of the metric however '
          f'meaningful that class is. A 10-point move in `{LB.names[bg]}` alone is '
          f'**{10 / nc:.2f} mIoU** before anything real has changed.\n',
          '| | full mIoU | catch-all-excluded mIoU | `' + LB.names[bg] + '` IoU |',
          '|---|---|---|---|',
          f'| published τ | {fb:.2f} | {rb:.2f} | {gb:.2f} |',
          f'| per-class τ (fitted) | **{ff:.2f}** | **{rf:.2f}** | {gf:.2f} |',
          f'| **Δ** | **{ff - fb:+.2f}** | **{rf - rb:+.2f}** | {gf - gb:+.2f} |\n']

    vb, vf = per_class_iou(Cb), per_class_iou(Cf)
    md += ['## Per class\n', '| class | published τ | fitted | Δ |', '|---|---|---|---|']
    for c in range(nc):
        md.append(f'| {LB.names[c]}{" *(catch-all)*" if c == bg else ""} | {vb[c]:.2f} | '
                  f'{vf[c]:.2f} | **{vf[c] - vb[c]:+.2f}** |')

    md += ['\n## Reading\n']
    d_full, d_real, d_bg = ff - fb, rf - rb, gf - gb
    share = abs(d_bg / nc) / abs(d_full) * 100 if abs(d_full) > 1e-6 else float('inf')
    if np.isfinite(share) and share >= 50:
        md.append(f'⛔ **{share:.0f}% of the change in full mIoU is the catch-all class '
                  f'alone** ({d_bg:+.2f} IoU, worth {d_bg / nc:+.2f} mIoU on its own). '
                  f'The headline number is not measuring land cover here, and quoting it '
                  f'without the second column would be misleading.')
    elif d_real > 0 and d_full <= 0.2 and d_bg < 0:
        md.append(f'⚠️ **Full mIoU is flat ({d_full:+.2f}) while land cover improves '
                  f'({d_real:+.2f}).** The catch-all pays for the gain ({d_bg:+.2f}), which '
                  f'is the OpenEarthMap artefact with its sign reversed. Report both.')
    elif d_real > 0 and d_full > 0:
        md.append(f'✅ **Both metrics agree**: full {d_full:+.2f}, land cover '
                  f'{d_real:+.2f}, catch-all {d_bg:+.2f}. The gain is land cover and is '
                  f'not an artefact of the catch-all being repaired.')
    else:
        md.append(f'⚠️ full {d_full:+.2f}, land cover {d_real:+.2f}, catch-all {d_bg:+.2f}. '
                  f'Report all three; the headline alone does not describe this.')
    md.append(f'\nBaseline gap between the two metrics: **{rb - fb:+.2f}** — the catch-all '
              f'sits at {gb:.2f} IoU against a real-class mean of {rb:.2f}, so it '
              + ('**depresses**' if rb > fb else '**inflates**')
              + ' the published headline by that much before any method is applied.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
