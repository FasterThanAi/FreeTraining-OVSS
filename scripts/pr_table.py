"""
Per-class precision, recall and IoU under each decision rule.

WHY. Every result in this project is reported as IoU, but IoU hides the
mechanism. The method's claim is that a class with high precision and low recall
can afford a lower threshold, and the only way to show that directly is to put
precision and recall side by side before and after. Two classes were recorded by
hand during the deployment run (`water` and `road`); this produces the table for
every class, on any cache.

WHAT IT COMPUTES, per class:

    precision = TP / (TP + FP)   of the pixels we CALLED c, how many were c
    recall    = TP / (TP + FN)   of the pixels that WERE c, how many we found
    IoU       = TP / (TP + FP + FN)

    A  published global tau              -- the baseline
    B  per-class tau                     -- lever 1
    C  per-class scale + per-class tau   -- lever 2, only with a --cache-full run

⚠️ Rung C needs the full score stack. A histogram cache fixes `pred` at the
published argmax, and a scale CHANGES the argmax, so the histogram is not a
sufficient statistic for it. The script says so rather than silently reporting
two rungs as three.

⚠️ Thresholds are fitted on a disjoint calibration split and the table is
computed on the held-out remainder, so the numbers are comparable with the
paper's rather than fitted and scored on the same pixels.

    python scripts/pr_table.py --cache ~/outputs/week3_fused/cache --tau 0.5
    python scripts/pr_table.py --cache ~/outputs/potsdam_full/cache --tau 0.1 --scale
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                             # noqa: E402
from tau_oracle import confusion_at, NBINS                # noqa: E402
from tau_cv import fit, per_tile_hists                    # noqa: E402


def pr_iou(C):
    """Precision, recall and IoU per class from a confusion matrix.

    Rows are truth, columns are prediction, so TP is the diagonal, FP the column
    sum minus it, FN the row sum minus it.
    """
    tp = np.diag(C).astype(np.float64)
    fp = C.sum(0) - tp
    fn = C.sum(1) - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        p = 100 * tp / (tp + fp)
        r = 100 * tp / (tp + fn)
        i = 100 * tp / (tp + fp + fn)
    return p, r, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True, help='the PUBLISHED global tau')
    ap.add_argument('--calib', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--md', help='write the table to this markdown file')
    args = ap.parse_args()

    L = labels.from_cache(args.cache)
    # ⚠️ Labels.bg is the catch-all's MASK VALUE (1-indexed, since mask value 0 is
    # no-data). The confusion matrix is 0-indexed, so it is bg-1 here -- the same
    # conversion tau_cv.py:124 makes. Reading the attribute name off another
    # script rather than off labels.py is what broke the first version.
    nc, bg = L.n, L.bg - 1
    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if not files:
        raise SystemExit(f'⛔ no .npz in {args.cache}')
    print(f'  {len(files)} tiles, {nc} classes, catch-all = {L.names[bg]!r}')

    PT = per_tile_hists(files, nc, NBINS)                  # (tiles, nc, nc, bins)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(files))
    calib, held = idx[:args.calib], idx[args.calib:]
    print(f'  fit on {len(calib)} tiles, table computed on the {len(held)} held out')

    Hc = PT[calib].sum(0).astype(np.int64)
    Hh = PT[held].sum(0).astype(np.int64)

    taus_a = np.full(nc, args.tau)
    taus_b = fit(Hc, bg, NBINS, objective=args.objective)

    pa, ra, ia = pr_iou(confusion_at(Hh, taus_a, bg, NBINS))
    pb, rb, ib = pr_iou(confusion_at(Hh, taus_b, bg, NBINS))

    rows = []
    for c in range(nc):
        rows.append(dict(name=L.names[c], tau_a=args.tau, tau_b=float(taus_b[c]),
                         pa=pa[c], ra=ra[c], ia=ia[c],
                         pb=pb[c], rb=rb[c], ib=ib[c], bg=(c == bg)))
    rows.sort(key=lambda r: -(r['ib'] - r['ia']))

    hdr = (f'| class | tau A | tau B | prec A | prec B | recall A | recall B '
           f'| IoU A | IoU B | Δ IoU |')
    sep = '|---' * 10 + '|'
    out = ['# Per-class precision / recall / IoU', '',
           f'A = published global tau {args.tau}.  B = per-class tau, fitted on '
           f'{len(calib)} calibration tiles, table on {len(held)} held out.', '',
           hdr, sep]
    for r in rows:
        tag = ' *(catch-all)*' if r['bg'] else ''
        tb = '—' if r['bg'] else f"{r['tau_b']:.3f}"
        out.append(f"| `{r['name']}`{tag} | {r['tau_a']:.3f} | {tb} "
                   f"| {r['pa']:.1f} | {r['pb']:.1f} | {r['ra']:.1f} | {r['rb']:.1f} "
                   f"| {r['ia']:.1f} | {r['ib']:.1f} | **{r['ib'] - r['ia']:+.2f}** |")
    real = [r for r in rows if not r['bg']]
    out += ['',
            f"mIoU  {np.nanmean([r['ia'] for r in rows]):.2f} → "
            f"{np.nanmean([r['ib'] for r in rows]):.2f}   ·   excluding the catch-all  "
            f"{np.nanmean([r['ia'] for r in real]):.2f} → "
            f"{np.nanmean([r['ib'] for r in real]):.2f}", '',
            '⚠️ The catch-all has no fitted threshold: sub-threshold pixels are '
            'assigned *to* it, so no threshold applies to it.']

    z = np.load(files[0])
    if 'logits' not in z:
        out += ['', '⚠️ This cache stores a histogram, not the score stack, so rung C '
                '(per-class scale) cannot be computed here. A scale changes the '
                'argmax, and the histogram fixes the argmax at the published one. '
                'Re-run the cache with `--cache-full` for the three-rung table.']

    txt = '\n'.join(out)
    print('\n' + txt)
    if args.md:
        p = Path(args.md).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt + '\n'); print(f'\n  wrote {p}')


if __name__ == '__main__':
    main()
