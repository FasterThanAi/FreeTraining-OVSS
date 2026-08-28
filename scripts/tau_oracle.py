"""
How much is threshold tuning worth, at its absolute best?

THE QUESTION A REVIEWER WILL ASK. Every result so far uses ONE global tau -- 0.5
on LoveDA, 0.1 on OpenEarthMap, each dataset's published value. "Did you just
tune the threshold badly?" is the cheapest possible objection to this whole line
of work, and it has never been answered.

It should be answered generously: give threshold tuning ORACLE access to the test
labels and let it pick the best value it possibly could. If even that is worth
little, the objection is closed for good. If it is worth a lot, the problem is
calibration rather than recovery, and this project has been aiming at the wrong
target.

Three rungs, each strictly more powerful than the last:

    published tau     what the baseline ships with
    best GLOBAL tau   one threshold, chosen with GT. Oracle over 1 parameter.
    best PER-CLASS tau  one threshold per predicted class, chosen with GT.
                        Oracle over N parameters.

The per-class rung is the interesting one and it has never been tested. The
per-class precision/recall gaps are large -- water +34.8 on LoveDA, pavement
+36.8 on OpenEarthMap -- which is exactly the signature of a single threshold
being wrong for individual classes in opposite directions.

⚠️ BOTH SWEPT ROWS ARE ORACLE BOUNDS, not methods. They select hyperparameters on
the evaluation set. They are reported to bound what threshold tuning could ever
achieve, and must never be quoted as a result.

WHY THIS NEEDS NO GPU. `pred` is an argmax and does not depend on tau; only the
"is conf below the threshold" test does, and `conf` is in the cache. So a
histogram over (gt, pred, conf-bin) -- 9x9x200 for OpenEarthMap, a few hundred
kilobytes -- is a sufficient statistic for the confusion matrix at ANY tau
vector. One pass over the cache, then every threshold evaluated in microseconds.

    python scripts/tau_oracle.py --cache ~/outputs/week3_fused/cache --tau 0.5
    python scripts/tau_oracle.py --cache ~/outputs/oem_tau0.1/cache --tau 0.1
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

NBINS = 200          # 0.005 resolution over [0,1]


def build_hist(files, nc, nbins, tau_note):
    """H[gt, pred, conf_bin] over every labelled pixel. The sufficient statistic.

    gt and pred are 0-indexed class ids here (mask value - 1).
    """
    H = np.zeros((nc, nc, nbins), np.int64)
    for i, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.int32)
        conf = z['conf'].astype(np.float32)     # float16 in the cache -> f32
        pred = z['pred'].astype(np.int32)       # already 0-indexed
        m = gt > 0                              # drop no-data
        if not m.any():
            continue
        g = gt[m] - 1
        p = np.clip(pred[m], 0, nc - 1)
        b = np.clip((conf[m] * nbins).astype(np.int32), 0, nbins - 1)
        np.add.at(H, (g, p, b), 1)
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}')
    return H


def confusion_at(H, taus, bg, nbins):
    """Confusion matrix for a per-class threshold vector.

    The rule the segmentor applies is: keep `pred` if conf >= tau[pred],
    otherwise assign background. So for a non-background predicted class, the
    pixels that survive are those in bins at or above its threshold; the rest
    fall into the background column. Pixels already predicted background stay
    there whatever their confidence, which is why tau[bg] has no effect -- it is
    excluded from the search rather than silently swept.
    """
    nc = H.shape[0]
    C = np.zeros((nc, nc), np.int64)
    edges = np.clip((np.asarray(taus) * nbins).astype(int), 0, nbins)
    for p in range(nc):
        cs = H[:, p, :].sum(1)                       # all bins, per gt
        if p == bg:
            C[:, bg] += cs
            continue
        keep = H[:, p, edges[p]:].sum(1)             # conf >= tau_p
        C[:, p] += keep
        C[:, bg] += cs - keep                        # the rest -> background
    return C


def per_class_iou(C):
    nc = C.shape[0]
    out = np.full(nc, np.nan)
    for k in range(nc):
        tp = C[k, k]
        den = C[k].sum() + C[:, k].sum() - tp
        if den > 0:
            out[k] = 100.0 * tp / den
    return out


def miou(C):
    v = per_class_iou(C)
    return float(np.nanmean(v)) if np.isfinite(v).any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True,
                    help="the dataset's published tau, for the baseline row")
    ap.add_argument('--rounds', type=int, default=6,
                    help='coordinate-ascent passes for the per-class search')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1        # bg as a 0-indexed class id
    print(f'  classes: {LB}')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    print(f'{len(files)} tiles | published τ = {args.tau} | {NBINS} bins\n')
    H = build_hist(files, nc, NBINS, args.tau)

    grid = np.arange(NBINS + 1) / NBINS

    # ---- rung 1: the published threshold
    base_t = np.full(nc, args.tau)
    C0 = confusion_at(H, base_t, bg, NBINS)
    m0 = miou(C0)

    # ---- rung 2: best single global threshold, chosen with GT
    gl = [(miou(confusion_at(H, np.full(nc, t), bg, NBINS)), t) for t in grid]
    m1, t1 = max(gl)

    # ---- rung 3: best per-class thresholds, coordinate ascent from rung 2
    taus = np.full(nc, t1)
    best = m1
    for r in range(args.rounds):
        moved = False
        for c in range(nc):
            if c == bg:
                continue            # tau[bg] cannot change any assignment
            cur = taus[c]
            cand = [(miou(confusion_at(H, np.where(np.arange(nc) == c, t, taus),
                                       bg, NBINS)), t) for t in grid]
            sc, st = max(cand)
            if sc > best + 1e-9:
                best, taus[c], moved = sc, st, True
            else:
                taus[c] = cur
        print(f'  round {r + 1}: mIoU {best:.2f}')
        if not moved:
            break
    C2 = confusion_at(H, taus, bg, NBINS)
    m2 = miou(C2)

    v0, v2 = per_class_iou(C0), per_class_iou(C2)
    d = v2 - v0
    real = [k for k in range(nc) if k != bg]
    bg_gain = d[bg] if np.isfinite(d[bg]) else 0.0
    real_gain = float(np.nansum(d[real]))

    md = ['# Oracle bound on threshold tuning\n',
          f'- cache: `{args.cache}`  |  tiles: **{len(files)}**  |  '
          f'classes: **{nc}**  |  bins: {NBINS}',
          f'- published τ: **{args.tau}**\n',
          '## Cross-check\n',
          f'The published-τ row is computed here from the confidence histogram, by a '
          f'different code path than `measure_discard_rate.py` and '
          f'`selective_recovery_miou.py`. It must agree with them: **{m0:.2f}** against '
          f'47.37 on the full LoveDA cache, 44.16 on OpenEarthMap. If it does not, the '
          f'histogram or the label convention is wrong and every row below is void.\n',
          '⚠️ **The two swept rows are ORACLE BOUNDS.** They choose thresholds using '
          'the evaluation labels, so they bound what threshold tuning could ever '
          'achieve. They are not methods and must not be quoted as results.\n',
          '| rung | free parameters | **mIoU** | Δ vs published |',
          '|---|---|---|---|',
          f'| published τ = {args.tau} | 0 | **{m0:.2f}** | — |',
          f'| best global τ = {t1:.3f} | 1 | **{m1:.2f}** | {m1 - m0:+.2f} |',
          f'| best per-class τ | {nc - 1} | **{m2:.2f}** | **{m2 - m0:+.2f}** |\n',
          '## Chosen per-class thresholds\n',
          '| class | τ | IoU before | after | Δ |', '|---|---|---|---|---|']
    present = H.sum(axis=(1, 2))          # GT pixels per class
    for k in range(nc):
        if present[k] == 0:
            md.append(f'| {LB.names[k]} | — | *absent from this split* | | |')
            continue
        t = '—  *(no effect)*' if k == bg else f'{taus[k]:.3f}'
        md.append(f'| {LB.names[k]} | {t} | {v0[k]:.2f} | {v2[k]:.2f} | '
                  f'**{d[k]:+.2f}** |')

    md += [f'\n`background` **{bg_gain:+.2f}**, the {nc - 1} real classes '
           f'**{real_gain:+.2f}** in aggregate.\n']
    if m2 - m0 > 0.5 and bg_gain > 0.8 * (bg_gain + real_gain):
        md.append('> ⚠️ **The gain is mostly the background row again.** Same pattern as '
                  'the recovery experiments: mIoU rises because one over-predicted class '
                  'is corrected, not because land cover is classified better. Report the '
                  'per-class column beside the headline.\n')

    md += ['## Verdict\n']
    gain = m2 - m0
    if gain < 0.5:
        md.append(f'✅ **Threshold tuning is worth at most {gain:+.2f} mIoU**, even with '
                  f'oracle access to the labels and a free parameter per class. The '
                  '"you just tuned τ badly" objection is closed: the published value is '
                  'within half a point of the best any threshold rule could do. This is '
                  'a strong row for the paper because it bounds an entire family of '
                  'trivial alternatives in one number.')
    elif gain < 2.0:
        md.append(f'⚠️ **Threshold tuning is worth {gain:+.2f} mIoU at the oracle bound.** '
                  'Real but modest, and unreachable without labels. Report it as the '
                  'ceiling on threshold tuning and note that a practical per-class rule '
                  'would capture only part of it.')
    else:
        md.append(f'⛔ **Threshold tuning is worth {gain:+.2f} mIoU at the oracle bound** '
                  f'— more than the recovery machinery achieves. That reframes the '
                  'problem: a substantial share of the residual is a CALIBRATION failure, '
                  'a single global τ being wrong for individual classes in opposite '
                  'directions, rather than a recovery failure. Before anything else, '
                  'find out how much of this bound a label-free rule can reach — '
                  'per-class τ set from the presence score, or from each class\'s own '
                  'confidence distribution, needs no ground truth and is worth testing '
                  'immediately.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
