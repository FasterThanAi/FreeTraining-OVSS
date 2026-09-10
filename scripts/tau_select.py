"""
Choosing a per-class threshold when the calibration curve is FLAT or NOISY.

THE PROBLEM, measured. `tau_curves.py` reports, per class, the fitted threshold
and the one an oracle would pick on the held-out tiles. Across three datasets the
fit captures 83% (LoveDA) and 73% (Potsdam) of the oracle gain -- but only
**11% on OpenEarthMap**, and almost all of the shortfall is one class:

    OEM `water`   fitted 0.710   oracle 0.240   IoU 69.18 -> 55.02   (-14.16)

Its calibration curve peaks near 0.7; its held-out curve peaks near 0.1. With 100
calibration tiles and water unevenly distributed, the peak was noise. The current
rule takes the argmax of the calibration curve however flat or noisy that curve
is, so it followed the noise.

⭐ WHY THIS IS FIXABLE ONE CLASS AT A TIME. `separability_proof.py` shows the
objective is separable across real classes given the argmax: IoU_c depends on
tau_c alone. So a selection rule can be applied, and validated, per class without
touching any other threshold -- which would be intractable on a coupled objective.

THREE RULES, all using ONLY calibration data:

  plain   argmax of the pooled calibration curve.        (what is deployed today)
  cv      argmax of the MEAN of k per-fold curves.        Averaging folds damps a
          peak that only one fold supports.
  1se     ⭐ among thresholds whose mean curve is within one standard error of the
          best, take the one CLOSEST TO THE PUBLISHED tau. The classic
          one-standard-error rule: when the evidence cannot separate candidates,
          prefer the least movement from the prior.

⚠️ `1se` is recommended on PRINCIPLE, not because it wins here. Picking the rule
that scores best on the held-out tiles would be selection on the test set --
exactly the error this script exists to fix, one level up. All three are reported.

    python scripts/tau_select.py --cache ~/outputs/oem_tau0.1/cache --tau 0.1 --calib 100
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                             # noqa: E402
from tau_oracle import NBINS                              # noqa: E402
from tau_cv import per_tile_hists                         # noqa: E402


def iou_curve(H, c):
    """IoU of class c at every grid threshold, from a (gt, pred, bin) histogram.

    Valid because the objective separates: a pixel predicted c either clears
    tau_c or becomes the catch-all, so TP/FP/FN for c depend on tau_c alone.

        TP(t) = sum of H[c, c, bins >= t]
        FP(t) = sum of H[g != c, c, bins >= t]
        FN(t) = (all pixels with gt = c) - TP(t)
    """
    nb = H.shape[2]
    col = H[:, c, :]                                  # (gt, bin) for pred == c
    tail = np.concatenate([col[:, ::-1].cumsum(1)[:, ::-1],
                           np.zeros((col.shape[0], 1), col.dtype)], axis=1)
    tp = tail[c].astype(np.float64)                   # kept and correct
    fp = tail.sum(0).astype(np.float64) - tp          # kept and wrong
    fn = float(H[c].sum()) - tp                       # gt = c but not kept as c
    with np.errstate(invalid='ignore', divide='ignore'):
        return 100.0 * tp / (tp + fp + fn)            # length nb + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True, help='the PUBLISHED tau')
    ap.add_argument('--calib', type=int, default=200)
    ap.add_argument('--folds', type=int, default=5, help='INNER folds, within calibration')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--md')
    args = ap.parse_args()

    L = labels.from_cache(args.cache)
    nc, bg = L.n, L.bg - 1
    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if not files:
        raise SystemExit(f'⛔ no .npz in {args.cache}')
    PT = per_tile_hists(files, nc, NBINS)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(files))
    cal, held = idx[:args.calib], idx[args.calib:]
    print(f'\ncalibration {len(cal)} tiles ({args.folds} inner folds) | '
          f'held out {len(held)} tiles')

    grid = np.arange(NBINS + 1) / NBINS
    pub_i = int(round(args.tau * NBINS))
    Hcal, Hheld = PT[cal].sum(0).astype(np.int64), PT[held].sum(0).astype(np.int64)
    inner = np.array_split(rng.permutation(cal), args.folds)

    rows, tot = [], {k: [] for k in ('pub', 'plain', 'cv', 'onese', 'oracle')}
    for c in range(nc):
        if c == bg:
            continue
        pooled = iou_curve(Hcal, c)
        folds = np.array([iou_curve(PT[f].sum(0).astype(np.int64), c) for f in inner])
        with np.errstate(invalid='ignore'):
            mu = np.nanmean(folds, 0)
            # ⚠️ np.maximum, not the builtin max: the count is one PER GRID POINT,
            # and a class absent from a fold contributes nan there rather than 0.
            n_ok = np.maximum(np.isfinite(folds).sum(0), 1)
            se = np.nanstd(folds, 0, ddof=1) / np.sqrt(n_ok)
        heldc = iou_curve(Hheld, c)

        i_plain = int(np.nanargmax(pooled))
        i_cv = int(np.nanargmax(mu))
        # ⭐ one standard error: any threshold statistically tied with the best is
        # admissible; among those take the least movement from the published tau.
        bar = mu[i_cv] - se[i_cv]
        adm = np.where(np.nan_to_num(mu, nan=-1e9) >= bar)[0]
        i_1se = int(adm[np.argmin(np.abs(adm - pub_i))]) if len(adm) else i_cv
        i_or = int(np.nanargmax(heldc))

        r = dict(name=L.names[c],
                 t_plain=grid[i_plain], t_cv=grid[i_cv], t_1se=grid[i_1se],
                 t_or=grid[i_or],
                 pub=heldc[pub_i], plain=heldc[i_plain], cv=heldc[i_cv],
                 onese=heldc[i_1se], oracle=heldc[i_or])
        rows.append(r)
        for k in tot:
            tot[k].append(r[k])

    hdr = ('| class | τ plain | τ cv | τ 1se | τ oracle | IoU pub | plain | cv '
           '| 1se | oracle |')
    out = ['# Threshold selection: plain argmax vs inner-CV vs one-standard-error',
           '',
           f'Published τ = {args.tau}. Fitted on {len(cal)} calibration tiles '
           f'({args.folds} inner folds); every IoU below is on the {len(held)} '
           f'held-out tiles.', '', hdr, '|---' * 10 + '|']
    for r in rows:
        out.append(f"| `{r['name']}` | {r['t_plain']:.3f} | {r['t_cv']:.3f} "
                   f"| {r['t_1se']:.3f} | {r['t_or']:.3f} "
                   f"| {r['pub']:.2f} | {r['plain']:.2f} | {r['cv']:.2f} "
                   f"| **{r['onese']:.2f}** | {r['oracle']:.2f} |")
    m = {k: float(np.nanmean(v)) for k, v in tot.items()}
    out += ['', '| rule | real-class mIoU | Δ vs published | share of oracle |',
            '|---|---|---|---|']
    span = m['oracle'] - m['pub']
    for k, nm in (('pub', 'published τ'), ('plain', 'plain argmax *(deployed today)*'),
                  ('cv', 'inner-CV mean'), ('onese', '⭐ one standard error'),
                  ('oracle', 'oracle *(bound)*')):
        d = m[k] - m['pub']
        sh = '—' if k in ('pub', 'oracle') or abs(span) < 1e-9 else f'{100 * d / span:.0f}%'
        out.append(f"| {nm} | {m[k]:.2f} | {d:+.2f} | {sh} |")
    out += ['', '⚠️ The rule is chosen on PRINCIPLE, not on this table. Selecting '
            'whichever column scores best on the held-out tiles would be selection '
            'on the evaluation set — the same error, one level up. All three are '
            'reported so the reader can see the spread.']

    txt = '\n'.join(out)
    print('\n' + txt)
    if args.md:
        p = Path(args.md).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt + '\n'); print(f'\n  wrote {p}')


if __name__ == '__main__':
    main()
