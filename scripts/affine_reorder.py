"""
LEVER 5 -- an affine reordering, w_c * s_c + b_c, instead of a multiplicative one.

⛔ WHY THIS EXISTS. After lever 4's null I wrote that the decision side was
"exhausted by construction": per-class tau is provably complete after a fixed
argmax, and the only thing left at the argmax is a GENERAL reordering, unbounded
in parameters against 200 tiles. That skipped a bounded family in the middle,
and it is the one the calibration literature calls the next step --
temperature -> VECTOR scaling -> matrix scaling (Guo 2017, Kull 2019). We fitted
the diagonal case and skipped vector scaling.

THE RULE:

    pred = argmax_c ( w_c * s_c + b_c )       <- b acts HERE, and only here
    keep pred if s_pred >= tau_pred           <- the threshold reads the RAW score

⭐ WHY THE BIAS IS NOT JUST ANOTHER SCALE. Where scores are large, w*s dominates
and b is irrelevant. At LOW-confidence pixels -- the discarded residual, by
definition -- b decides who wins, and a multiplier cannot reach there: scaling a
small number leaves it small. So b buys reach in exactly the regime the method
exists to fix.

⚠️ AND THE COUNTER-ARGUMENT, stated before the run: tau already governs the
low-confidence regime per class. b may re-parameterise what tau does, in which
case refitting tau on top absorbs it -- the fate of levers 3 and 4.

⭐ b IS CONFINED TO THE REORDERING, like w. The threshold reads the raw score, so
b cannot masquerade as a threshold change. Shifting the score tau sees would just
reparameterise tau, which is the completeness argument.

FOUR RUNGS:

    A  published tau, w = 1,    b = 0     the baseline
    B  fitted tau,    w = 1,    b = 0     lever 1
    C  fitted tau,    fitted w, b = 0     lever 1 + 2, the deployed method
    D  fitted tau,    fitted w, fitted b  this experiment

⭐ The result is D - C.

Predictions committed in prereg/predict_affine.md before this was run.

    python scripts/affine_reorder.py --cache ~/outputs/loveda_full_all/cache \\
        --tau 0.5 --objective real --md ~/outputs/week4/affine_loveda.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, miou, NBINS                  # noqa: E402
from tau_cv import fit as fit_tau, obj_miou                       # noqa: E402

W_GRID = np.round(np.exp(np.linspace(np.log(0.40), np.log(2.50), 11)), 3)
# ⛔ Sweep w from 1.0 OUTWARD, not from the grid's low end. With strict `>` the
# incumbent keeps every tie, and every uniform scaling of w gives an identical
# argmax -- so a low-end-first sweep walks w to a uniform 0.40 purely by
# tie-breaking. That is argmax-equivalent to 1.0 and therefore harmless on its
# own, but it is NOT harmless once a bias is added (see normalise() below).
W_ORDER = np.argsort(np.abs(np.log(W_GRID)), kind='stable')
# b grid: scores live in [0,1], so +-0.25 is a large shift. 0.0 is a grid point
# and is swept FIRST, so the baseline wins every tie and a class whose bias is
# irrelevant keeps b = 0 instead of drifting to an endpoint -- the bug that made
# lever 4 report a confident rho = 0.25 on classes where nothing mattered.
B_GRID = np.round(np.linspace(-0.25, 0.25, 11), 3)
B_ORDER = np.argsort(np.abs(B_GRID), kind='stable')


def normalise(w):
    """Rescale w to geometric mean 1. ⛔ NOT cosmetic -- it is required for the
    bias to be identifiable.

    Only RATIOS of w affect an argmax, so (w, b) carries a redundant degree of
    freedom: scaling every w by k and every b by k leaves the decision unchanged.
    That means the MEANING of a fixed b grid depends on w's overall magnitude. On
    the unit test the w search settled on a uniform 0.40 -- identical argmax, but
    against those shrunken scores a b of +0.05 is over six times larger in
    relative terms, so it overshot and stole a class it should not have. The b
    search then correctly found that every grid point was worse than 0 and
    returned a null, at a point where the true answer scores 100.

    Fixing w's gauge before fitting b removes the redundancy.
    """
    return w / float(np.exp(np.mean(np.log(w))))


def load(files, nsub, nc, rng):
    S, G = [], []
    for i, f in enumerate(files):
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}', flush=True)
        z = np.load(f)
        if 'logits' not in z.files:
            raise SystemExit(
                f'{f.name} has no `logits`. A bias changes the argmax, so a '
                f'histogram cache is not a sufficient statistic. Re-cache with '
                f'--cache-full.')
        gt = z['gt'].ravel()
        keep = np.flatnonzero(gt > 0)
        if keep.size == 0:
            continue
        if keep.size > nsub:
            keep = rng.choice(keep, size=nsub, replace=False)
        S.append(z['logits'].astype(np.float32).reshape(nc, -1)[:, keep])
        G.append(gt[keep].astype(np.int32) - 1)
    return S, G


def hists(S, G, idx, w, b, nc, nbins):
    H = np.zeros((nc, nc, nbins + 1), np.int64)
    for t in idx:
        pred = np.argmax(S[t] * w[:, None] + b[:, None], axis=0)
        conf = S[t][pred, np.arange(S[t].shape[1])]     # RAW score, as in predict()
        k = np.clip(np.rint(conf * nbins).astype(np.int64), 0, nbins)
        np.add.at(H, (G[t], pred, k), 1)
    return H


def coord(S, G, idx, nc, bg, obj, grid, order, vec, w, b, which, tau_rounds):
    """One coordinate-ascent pass. ⛔ tau is refitted per candidate -- freezing it
    caps the search below its own answer (test_head_fusion.py exists for that)."""
    for c in range(nc):
        best, bs = vec[c], -np.inf
        for j in order:
            trial = vec.copy(); trial[c] = grid[j]
            ww, bb = (trial, b) if which == 'w' else (w, trial)
            H = hists(S, G, idx, ww, bb, nc, NBINS)
            t = fit_tau(H, bg, NBINS, rounds=tau_rounds, objective=obj)
            sc = obj_miou(confusion_at(H, t, bg, NBINS), bg, obj)
            if sc > bs + 1e-9:
                bs, best = sc, grid[j]
        vec[c] = best
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='a --cache-full cache')
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--subsample', type=int, default=40000)
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--tau-rounds', type=int, default=2)
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
    T = len(files)
    if not T:
        raise SystemExit(f'no .npz under {args.cache}')
    print(f'{T} tiles | published τ = {args.tau} | b grid '
          f'{B_GRID[0]}..{B_GRID[-1]}\n')

    order = np.random.default_rng(args.seed).permutation(T)
    rng = np.random.default_rng(args.seed + 1000)
    S, G = load(files, args.subsample, nc, rng)
    folds = np.array_split(order, args.folds)
    one, zero = np.ones(nc), np.zeros(nc)
    rows = []

    for k in range(args.folds):
        ev, ca = folds[k], np.concatenate(
            [folds[j] for j in range(args.folds) if j != k])
        print(f'\nfold {k + 1}:', flush=True)
        H0 = hists(S, G, ca, one, zero, nc, NBINS)
        tau_b = fit_tau(H0, bg, NBINS, objective=args.objective)
        He0 = hists(S, G, ev, one, zero, nc, NBINS)
        A = miou(confusion_at(He0, np.full(nc, args.tau), bg, NBINS))
        B = miou(confusion_at(He0, tau_b, bg, NBINS))

        w = one.copy()
        for _ in range(args.rounds):
            w = coord(S, G, ca, nc, bg, args.objective, W_GRID,
                      W_ORDER, w, w, zero, 'w', args.tau_rounds)
        w = normalise(w)          # ⛔ gauge-fix before b is fitted -- see normalise()
        tau_c = fit_tau(hists(S, G, ca, w, zero, nc, NBINS), bg, NBINS,
                        objective=args.objective)
        C = miou(confusion_at(hists(S, G, ev, w, zero, nc, NBINS),
                              tau_c, bg, NBINS))

        b = zero.copy()
        for _ in range(args.rounds):
            b = coord(S, G, ca, nc, bg, args.objective, B_GRID, B_ORDER,
                      b, w, b, 'b', args.tau_rounds)
        tau_d = fit_tau(hists(S, G, ca, w, b, nc, NBINS), bg, NBINS,
                        objective=args.objective)
        D = miou(confusion_at(hists(S, G, ev, w, b, nc, NBINS), tau_d, bg, NBINS))

        print('    w = ' + ', '.join(f'{x:.2f}' for x in w))
        print('    b = ' + ', '.join(f'{x:+.2f}' for x in b))
        print(f'    A {A:.2f}   B {B:.2f}   C {C:.2f}   D {D:.2f}   (D−C {D-C:+.2f})')
        rows.append(dict(A=A, B=B, C=C, D=D, w=w.copy(), b=b.copy()))

    dc = np.array([r['D'] - r['C'] for r in rows])
    m, sd, pos = dc.mean(), dc.std(ddof=1), int((dc > 0).sum())
    Bm = np.array([r['b'] for r in rows])
    step = float(B_GRID[1] - B_GRID[0])
    stay = int((np.abs(Bm) <= step / 2 + 1e-9).all(axis=0).sum())

    md = ['# An affine reordering — does a per-class BIAS add anything to a scale?\n',
          f'- cache `{args.cache}` | tiles **{T}** | τ **{args.tau}** | {args.folds}-fold '
          f'| objective **`{args.objective}`**',
          f'- `b` grid **{B_GRID[0]}–{B_GRID[-1]}**, step {step:.2f}, symmetric about '
          f'**b = 0 = lever 2 unchanged**\n',
          '`pred = argmax_c (w_c · s_c + b_c)`, then keep `pred` if `s_pred ≥ τ_pred`. '
          '⭐ The threshold reads the **raw** score, so `b` is confined to the '
          'reordering — shifting the score τ sees would merely reparameterise τ.\n',
          '⭐ Where scores are large `w·s` dominates and `b` is irrelevant; at '
          '**low-confidence** pixels `b` decides who wins, and a multiplier cannot '
          'reach there. That is the regime the residual lives in.\n',
          '| fold | A published | B +τ | C +scale | D +bias | **D − C** |',
          '|---|---|---|---|---|---|']
    for i, r in enumerate(rows):
        md.append(f'| {i+1} | {r["A"]:.2f} | {r["B"]:.2f} | {r["C"]:.2f} | {r["D"]:.2f} '
                  f'| **{r["D"]-r["C"]:+.2f}** |')
    md += [f'\n- **D − C = {m:+.2f} ± {sd:.2f}**, {pos}/{args.folds} folds positive, '
           f'mean − 2·sd = **{m - 2*sd:+.2f}**\n', '## Fitted `b` per class\n',
           '| class | ' + ' | '.join(f'f{i+1}' for i in range(args.folds)) + ' | mean |',
           '|---|' + '---|' * (args.folds + 1)]
    for c, n in enumerate(LB.names):
        md.append(f'| {n} | ' + ' | '.join(f'{Bm[k, c]:+.2f}' for k in range(args.folds))
                  + f' | **{Bm[:, c].mean():+.2f}** |')
    md.append('\n`b < 0` = this class is suppressed where scores are small; `b > 0` = '
              'favoured there; `b = 0` = lever 2 alone was already right.\n')

    gate = (m - 2 * sd) > 0 and pos == args.folds
    md.append('## Verdict\n')
    if gate:
        md.append(f'⭐ **An affine reordering adds {m:+.2f} ± {sd:.2f} mIoU over a '
                  f'multiplicative one**, {pos}/{args.folds} folds, mean − 2·sd '
                  f'{m - 2*sd:+.2f}. ⛔ **The decision-side family is larger than the '
                  f'completeness argument suggested, and CLAUDE.md’s "there is no '
                  f'lever 5" must be corrected, not quietly amended.** ⚠️ A cached '
                  f'prediction until `eval.py` reproduces it (§9c), and ⚠️ it costs '
                  f'2N−1 parameters against N−1 — re-measure the calibration curve '
                  f'before the ~200-tile budget claim is repeated.')
    elif pos == args.folds:
        md.append(f'⚠️ **Promising, not established: {m:+.2f} ± {sd:.2f}**, every fold '
                  f'positive but the spread covers zero. Do not add it to the method.')
    else:
        md.append(f'⛔ **Null: {m:+.2f} ± {sd:.2f}**, {pos}/{args.folds} folds positive. '
                  f'A per-class bias buys nothing over a per-class scale. ⭐ The '
                  f'decision side is then exhausted **empirically as well as '
                  f'structurally** — a stronger closing statement than the one that '
                  f'was over-claimed, and reached by testing the family rather than '
                  f'by asserting it.')
    md.append(f'\n⭐ **{stay} of {nc} classes keep `b` = 0 in every fold** (within half a '
              f'grid step) — lever 2 alone was already right for those, whichever way '
              f'D − C goes.')
    out = '\n'.join(md)
    print('\n' + out)
    if args.md:
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).expanduser().write_text(out)
        print(f'\nwrote {args.md}')


if __name__ == '__main__':
    main()
