"""
Should some classes NOT be calibrated at all? Keep the published tau for them.

THE QUESTION. discard_after.py showed the four classes whose fitted tau moves
DOWN recover heavily and accurately, while `road` and `agricultural` -- whose tau
moves UP -- recover almost nothing and get it wrong. The obvious move is to fit
only the classes that benefit and leave the rest at the published value.

⛔ AND THE OBVIOUS VERSION OF THAT IS CHEATING. Naming `road` and `agricultural`
is a decision made by looking at LoveDA's held-out answer. Hardcode it and the
number goes up and means nothing -- it would not survive a new dataset, and it is
the same act as choosing a threshold on the test set. So this script runs FOUR
arms and only one of them is deployable:

    A        published tau everywhere                    the baseline
    B        fitted tau everywhere                       THE DEPLOYED METHOD
    hardcode fitted, except named classes pinned to published   ⛔ NOT deployable
    cv       per class, decide on the CALIBRATION split alone   ✅ deployable
    oracle   per class, decide on the EVALUATION fold           ⛔ upper bound

⭐ The two numbers that matter are `oracle - B`, which is all the freezing that
is available to any rule, and `cv - B`, which is what an honest rule extracts.
If oracle is large and cv is ~0, the information exists and is not reachable --
exactly the shape of the one-standard-error negative in SEPARABILITY_RESULTS 4.

⭐ WHY THIS IS CHEAP AND EXACT. Under `--objective real` the objective SEPARATES
(SEPARABILITY_RESULTS): for a fixed argmax a pixel predicted c either clears
tau_c or becomes the catch-all, never another real class. So each class's
TP/FP/FN depend on tau_c alone, "freeze class c" is a clean per-class decision,
and the histogram is a sufficient statistic. Rung B only -- lever 2 changes the
argmax and would break separability.

    python scripts/freeze_selection.py --cache ~/outputs/loveda_full_all/cache \\
        --tau 0.5 --freeze road,agricultural \\
        --md ~/outputs/week4/freeze_loveda.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, per_class_iou, miou, NBINS   # noqa: E402
from tau_cv import fit as fit_tau, per_tile_hists                 # noqa: E402


def iou_of(H, tau, bg):
    return per_class_iou(confusion_at(H, tau, bg, NBINS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--inner', type=int, default=5, help='inner folds for the cv arm')
    ap.add_argument('--freeze', default='',
                    help='comma-separated class names for the hardcoded arm')
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    print(f'  classes: {LB}')
    named = [n.strip() for n in args.freeze.split(',') if n.strip()]
    bad = [n for n in named if n not in LB.names]
    if bad:
        raise SystemExit(f'--freeze names not in this cache: {bad}\n'
                         f'available: {", ".join(LB.names)}')
    named_idx = [LB.names.index(n) for n in named]

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')
    print(f'{len(files)} tiles | published τ = {args.tau}\n')
    Hs = per_tile_hists(files, nc, NBINS)          # (T, nc, nc, nbins+1)
    T = len(Hs)
    order = np.random.default_rng(args.seed).permutation(T)
    folds = np.array_split(order, args.folds)
    pub = np.full(nc, args.tau)

    arms = ['A', 'B', 'hardcode', 'cv', 'oracle']
    sc = {a: [] for a in arms}
    dio = []                    # per-class held-out ΔIoU at rung B
    frozen_cv = np.zeros(nc, int)
    frozen_or = np.zeros(nc, int)

    for k in range(args.folds):
        ev, ca = folds[k], np.concatenate(
            [folds[j] for j in range(args.folds) if j != k])
        He, Hc = Hs[ev].sum(0), Hs[ca].sum(0)
        fit = fit_tau(Hc, bg, NBINS, objective=args.objective)

        # ---- cv arm: decide per class INSIDE the calibration split only
        inner = np.array_split(np.random.default_rng(args.seed + 100 + k)
                               .permutation(len(ca)), args.inner)
        gain = np.zeros(nc)
        for j in range(args.inner):
            iv = ca[inner[j]]
            it = ca[np.concatenate([inner[m] for m in range(args.inner) if m != j])]
            f_in = fit_tau(Hs[it].sum(0), bg, NBINS, objective=args.objective)
            Hv = Hs[iv].sum(0)
            gain += iou_of(Hv, f_in, bg) - iou_of(Hv, pub, bg)
        keep_cv = gain > 0                        # fit this class only if it paid
        tau_cv = np.where(keep_cv, fit, pub)
        frozen_cv += ~keep_cv

        # ---- oracle arm: decide on the EVALUATION fold. Peeking, upper bound.
        keep_or = (iou_of(He, fit, bg) - iou_of(He, pub, bg)) > 0
        tau_or = np.where(keep_or, fit, pub)
        frozen_or += ~keep_or

        tau_hc = fit.copy()
        tau_hc[named_idx] = args.tau

        for a, t in (('A', pub), ('B', fit), ('hardcode', tau_hc),
                     ('cv', tau_cv), ('oracle', tau_or)):
            sc[a].append(miou(confusion_at(He, t, bg, NBINS)))
        dio.append(iou_of(He, fit, bg) - iou_of(He, pub, bg))
        print(f'fold {k+1}:  ' + '  '.join(f'{a} {sc[a][-1]:.2f}' for a in arms))

    dio = np.array(dio)
    m = {a: np.mean(sc[a]) for a in arms}
    sd = {a: np.std(sc[a], ddof=1) for a in arms}
    d = {a: np.array(sc[a]) - np.array(sc['B']) for a in arms}

    md = ['# Should some classes keep the published τ? — a selection test\n',
          f'- cache `{args.cache}` | tiles **{T}** | published τ **{args.tau}** | '
          f'{args.folds}-fold held out | objective **`{args.objective}`**',
          f'- hardcoded arm freezes: **{", ".join(named) if named else "(none)"}**\n',
          '⭐ Under `--objective real` the objective **separates**, so "freeze class *c*" '
          'is a clean per-class decision and the histogram is a sufficient statistic. '
          'Rung B only — lever 2 changes the argmax and would break separability.\n',
          '| arm | how classes are chosen | mIoU | sd | **vs B** | deployable? |',
          '|---|---|---|---|---|---|']
    how = {'A': 'none fitted — the baseline', 'B': 'all fitted — **the method**',
           'hardcode': 'named by hand', 'cv': 'inner-CV on calibration only',
           'oracle': 'chosen on the evaluation fold'}
    dep = {'A': '—', 'B': '✅', 'hardcode': '⛔ **no**', 'cv': '✅', 'oracle': '⛔ no'}
    for a in arms:
        v = '—' if a == 'B' else f'**{d[a].mean():+.2f}**'
        md.append(f'| **{a}** | {how[a]} | {m[a]:.2f} | {sd[a]:.2f} | {v} | {dep[a]} |')

    md += ['\n## Per-class held-out Δ IoU at rung B (fitted − published)\n',
           '| class | ' + ' | '.join(f'f{i+1}' for i in range(args.folds))
           + ' | mean | folds − | frozen by cv | by oracle |',
           '|---|' + '---|' * (args.folds + 4)]
    for c in range(nc):
        md.append(f'| {LB.names[c]} | '
                  + ' | '.join(f'{dio[k, c]:+.2f}' for k in range(args.folds))
                  + f' | **{dio[:, c].mean():+.2f}** | {int((dio[:, c] < 0).sum())}/'
                  f'{args.folds} | {frozen_cv[c]}/{args.folds} | '
                  f'{frozen_or[c]}/{args.folds} |')

    avail, got = d['oracle'].mean(), d['cv'].mean()
    md += ['\n## Verdict\n']
    if avail <= 0.05:
        md.append(f'⛔ **There is nothing to select.** Even choosing on the evaluation '
                  f'fold — which no deployable rule may do — freezing is worth only '
                  f'**{avail:+.2f}** mIoU. **Every class that can be fitted should be, '
                  f'and the question is closed by the upper bound rather than by a '
                  f'failed rule.**')
    elif got > 0.05 and (d['cv'].mean() - 2 * d['cv'].std(ddof=1)) > 0:
        md.append(f'⭐ **A deployable selection rule gains {got:+.2f} mIoU**, against '
                  f'{avail:+.2f} available to a peeking oracle — {100*got/avail:.0f}% '
                  f'captured. ⚠️ Verify end to end before quoting it (§9c).')
    else:
        md.append(f'⛔ **The information exists and is not reachable.** A peeking '
                  f'oracle gains **{avail:+.2f}** mIoU by freezing; the honest '
                  f'inner-CV rule gets **{got:+.2f}**. Same shape as the '
                  f'one-standard-error negative (SEPARABILITY_RESULTS §4): which '
                  f'classes to calibrate is decidable *after* the fact and not '
                  f'*before*. **Fit every class.**')
    if named:
        md.append(f'\n⛔ **The hardcoded arm is not a result at any value.** '
                  f'`{", ".join(named)}` were named by reading LoveDA’s own held-out '
                  f'table, so its {d["hardcode"].mean():+.2f} is a measurement of how '
                  f'much peeking is worth, not of a method. It is reported only to '
                  f'show that gap — a rule chosen this way cannot transfer to a '
                  f'dataset whose losing classes differ, and §9e already showed the '
                  f'fitted thresholds themselves do not transfer across a domain.')
    md.append('\n⚠️ Rung B only. The deployed method is rung C (+ per-class scale), '
              'where the argmax moves and the per-class decision is no longer separable.')

    out = '\n'.join(md)
    print('\n' + out)
    if args.md:
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).expanduser().write_text(out)
        print(f'\nwrote {args.md}')


if __name__ == '__main__':
    main()
