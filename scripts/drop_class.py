"""
What if the catch-all is not in the VOCABULARY at all?

THE IDEA. `background` is prompted as a word, and SAM 3 has nothing to detect:
median S_pres = 0.022, twenty to forty times below every real class (WEEK1 9.2b).
It is not a visual concept, it is LoveDA's bin for "none of the above". Yet it
still COMPETES IN THE ARGMAX, and when it wins it is wrong: WEEK1 7.7 measured
6.0% of background assignments as argmax wins at conf >= tau -- 19,378,177
pixels, in 24 tiles, EVERY ONE OF THEM WATER.

So: drop its channel, and let tau alone produce background. A pixel becomes
background because nothing cleared its threshold, not because a bad prompt
out-competed a good one.

⭐ FREE, AND EXACT. Every class is an independent forward pass and the only
cross-class operation is the argmax (vocab_intervention.py 's docstring proves
this from the source), so dropping a class from the vocabulary IS dropping its
channel -- exactly, not approximately. No GPU, and every arm reads identical
model outputs.

⛔ ORACLE FIRST. Run before the fit, and reported even if the fit is not run:
how many real-class pixels does the catch-all currently STEAL at the argmax, and
what is the most that recovering all of them could be worth? If that ceiling is
small, the idea is dead in seconds rather than in GPU-hours. This is the
screening step that levers 3, 4 and 5 did not get.

    python scripts/drop_class.py --cache ~/outputs/loveda_full_all/cache \\
        --tau 0.5 --md ~/outputs/week4/drop_bg_loveda.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, per_class_iou, miou, NBINS   # noqa: E402
from tau_cv import fit as fit_tau, obj_miou                       # noqa: E402

W_GRID = np.round(np.exp(np.linspace(np.log(0.40), np.log(2.50), 11)), 3)
W_ORDER = np.argsort(np.abs(np.log(W_GRID)), kind='stable')


def hists(S, G, idx, w, nc, nbins, bg):
    """(gt, pred, conf-bin) histogram. `bg` is the catch-all COLUMN index in the
    output space, which differs from its row index once a channel is dropped."""
    H = np.zeros((nc, nc, nbins + 1), np.int64)
    for t in idx:
        pred = np.argmax(S[t] * w[:, None], axis=0)
        conf = S[t][pred, np.arange(S[t].shape[1])]
        k = np.clip(np.rint(conf * nbins).astype(np.int64), 0, nbins)
        np.add.at(H, (G[t], pred, k), 1)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='a --cache-full cache')
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--subsample', type=int, default=40000)
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--tau-rounds', type=int, default=2)
    ap.add_argument('--oracle-only', action='store_true',
                    help='print the ceiling and stop -- seconds, no fit')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    print(f'  classes: {LB}   catch-all index {bg} (`{LB.names[bg]}`)')
    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')
    print(f'{len(files)} tiles | published τ = {args.tau}\n')

    rng = np.random.default_rng(args.seed + 1000)
    S, G = [], []
    steal = np.zeros(nc, np.int64)       # real-class px the catch-all wins at argmax
    steal_clear = np.zeros(nc, np.int64)  # ... and whose runner-up would clear τ
    runner_right = np.zeros(nc, np.int64)  # ... and whose runner-up is CORRECT
    real_px = np.zeros(nc, np.int64)
    for i, f in enumerate(files):
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}', flush=True)
        z = np.load(f)
        if 'logits' not in z.files:
            raise SystemExit(f'{f.name} has no `logits`; needs --cache-full.')
        gt = z['gt'].ravel()
        keep = np.flatnonzero(gt > 0)
        if keep.size == 0:
            continue
        if keep.size > args.subsample:
            keep = rng.choice(keep, size=args.subsample, replace=False)
        lg = z['logits'].astype(np.float32).reshape(nc, -1)[:, keep]
        g = gt[keep].astype(np.int32) - 1
        S.append(lg); G.append(g)

        # ---- the oracle screen, on every tile
        m = g != bg
        if not m.any():
            continue
        gm, sm = g[m], lg[:, m]
        np.add.at(real_px, gm, 1)
        won = np.argmax(sm, axis=0)
        stolen = won == bg                      # catch-all took a real-class pixel
        if stolen.any():
            np.add.at(steal, gm[stolen], 1)
            no_bg = sm[:, stolen].copy()
            no_bg[bg] = -np.inf                 # what wins once bg is gone
            rup = np.argmax(no_bg, axis=0)
            rconf = no_bg[rup, np.arange(rup.size)]
            clears = rconf >= args.tau
            np.add.at(steal_clear, gm[stolen][clears], 1)
            np.add.at(runner_right, gm[stolen][clears & (rup == gm[stolen])], 1)

    real = [c for c in range(nc) if c != bg]
    tot = int(real_px[real].sum())
    st, cl, ok = (int(x[real].sum()) for x in (steal, steal_clear, runner_right))
    pc = lambda x: 100.0 * x / max(tot, 1)

    md = ['# Dropping the catch-all from the vocabulary — ceiling first\n',
          f'- cache `{args.cache}` | tiles **{len(files)}** | τ **{args.tau}** | '
          f'catch-all **`{LB.names[bg]}`**',
          f'- real-class pixels scored: **{tot:,}** (subsample {args.subsample}/tile)\n',
          '⭐ Dropping a class from the vocabulary **is** dropping its channel — every class is '
          'an independent forward pass and the only cross-class operation is the argmax. No GPU, '
          'and both arms read identical model outputs.\n',
          '## The ceiling, before any fit\n',
          '| | pixels | share of real-class |',
          '|---|---|---|',
          f'| the catch-all wins the argmax over a real class | {st:,} | **{pc(st):.2f}%** |',
          f'| …and the runner-up would clear the published τ | {cl:,} | **{pc(cl):.2f}%** |',
          f'| ⭐ …and that runner-up is the **correct** class | {ok:,} | ⭐ **{pc(ok):.2f}%** |\n',
          f'> **At most {pc(ok):.2f}% of real-class pixels can be recovered by removing the '
          f'catch-all from the vocabulary, and that is with perfect luck on every one of them.**\n']
    if st:
        md += ['Per class, the pixels the catch-all steals:\n',
               '| class | real px | stolen | runner-up correct |', '|---|---|---|---|']
        for c in real:
            if steal[c]:
                md.append(f'| {LB.names[c]} | {real_px[c]:,} | '
                          f'{100.0*steal[c]/max(int(real_px[c]),1):.2f}% | '
                          f'{100.0*runner_right[c]/max(int(steal[c]),1):.1f}% |')
    print('\n'.join(md))

    if args.oracle_only or st == 0:
        if args.md:
            Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
            Path(args.md).expanduser().write_text('\n'.join(md))
        return

    # ------------------------------------------------ the measured arm
    # ⚠️ Dropping the channel changes the OUTPUT SPACE: the argmax can no longer
    # return bg, so tau alone produces it. We keep the same nc columns and simply
    # make bg unreachable at the argmax, which keeps every metric comparable.
    order = np.random.default_rng(args.seed).permutation(len(S))
    folds = np.array_split(order, args.folds)
    one = np.ones(nc)
    rows = []
    for k in range(args.folds):
        ev, ca = folds[k], np.concatenate(
            [folds[j] for j in range(args.folds) if j != k])
        print(f'\nfold {k+1}:', flush=True)
        A = miou(confusion_at(hists(S, G, ev, one, nc, NBINS, bg),
                              np.full(nc, args.tau), bg, NBINS))
        tau_b = fit_tau(hists(S, G, ca, one, nc, NBINS, bg), bg, NBINS,
                        objective=args.objective)
        B = miou(confusion_at(hists(S, G, ev, one, nc, NBINS, bg), tau_b, bg, NBINS))

        Sd = [s.copy() for s in S]
        for s in Sd:
            s[bg] = -np.inf                      # <<< the intervention
        Ad = miou(confusion_at(hists(Sd, G, ev, one, nc, NBINS, bg),
                               np.full(nc, args.tau), bg, NBINS))
        tau_d = fit_tau(hists(Sd, G, ca, one, nc, NBINS, bg), bg, NBINS,
                        objective=args.objective)
        Bd = miou(confusion_at(hists(Sd, G, ev, one, nc, NBINS, bg), tau_d, bg, NBINS))
        print(f'    with bg: A {A:.2f}  B {B:.2f}   |   no bg: A {Ad:.2f}  B {Bd:.2f}'
              f'   (ΔB {Bd-B:+.2f})')
        rows.append((A, B, Ad, Bd))

    d = np.array([r[3] - r[1] for r in rows])
    m, sd, pos = d.mean(), d.std(ddof=1), int((d > 0).sum())
    md += ['\n## Measured, 5-fold held out\n',
           '| fold | published τ | +per-class τ | **no bg**, published τ | **no bg**, +τ | Δ |',
           '|---|---|---|---|---|---|']
    for i, (A, B, Ad, Bd) in enumerate(rows):
        md.append(f'| {i+1} | {A:.2f} | {B:.2f} | {Ad:.2f} | {Bd:.2f} | **{Bd-B:+.2f}** |')
    md.append(f'\n- **Δ = {m:+.2f} ± {sd:.2f}**, {pos}/{args.folds} folds positive, '
              f'mean − 2·sd = **{m - 2*sd:+.2f}**')
    md.append('\n## Verdict\n')
    if (m - 2 * sd) > 0 and pos == args.folds:
        md.append(f'⭐ **Removing the catch-all from the vocabulary adds {m:+.2f} ± {sd:.2f} '
                  f'mIoU** on top of per-class τ. ⚠️ Verify end to end before quoting (§9c).')
    else:
        md.append(f'⛔ **Null: {m:+.2f} ± {sd:.2f}**, {pos}/{args.folds} folds. The catch-all '
                  f'earns its place in the vocabulary, and the {pc(ok):.2f}% ceiling above '
                  f'is why — there was never much to win.')
    out = '\n'.join(md)
    print('\n' + '\n'.join(md[md.index('\n## Measured, 5-fold held out\n'):]))
    if args.md:
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).expanduser().write_text(out)
        print(f'\nwrote {args.md}')


if __name__ == '__main__':
    main()
