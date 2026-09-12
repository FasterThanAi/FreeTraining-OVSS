"""
How much of the discarded residual does the METHOD actually bring back?

⛔ THE GAP THIS FILLS. Every discard figure in this project -- the 29.68%, the
per-class table in WEEK1 7.3, the whole motivation section -- is measured at the
PUBLISHED tau, before either lever. Nothing has ever reported the discard rate
AFTER calibration. So the paper motivates itself with "323 million real-class
pixels are thrown away" and never says how many came back. That is the first
question a reviewer asks after the introduction.

⚠️ AND THE ANSWER MAY BE SMALL, BY DESIGN. Per-class tau does not only lower
thresholds. On LoveDA `road` goes 0.5 -> 0.675, deliberately discarding MORE to
raise its precision, while `water` goes 0.5 -> 0.175 and discards far less. The
method optimises IoU, not recall, so the net discard can barely move while mIoU
improves. Reporting recovered pixels WITHOUT the newly-discarded ones would be
the same error as quoting a recall gain with no precision column -- which is
exactly what WEEK1 8.2 exists to prevent.

So this reports, per rung and per class:

    discarded        real-class GT pixels assigned to the catch-all
    recovered        discarded at the published tau, NOT discarded now
    newly discarded  kept at the published tau, discarded now
    recovered CORRECTLY   of the recovered, how many landed on the right class

⭐ The last one is the honest number. A pixel recovered into the wrong class is
not a recovery; WEEK1 8.2 measured tau relaxation at 1.73 wrong per right.

RUNGS (identical to argmax_reorder.py, so the numbers join up):

    A  published tau, w = 1     the baseline
    B  fitted tau,    w = 1     lever 1
    C  fitted tau,    fitted w  lever 1 + 2, the deployed method

5-fold, calibration and evaluation tiles always disjoint.

    python scripts/discard_after.py --cache ~/outputs/loveda_full_all/cache \\
        --tau 0.5 --objective real --md ~/outputs/week4/discard_after_loveda.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, NBINS                        # noqa: E402
from tau_cv import fit as fit_tau, obj_miou                       # noqa: E402

W_GRID = np.round(np.exp(np.linspace(np.log(0.40), np.log(2.50), 11)), 3)


def load(files, nsub, nc, rng):
    S, G = [], []
    for i, f in enumerate(files):
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}', flush=True)
        z = np.load(f)
        if 'logits' not in z.files:
            raise SystemExit(
                f'{f.name} has no `logits`. Rung C changes the argmax, so a '
                f'histogram cache is not a sufficient statistic. Needs a cache '
                f'written with --cache-full.')
        gt = z['gt'].ravel()
        keep = np.flatnonzero(gt > 0)
        if keep.size == 0:
            continue
        if keep.size > nsub:
            keep = rng.choice(keep, size=nsub, replace=False)
        # float32 before any tau comparison -- cached conf is float16 and the
        # boundary rounds (WEEK1 7.7, 100,493 px at 0.031%).
        S.append(z['logits'].astype(np.float32).reshape(nc, -1)[:, keep])
        G.append(gt[keep].astype(np.int32) - 1)
    return S, G


def predict(s, w, tau, bg):
    """The segmentor's rule: argmax of the SCALED score, threshold the RAW one."""
    pred = np.argmax(s * w[:, None], axis=0)
    conf = s[pred, np.arange(s.shape[1])]
    pred = np.where(conf < tau[pred], bg, pred)
    return pred


def hists(S, G, idx, w, nc, nbins):
    H = np.zeros((nc, nc, nbins + 1), np.int64)
    for t in idx:
        sc = S[t] * w[:, None]
        pred = np.argmax(sc, axis=0)
        conf = S[t][pred, np.arange(sc.shape[1])]
        b = np.clip(np.rint(conf * nbins).astype(np.int64), 0, nbins)
        np.add.at(H, (G[t], pred, b), 1)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='a --cache-full cache')
    ap.add_argument('--tau', type=float, required=True, help='the published threshold')
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
    print(f'  classes: {LB}   catch-all index {bg} (`{LB.names[bg]}`)')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    T = len(files)
    if not T:
        raise SystemExit(f'no .npz under {args.cache}')
    print(f'{T} tiles | published τ = {args.tau}\n')

    order = np.random.default_rng(args.seed).permutation(T)
    rng = np.random.default_rng(args.seed + 1000)
    S, G = load(files, args.subsample, nc, rng)
    folds = np.array_split(order, args.folds)
    one = np.ones(nc)

    RUNGS = ('A', 'B', 'C')
    real_px = np.zeros(nc, np.int64)
    disc = {k: np.zeros(nc, np.int64) for k in RUNGS}
    rec = {k: np.zeros(nc, np.int64) for k in RUNGS[1:]}
    rec_ok = {k: np.zeros(nc, np.int64) for k in RUNGS[1:]}
    new = {k: np.zeros(nc, np.int64) for k in RUNGS[1:]}

    for k in range(args.folds):
        ev = folds[k]
        ca = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        print(f'fold {k + 1}:', flush=True)

        tau_b = fit_tau(hists(S, G, ca, one, nc, NBINS), bg, NBINS,
                        objective=args.objective)
        w, tau_c = one.copy(), tau_b
        for _ in range(args.rounds):
            for c in range(nc):
                bs, bw = -np.inf, w[c]
                for v in W_GRID:
                    tr = w.copy(); tr[c] = v
                    H = hists(S, G, ca, tr, nc, NBINS)
                    t = fit_tau(H, bg, NBINS, rounds=args.tau_rounds,
                                objective=args.objective)
                    sc = obj_miou(confusion_at(H, t, bg, NBINS), bg, args.objective)
                    if sc > bs + 1e-9:
                        bs, bw = sc, v
                w[c] = bw
            tau_c = fit_tau(hists(S, G, ca, w, nc, NBINS), bg, NBINS,
                            objective=args.objective)

        cfg = {'A': (one, np.full(nc, args.tau)), 'B': (one, tau_b), 'C': (w, tau_c)}
        for t in ev:
            g = G[t]
            m = (g != bg)                     # GT carries a real land-cover class
            if not m.any():
                continue
            gm = g[m]
            np.add.at(real_px, gm, 1)
            p = {r: predict(S[t], *cfg[r], bg)[m] for r in RUNGS}
            dA = p['A'] == bg
            for r in RUNGS:
                np.add.at(disc[r], gm[p[r] == bg], 1)
            for r in RUNGS[1:]:
                dR = p[r] == bg
                np.add.at(rec[r], gm[dA & ~dR], 1)
                np.add.at(rec_ok[r], gm[dA & ~dR & (p[r] == gm)], 1)
                np.add.at(new[r], gm[~dA & dR], 1)
        print(f'  ρ/w = ' + ', '.join(f'{x:.2f}' for x in w))

    real = [c for c in range(nc) if c != bg]
    tot = int(real_px[real].sum())
    D = {r: int(disc[r][real].sum()) for r in RUNGS}
    R = {r: int(rec[r][real].sum()) for r in RUNGS[1:]}
    RO = {r: int(rec_ok[r][real].sum()) for r in RUNGS[1:]}
    N = {r: int(new[r][real].sum()) for r in RUNGS[1:]}
    pc = lambda x: 100.0 * x / max(tot, 1)

    # ⛔ SELF-CHECK, on every run. A pixel discarded at A and not at X is
    # `recovered`; one kept at A and discarded at X is `newly discarded`. Those
    # are the only two ways the count can move, so
    #     discard_A − discard_X  ==  recovered_X − newlyDiscarded_X
    # must hold EXACTLY as integers. It is cheap and it would catch a masking
    # slip that otherwise produces a complete, plausible, wrong table -- the
    # failure mode this codebase keeps hitting.
    for r in RUNGS[1:]:
        lhs, rhs = D['A'] - D[r], R[r] - N[r]
        if lhs != rhs:
            raise SystemExit(
                f'\n⛔ ACCOUNTING BUG at rung {r}: discard_A − discard_{r} = {lhs} '
                f'but recovered − newly_discarded = {rhs}. The per-pixel masks '
                f'disagree with the totals; do not trust any number in this run.\n')
        if RO[r] > R[r]:
            raise SystemExit(f'\n⛔ rung {r}: more correct recoveries than '
                             f'recoveries ({RO[r]} > {R[r]}).\n')
    print(f'  ✅ accounting identity holds at every rung '
          f'(discard_A − discard_X == recovered − newly discarded)')

    md = ['# How much of the discard does calibration actually bring back?\n',
          f'- cache `{args.cache}` | tiles **{T}** | published τ **{args.tau}** | '
          f'{args.folds}-fold, held out | objective **`{args.objective}`**',
          f'- catch-all: **`{LB.names[bg]}`** | real-class pixels scored: '
          f'**{tot:,}** (subsampled {args.subsample}/tile)',
          '- ✅ accounting identity verified at every rung: '
          '`discard_A − discard_X == recovered − newly discarded`\n',
          '⚠️ **A subsample, so these are RATES, not the split’s absolute pixel '
          'counts.** The rate is what transfers; `measure_discard_rate.py` owns the '
          'absolutes at the published τ.\n',
          '| rung | rule | **discarded** | recovered vs A | of those, **correct** | newly discarded |',
          '|---|---|---|---|---|---|',
          f'| **A** | published τ | **{pc(D["A"]):.2f}%** | — | — | — |']
    for r in RUNGS[1:]:
        lbl = {'B': 'per-class τ', 'C': '+ per-class scale'}[r]
        ok = 100.0 * RO[r] / max(R[r], 1)
        md.append(f'| **{r}** | {lbl} | **{pc(D[r]):.2f}%** | {pc(R[r]):.2f}% | '
                  f'**{ok:.1f}%** | {pc(N[r]):.2f}% |')

    md += ['\n## Per class, rung C against the baseline\n',
           '| class | real px | discard A | discard C | change | recovered | correctly |',
           '|---|---|---|---|---|---|---|']
    for c in real:
        n = max(int(real_px[c]), 1)
        p = lambda x: 100.0 * x / n
        ok = 100.0 * rec_ok['C'][c] / max(int(rec['C'][c]), 1)
        md.append(f'| {LB.names[c]} | {real_px[c]:,} | {p(disc["A"][c]):.1f}% | '
                  f'{p(disc["C"][c]):.1f}% | **{p(disc["C"][c]) - p(disc["A"][c]):+.1f}** | '
                  f'{p(rec["C"][c]):.1f}% | {ok:.1f}% |')

    net = pc(D['C']) - pc(D['A'])
    okC = 100.0 * RO['C'] / max(R['C'], 1)
    ratio = (R['C'] - RO['C']) / max(RO['C'], 1)
    md += ['\n## Verdict\n',
           f'**Discard moves {pc(D["A"]):.2f}% → {pc(D["C"]):.2f}%, a change of '
           f'{net:+.2f} points.**\n']
    if abs(net) < 1.0:
        md.append('⭐ **The method barely changes how much is discarded — it changes '
                  'WHICH pixels are.** That is the point, and it is what the '
                  'per-class table shows: some classes discard less, others '
                  'deliberately discard more to buy precision. ⛔ **Do not report the '
                  'gain as "recovering the residual".**')
    elif net < 0:
        md.append(f'The method discards **{abs(net):.2f} points less** than the '
                  f'baseline. Report it beside the newly-discarded column, never alone.')
    else:
        md.append(f'⚠️ The method discards **{net:.2f} points MORE** than the baseline '
                  f'and still improves mIoU — precision bought at the cost of recall. '
                  f'State this explicitly; a reader who has only seen the motivation '
                  f'section will expect the opposite sign.')
    md.append(f'\n⭐ **Of the pixels it recovers, {okC:.1f}% land on the correct class** '
              f'— {ratio:.2f} wrong per right, against **1.73** for the τ→0.1 sweep '
              f'(WEEK1 §8.2). That comparison is the one worth quoting: the same '
              f'residual, reached selectively rather than by relaxing a global knob.')
    md.append('\n⚠️ **Recovered and newly-discarded must always appear together.** '
              'Quoting recovery alone is the error §8.2 exists to prevent.')

    out = '\n'.join(md)
    print('\n' + out)
    if args.md:
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).expanduser().write_text(out)
        print(f'\nwrote {args.md}')


if __name__ == '__main__':
    main()
