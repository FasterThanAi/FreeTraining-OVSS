"""
Why is the calibration gain a rural result? Decompose it. ROADMAP §6.7 / WEEK3 §9e.

THE OPEN QUESTION. §9e found per-class calibration is worth +2.77 mIoU on LoveDA's
rural half and +0.10 on urban. §9f then ruled out the obvious explanation: the
gain does NOT track residual size, on either dataset. §7c had already ruled out
catch-all share, because the effect saturates and both strata sit past it. So the
paper currently says the gap is unexplained, which is honest and unsatisfying.

THE REMAINING CANDIDATE is class composition. mIoU is an UNWEIGHTED mean over
classes, so a domain does not gain simply by containing more of a class -- it
gains when a class's own IoU improves more there. The question is therefore which
classes move, and what distinguishes them.

WHAT THIS DOES, in two steps that must both hold:

  1. DECOMPOSE. Delta mIoU is the mean of per-class Delta IoU, so the urban/rural
     gap decomposes EXACTLY into per-class contributions. This is arithmetic, not
     inference -- it says which classes the gap is made of, with no room to argue.

  2. EXPLAIN. Across every (domain, class) cell, ask which per-class statistic
     predicts Delta IoU: the class's share of the domain, how much of it is being
     discarded, its precision, its recall, or its precision-recall gap. If one of
     them ranks the cells, the gap has a mechanism; if none does, composition is
     ruled out too and the paper says so.

⚠️ Step 2 has few points (classes x domains), so Spearman is reported with an
EXACT permutation p-value rather than a table lookup -- the same discipline as
§9d, where |rho| = 0.6 over six classes arises by chance about a fifth of the time.

    python scripts/composition.py --cache ~/outputs/week3_fused/cache --tau 0.5 \\
        --map ~/splits/loveda_domain.txt --md ~/outputs/week3/composition.md
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, per_class_iou, NBINS         # noqa: E402
from tau_cv import per_tile_hists, fit                            # noqa: E402
from tau_domain import read_map                                   # noqa: E402


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float('nan')


def perm_p(a, b, exact_cap=8, draws=20000, seed=0):
    """Two-sided p by permutation: EXACT where that is affordable, sampled
    otherwise. ⚠️ A table lookup is misleading at this many points -- over 6
    items the smallest attainable two-sided p is 2/720, and |rho| = 0.6 arises
    by chance about a fifth of the time. Returns (p, how) so the report can say
    which; silently degrading to nan, as an earlier version did, hides the only
    number that says whether rho means anything."""
    n, r = len(a), abs(spearman(a, b))
    b = list(b)
    if n <= exact_cap:
        hits = sum(1 for pm in itertools.permutations(range(n))
                   if abs(spearman(a, [b[i] for i in pm])) >= r - 1e-12)
        return hits / math.factorial(n), "exact"
    rng = np.random.default_rng(seed)
    arr = np.asarray(b, float)
    hits = sum(1 for _ in range(draws)
               if abs(spearman(a, rng.permutation(arr))) >= r - 1e-12)
    return (hits + 1) / (draws + 1), "sampled"


def stats_at(C, bg):
    """share, discard, precision, recall per class from one confusion matrix."""
    tot = C.sum()
    out = {}
    for c in range(C.shape[0]):
        row, col = C[c].sum(), C[:, c].sum()
        out[c] = dict(share=row / tot if tot else 0.0,
                      discard=C[c, bg] / row if row else 0.0,
                      prec=C[c, c] / col if col else float('nan'),
                      rec=C[c, c] / row if row else float('nan'))
        out[c]['gap'] = out[c]['prec'] - out[c]['rec']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    dom = read_map(args.map, [f.stem for f in files])
    names = sorted(set(dom))
    idx = {d: np.array([i for i, x in enumerate(dom) if x == d]) for d in names}
    print(f'  classes: {LB}')
    print('  ' + ' | '.join(f'{d}: {len(idx[d])} tiles' for d in names) + '\n')

    PT = per_tile_hists(files, nc, NBINS)
    rng = np.random.default_rng(args.seed)
    pub = np.full(nc, args.tau)

    dIoU, base = {}, {}
    for d in names:
        ii = idx[d]
        order = rng.permutation(len(ii))
        parts = np.array_split(order, args.folds)
        acc = []
        for k in range(args.folds):
            te = ii[parts[k]]
            tr = ii[np.concatenate([parts[j] for j in range(args.folds) if j != k])]
            taus = fit(PT[tr].sum(0).astype(np.int64), bg, NBINS, objective=args.objective)
            H = PT[te].sum(0).astype(np.int64)
            acc.append(per_class_iou(confusion_at(H, taus, bg, NBINS))
                       - per_class_iou(confusion_at(H, pub, bg, NBINS)))
        dIoU[d] = np.nanmean(np.array(acc), axis=0)
        base[d] = stats_at(confusion_at(PT[ii].sum(0).astype(np.int64), pub, bg, NBINS), bg)
        print(f'  {d}: Δ mIoU {np.nanmean(dIoU[d]):+.2f}')

    A, B = names[0], names[1]
    gapd = float(np.nanmean(dIoU[B]) - np.nanmean(dIoU[A]))
    hi, lo = (B, A) if gapd > 0 else (A, B)
    contrib = [(LB.names[c], dIoU[hi][c] - dIoU[lo][c],
                (dIoU[hi][c] - dIoU[lo][c]) / nc) for c in range(nc)]
    contrib.sort(key=lambda r: -abs(r[1]))
    total = abs(gapd)

    md = [f'# What is the {hi}/{lo} calibration gap made of?\n',
          f'- cache: `{args.cache}` | τ = **{args.tau}** | {args.folds}-fold within each '
          f'domain | objective **`{args.objective}`**',
          f'- Δ mIoU: **{hi} {np.nanmean(dIoU[hi]):+.2f}**, {lo} '
          f'{np.nanmean(dIoU[lo]):+.2f} — a gap of **{total:.2f}**\n',
          '## 1. Decomposition — arithmetic, not inference\n',
          f'mIoU is the *unweighted* mean over {nc} classes, so the gap is exactly the mean '
          f'of the per-class differences. A domain does not gain by containing more of a '
          f'class; it gains when that class\'s own IoU improves more there.\n',
          f'| class | Δ IoU {hi} | Δ IoU {lo} | difference | share of the gap |',
          '|---|---|---|---|---|']
    run = 0.0
    for nm, diff, share in contrib:
        c = LB.names.index(nm)
        run += share
        md.append(f'| {nm}{" *(catch-all)*" if c == bg else ""} | {dIoU[hi][c]:+.2f} | '
                  f'{dIoU[lo][c]:+.2f} | **{diff:+.2f}** | '
                  f'**{share / total * 100:+.0f}%** |')
    top2 = sum(r[2] for r in contrib[:2]) / total * 100
    md.append(f'\n⭐ **`{contrib[0][0]}` and `{contrib[1][0]}` alone account for '
              f'{top2:.0f}% of the gap.**\n')

    md += ['## 2. What distinguishes the classes that move?\n',
           f'Every (domain, class) cell, {len(names) * nc} in all, at the published τ. '
           f'`discard` is the fraction of that class assigned to the catch-all; `gap` is '
           f'precision − recall.\n',
           '| domain | class | share | discard | precision | recall | P−R gap | **Δ IoU** |',
           '|---|---|---|---|---|---|---|---|']
    cells = []
    for d in names:
        for c in range(nc):
            if c == bg:
                continue
            s = base[d][c]
            cells.append((d, c, s, dIoU[d][c]))
            md.append(f'| {d} | {LB.names[c]} | {s["share"] * 100:.1f}% | '
                      f'{s["discard"] * 100:.1f}% | {s["prec"] * 100:.1f} | '
                      f'{s["rec"] * 100:.1f} | {s["gap"] * 100:+.1f} | '
                      f'**{dIoU[d][c]:+.2f}** |')

    y = [r[3] for r in cells]
    md += ['\n| candidate | ρ vs Δ IoU | permutation p |', '|---|---|---|']
    best = None
    for key, lab in (('share', 'class share of the domain'),
                     ('discard', 'fraction discarded to the catch-all'),
                     ('gap', 'precision − recall gap'),
                     ('prec', 'precision'), ('rec', 'recall')):
        x = [r[2][key] for r in cells]
        rho, (pv, how) = spearman(x, y), perm_p(x, y)
        md.append(f'| {lab} | **{rho:+.3f}** | {pv:.4f} *({how})* |')
        if best is None or abs(rho) > abs(best[1]):
            best = (lab, rho, pv, how, key)

    # ⚠️ A statistic that RANKS the cells on average has not necessarily explained
    # the classes the gap is MADE OF. Check the top contributors individually: if
    # a class carries a large share of the gap while its winning statistic is
    # nearly the same in both domains, that share is still unexplained and the
    # verdict must not claim otherwise.
    unexplained = []
    for nm, diff, sh in contrib[:3]:
        c = LB.names.index(nm)
        if c == bg or abs(sh) / total < 0.10:
            continue
        va, vb = base[hi][c][best[4]], base[lo][c][best[4]]
        if abs(va - vb) / (abs(va) + abs(vb) + 1e-9) < 0.15:
            unexplained.append((nm, sh / total * 100, va, vb, dIoU[hi][c], dIoU[lo][c]))

    md += ['\n## Verdict\n']
    strong = abs(best[1]) >= 0.6 and best[2] <= 0.05
    if strong:
        md.append(f'✅ **The gap is class composition, and the class statistic that explains '
                  f'it is {best[0]}** (ρ = {best[1]:+.3f}, {best[3]} '
                  f'p = {best[2]:.4f}). Together with the decomposition above — `{contrib[0][0]}` and '
                    f'`{contrib[1][0]}` carrying {top2:.0f}% of it — the '
                    f'{hi}/{lo} difference is no longer unexplained: those domains differ '
                    f'in how much of each class the baseline leaves on the table, and '
                    f'calibration collects exactly that.\n\n'
                    f'⚠️ This is an *explanation*, not a predictor. The statistic is '
                    f'label-derived, so it does not resurrect the label-free rule §9f ruled '
                    f'out — it says what the gain is made of, not how to know in advance.')
        if unexplained:
            worst = sum(u[1] for u in unexplained)
            md.append(f'\n⛔ **But it does not explain the largest contributor'
                      f'{"s" if len(unexplained) > 1 else ""}, and that must be stated '
                      f'beside the ρ.** ' + ' '.join(
                          f'`{nm}` carries **{sh:.0f}%** of the gap, yet its {best[0]} is '
                          f'{va * 100:+.1f} in {hi} against {vb * 100:+.1f} in {lo} — '
                          f'effectively the same — while its Δ IoU is {dh:+.2f} against '
                          f'{dl:+.2f}.' for nm, sh, va, vb, dh, dl in unexplained)
                      + f' So the statistic ranks the cells on average and **{worst:.0f}% '
                        f'of the gap still has no mechanism.** A reviewer comparing those '
                        f'two rows will see it immediately; say it first.')
    else:
        md.append(f'⛔ **Composition does not explain it either.** The best candidate is '
                  f'{best[0]} at ρ = {best[1]:+.3f} ({best[3]} p = '
                  f'{best[2]:.4f}), which over {len(cells)} cells is not distinguishable from chance. '
                    f'The decomposition still stands as arithmetic — `{contrib[0][0]}` and '
                    f'`{contrib[1][0]}` carry {top2:.0f}% of the gap — but *why those '
                    f'classes* is unresolved, and the paper must say so.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
