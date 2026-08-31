"""
Does calibrated per-class τ survive a domain shift? LoveDA urban vs rural.

WHY THIS, AND WHY NOW. §9b's +1.18 ± 0.45 is the project's only positive method
result, and it rests on ONE dataset. It also carries a scope limit that was
measured but never bounded: fitting on LoveDA *train* and evaluating on *val*
gives −0.12, so calibration evidently has to match the evaluation distribution.
"Has to match" is not a usable statement -- a practitioner needs to know how much
mismatch costs. This measures it, on strata that are known to differ: ANALYSIS
§4.4 finds 10 of 15 class pairs flip PMI sign between urban and rural, and §7a
measures discard at 18.5% vs 39.3%. If a calibrated threshold transfers across
THAT, the method is more general than claimed. If it does not, the paper gets a
quantified scope statement instead of a caveat.

⚠️ It is also, deliberately, close to a second and third dataset for free. The two
domains have different class mixes, different scene structure and a 2x different
residual, and the cache already exists -- so this is CPU-only.

THREE ARMS, all evaluated on IDENTICAL held-out tiles so the comparison is of
calibration sources and nothing else:

  matched      fit on N tiles from the target domain          <- the §9b protocol
  mismatched   fit on N tiles from the OTHER domain           <- the transfer test
  pooled       fit on N tiles drawn across both domains       <- what a
                                                                 practitioner
                                                                 would actually do

⚠️ ALL THREE DRAW THE SAME N. Fitting `mismatched` on the whole of the other
domain would confound domain shift with calibration-set size, and the learning
curve in §9b shows size matters a lot below 200 tiles. Equal N is the only way
the difference can be attributed to the domain.

    python scripts/tau_domain.py --cache ~/outputs/week3_fused/cache --tau 0.5 \
        --map ~/splits/loveda_domain.txt --md ~/outputs/week3/tau_domain.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                    # noqa: E402
from tau_oracle import confusion_at, miou, per_class_iou, NBINS   # noqa: E402
from tau_cv import per_tile_hists, fit                            # noqa: E402


def read_map(path, stems):
    """tile stem -> domain, from the file confound_split.py --make-map writes."""
    m = {}
    for line in Path(path).expanduser().read_text().splitlines():
        if line.strip():
            k, _, v = line.partition('\t')
            m[k.strip()] = v.strip().lower()
    missing = [s for s in stems if s not in m]
    if missing:
        raise SystemExit(
            f'{len(missing)} of {len(stems)} cached tiles are absent from {path} '
            f'(e.g. {missing[:5]}).\nA partial map would silently evaluate on a '
            f'biased subset, so this is fatal. Rebuild it with:\n\n'
            f'    python scripts/confound_split.py --make-map <LoveDA Val root> '
            f'--map-out {path}')
    return [m[s] for s in stems]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--map', required=True, help='tile\tdomain, from confound_split --make-map')
    ap.add_argument('--calib', type=int, default=200,
                    help='calibration tiles, IDENTICAL for all three arms')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--folds', type=int, default=5, help='within-domain k-fold')
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')
    dom = read_map(args.map, [f.stem for f in files])
    names = sorted(set(dom))
    if len(names) != 2:
        raise SystemExit(f'expected exactly 2 domains, found {names}')
    A, B = names
    idx = {d: np.array([i for i, x in enumerate(dom) if x == d]) for d in names}
    print(f'  classes: {LB}')
    print(f'  {A}: {len(idx[A])} tiles | {B}: {len(idx[B])} tiles | '
          f'calib N = {args.calib}, identical for every arm\n')
    for d in names:
        if len(idx[d]) <= args.calib:
            raise SystemExit(f'domain `{d}` has {len(idx[d])} tiles, which is not more '
                             f'than --calib {args.calib}; nothing would be held out.')

    PT = per_tile_hists(files, nc, NBINS)
    rng = np.random.default_rng(args.seed)
    pub = np.full(nc, args.tau)

    def H(ii):
        return PT[ii].sum(0).astype(np.int64)

    def ev(ii, taus):
        return miou(confusion_at(H(ii), taus, bg, NBINS))

    # ---------------------------------------------- within each domain, k-fold
    within, within_pc, within_tau = {}, {}, {}
    for d in names:
        ii = idx[d]
        order = rng.permutation(len(ii))
        folds = np.array_split(order, args.folds)
        gs, pcs = [], []
        for k in range(args.folds):
            te = ii[folds[k]]
            tr = ii[np.concatenate([folds[j] for j in range(args.folds) if j != k])]
            assert not set(tr) & set(te)
            taus = fit(H(tr), bg, NBINS, objective=args.objective)
            b, g = ev(te, pub), ev(te, taus)
            gs.append(g - b)
            pcs.append(per_class_iou(confusion_at(H(te), taus, bg, NBINS))
                       - per_class_iou(confusion_at(H(te), pub, bg, NBINS)))
            print(f'  {d} fold {k + 1}: {b:.2f} -> {g:.2f}  ({g - b:+.2f})')
        within[d] = np.array(gs)
        within_pc[d] = np.nanmean(np.array(pcs), axis=0)
        # one fit on the whole domain, purely to compare the THRESHOLDS themselves
        within_tau[d] = fit(H(ii), bg, NBINS, objective=args.objective)

    # ---------------------------------------------- transfer, equal-N, same eval
    arms = ('matched', 'mismatched', 'pooled')
    tr_res = {d: {a: [] for a in arms} for d in names}
    for tgt in names:
        src = B if tgt == A else A
        for r in range(args.repeats):
            perm = rng.permutation(idx[tgt])
            cal_t, ev_t = perm[:args.calib], perm[args.calib:]
            pool = np.concatenate([idx[src], cal_t])
            draws = {
                'matched': cal_t,
                'mismatched': rng.choice(idx[src], args.calib, replace=False),
                'pooled': rng.choice(pool, args.calib, replace=False),
            }
            base = ev(ev_t, pub)
            for a in arms:
                assert not set(draws[a]) & set(ev_t), 'calibration leaked into eval'
                assert len(draws[a]) == args.calib
                tr_res[tgt][a].append(ev(ev_t, fit(H(draws[a]), bg, NBINS,
                                                   objective=args.objective)) - base)
        for a in arms:
            v = np.array(tr_res[tgt][a])
            print(f'  -> {tgt}: {a:<10} {v.mean():+.2f} ± '
                  f'{v.std(ddof=1) if len(v) > 1 else 0:.2f}')

    # ---------------------------------------------- report
    md = [f'# Per-class τ across a domain shift — LoveDA {A} vs {B}\n',
          f'- cache: `{args.cache}` | published τ: **{args.tau}** | fit objective: '
          f'**`{args.objective}`**',
          f'- {A}: **{len(idx[A])}** tiles | {B}: **{len(idx[B])}** tiles',
          f'- calibration size **{args.calib} tiles for every arm**, so a difference '
          f'between arms cannot be a calibration-size effect',
          f'- {args.repeats} draws per arm; all three arms scored on the **same** '
          f'held-out tiles within a draw\n',
          '## 1. Does the method work inside each domain?\n',
          f'{args.folds}-fold within each domain, calibration and evaluation disjoint, '
          'the published-τ baseline recomputed on the same held-out tiles.\n',
          '| domain | tiles | mean Δ mIoU | sd | worst fold |', '|---|---|---|---|---|']
    for d in names:
        g = within[d]
        md.append(f'| **{d}** | {len(idx[d])} | **{g.mean():+.2f}** | '
                  f'{g.std(ddof=1):.2f} | {g.min():+.2f} |')

    md += ['\n### Per-class Δ IoU, by domain\n',
           '| class | ' + ' | '.join(names) + ' |',
           '|---' * (len(names) + 1) + '|']
    for c in range(nc):
        cells = ' | '.join(f'**{within_pc[d][c]:+.2f}**' if np.isfinite(within_pc[d][c])
                           else '—' for d in names)
        md.append(f'| {LB.names[c]}{" *(catch-all)*" if c == bg else ""} | {cells} |')
    for d in names:
        rs = float(np.nansum([within_pc[d][c] for c in range(nc) if c != bg]))
        md.append(f'\n`{d}`: catch-all **{within_pc[d][bg]:+.2f}**, real classes '
                  f'**{rs:+.2f}** in aggregate.')

    md += ['\n## 2. What do the two domains actually want?\n',
           'One fit per domain on all of its tiles — these are the thresholds, not a '
           'held-out score, and they are here to show *whether the domains disagree*.\n',
           '| class | ' + ' | '.join(names) + ' | difference |',
           '|---' * (len(names) + 2) + '|']
    diffs = []
    for c in range(nc):
        if c == bg:
            continue
        va, vb = within_tau[A][c], within_tau[B][c]
        diffs.append(abs(va - vb))
        md.append(f'| {LB.names[c]} | {va:.3f} | {vb:.3f} | **{abs(va - vb):.3f}** |')
    md.append(f'\nPublished τ is a single **{args.tau}** for every class and both domains. '
              f'Mean |difference| between the domains: **{np.mean(diffs):.3f}**, '
              f'max **{np.max(diffs):.3f}**.')

    md += ['\n## 3. Transfer — where should the calibration tiles come from?\n',
           f'Each row fits on {args.calib} tiles from the stated source and evaluates on '
           f'held-out tiles of the target domain. Δ is against the published τ on those '
           f'same tiles.\n',
           '| target | matched (own domain) | mismatched (other domain) | pooled (both) |',
           '|---|---|---|---|']
    for tgt in names:
        cells = []
        for a in arms:
            v = np.array(tr_res[tgt][a])
            cells.append(f'**{v.mean():+.2f}** ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}')
        md.append(f'| **{tgt}** | ' + ' | '.join(cells) + ' |')

    mm = {d: np.mean(tr_res[d]['matched']) for d in names}
    xx = {d: np.mean(tr_res[d]['mismatched']) for d in names}
    pp = {d: np.mean(tr_res[d]['pooled']) for d in names}
    M, X, P = np.mean(list(mm.values())), np.mean(list(xx.values())), np.mean(list(pp.values()))
    keep = X / M if M > 0 else float('nan')

    md += ['\n## Verdict\n']
    if min(within[d].mean() for d in names) <= 0:
        weak = min(names, key=lambda d: within[d].mean())
        md.append(f'⛔ **The method does not hold in both domains** — `{weak}` gives '
                  f'{within[weak].mean():+.2f}. §9b\'s pooled gain is carried by one '
                  f'stratum, and that must be stated wherever +1.18 appears.')
    else:
        md.append(f'✅ **The method holds independently in both domains** — '
                  + ', '.join(f'`{d}` {within[d].mean():+.2f} ± {within[d].std(ddof=1):.2f}'
                              for d in names)
                  + '.'
                  + (' These strata differ in discard rate by roughly 2x and flip the '
                     'sign of 10 of 15 class-adjacency pairs (ANALYSIS §4.4), so this '
                     'is closer to independent replication than to a re-split.'
                     if {'urban', 'rural'} <= set(names) else ''))
    if np.isfinite(keep) and keep >= 0.8:
        md.append(f'\n✅ **And it transfers.** Calibrating on the *other* domain retains '
                  f'{keep * 100:.0f}% of the matched gain ({X:+.2f} against {M:+.2f}). '
                  f'The thresholds are a property of the model\'s per-class calibration '
                  f'more than of the scene, which widens the method\'s scope: the '
                  f'−0.12 train→val result is then about those splits differing 2x in '
                  f'discard rate, not about domain shift as such.')
    elif np.isfinite(keep) and keep > 0:
        md.append(f'\n⚠️ **Transfer is partial** — the other domain retains {keep * 100:.0f}% '
                  f'({X:+.2f} against {M:+.2f}), still positive but materially worse. '
                  f'Calibration data should come from the target domain where it can.')
    else:
        md.append(f'\n⛔ **It does not transfer** — calibrating on the other domain gives '
                  f'{X:+.2f} against {M:+.2f} matched. The fitted thresholds are '
                  f'domain-specific, and that is the honest scope of §9b.')
    # The mean over targets hides direction, and direction is the useful part: a
    # threshold set may generalise one way and not the other.
    ret = {d: (xx[d] / mm[d] if mm[d] > 0 else float('nan')) for d in names}
    worst = min(names, key=lambda d: ret[d] if np.isfinite(ret[d]) else -1e9)
    best = max(names, key=lambda d: ret[d] if np.isfinite(ret[d]) else -1e9)
    if np.isfinite(ret[worst]) and np.isfinite(ret[best]) and ret[best] - ret[worst] > 0.15:
        src_w = B if worst == A else A
        md.append(f'\n⚠️ **Transfer is asymmetric, and the mean hides it.** Calibrating on '
                  f'`{src_w}` and evaluating on `{worst}` retains {ret[worst] * 100:.0f}% '
                  f'({xx[worst]:+.2f} of {mm[worst]:+.2f}), while the reverse direction '
                  f'retains {ret[best] * 100:.0f}%. So `{worst}` needs something '
                  f'`{src_w}` cannot supply — read the threshold table above for which '
                  f'class it is. Quote the per-direction figures, not the average.')

    md.append(f'\n**Pooling both domains gives {P:+.2f}**, against {M:+.2f} matched and '
              f'{X:+.2f} mismatched — at the same {args.calib}-tile budget. That is the '
              f'practical recommendation, since a practitioner holding mixed calibration '
              f'tiles does not have to stratify them'
              + (' and loses nothing by not doing so.' if P >= 0.9 * M else
                 ', though stratifying is measurably better here.'))

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
