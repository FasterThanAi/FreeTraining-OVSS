"""
Week 3 — the go/no-go. Does recovering the residual actually MOVE mIoU?

Everything measured so far scores accuracy ON the residual. None of it answers
the only question a reviewer asks: does the number in Table 1 go up?

    baseline (SegEarth-OV3, tau=0.5)                    47.38 mIoU
    lower tau to 0.1                                    41.83   (-5.54)
    remove presence gating                              35.39  (-11.97)
    recover the residual selectively                    ??

WHY SELECTIVITY IS THE METHOD, NOT A DETAIL
-------------------------------------------
prior_ceiling.py says region-level propagation labels the reachable residual at
48.4% precision -- better than threshold relaxation's 36.6% (1 right per 1.73
wrong, WEEK1_RESULTS 8.2), but still wrong more often than not. Relabelling all
166M reachable pixels at 48% would trade background IoU for real-class recall at
roughly break-even, which is not a contribution.

But precision is NOT uniform. It runs 86.8% on building and 21.9% on forest;
81.7% on components under 1k px and 33.1% on those over 100k. So the method's
real job is not "what label" -- the neighbour vote answers that -- it is
**WHEN TO ANSWER AT ALL**. Abstain where the evidence is weak, commit where it
is strong, and the recovered set can be far more precise than 48%.

That is the axis this script sweeps, and it is what distinguishes the method
from the two baselines a reviewer will raise: threshold relaxation has no
abstention (it commits everywhere below tau) and DenseCRF has none either (it
relabels every pixel). An explicit calibrated abstention is the claim.

TWO ABSTENTION SIGNALS, BOTH FREE
---------------------------------
    margin  = score(top1) - score(top2) over candidate classes. High margin
              means the neighbourhood points one way.
    purity  = share of the region's boundary held by its top neighbour class.
              High purity means the region sits inside one thing.

VALIDATION GATE. The `recover nothing` row MUST reproduce 47.37 mIoU and
323,084,415 background-assigned pixels. If it does not, the confusion
bookkeeping here disagrees with measure_discard_rate.py and every number below
is void.

    python scripts/selective_recovery_miou.py \
        --cache ~/outputs/week2_tau0.5_instrumented/cache --tau 0.5 \
        --m-pred ~/outputs/week3/M_global_pred_t07.npz \
        --md ~/outputs/week3/selective_recovery.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

LOVEDA = ['unknown', 'background', 'building', 'road',
          'water', 'barren', 'forest', 'agriculture']
NC = len(LOVEDA)
REAL = list(range(2, NC))
NCLS = 7                                   # background + 6 real, for the metric


from atoms import get_atomiser, load_image      # noqa: E402


def neighbour_counts(comp, ncomp, committed):
    acc = np.zeros((ncomp + 1) * NC, np.int64)

    def add(ca, lb):
        m = (ca > 0) & (lb >= 2)
        if m.any():
            acc[:] += np.bincount(ca[m].astype(np.int64) * NC + lb[m].astype(np.int64),
                                  minlength=(ncomp + 1) * NC)

    add(comp[:, :-1], committed[:, 1:]);  add(comp[:, 1:], committed[:, :-1])
    add(comp[:-1, :], committed[1:, :]);  add(comp[1:, :], committed[:-1, :])
    return acc.reshape(ncomp + 1, NC)


def rowz(P):
    Q = P.copy()
    for i in REAL:
        r = Q[i, REAL]
        if r.std() > 1e-9:
            Q[i, REAL] = (r - r.mean()) / r.std()
    return Q


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std() if v.std() > 1e-9 else np.zeros_like(v)


def miou(C):
    """mIoU over the 7 LoveDA classes from a confusion matrix (rows=true)."""
    ious = []
    for k in range(NCLS):
        tp = C[k, k]
        den = C[k].sum() + C[:, k].sum() - tp
        if den > 0:
            ious.append(tp / den)
    return 100.0 * float(np.mean(ious)) if ious else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--m-pred', required=True)
    ap.add_argument('--beta', type=float, default=0.25,
                    help='co-occurrence weight in the mixture; 0 = pure neighbour vote')
    ap.add_argument('--min-size', type=int, default=64)
    ap.add_argument('--margins', type=float, nargs='+', default=[0.0, 1.0])
    ap.add_argument('--purities', type=float, nargs='+', default=[0.0, 0.7])
    ap.add_argument('--max-sizes', type=int, nargs='+',
                    default=[500, 2000, 10000, 0],
                    help='atom size CEILING in px; 0 = no limit. Small atoms are '
                         'boundary seams and vote 81.7%% correct; >100k px atoms 33.1%%.')
    ap.add_argument('--atoms', choices=['slic', 'cc'], default='slic',
                    help="'slic' -- oracle ceiling 92.8%%, settled by atom_quality.py. "
                         "'cc' is the ablation row: ceiling 72.8%%, atoms sprawl to a "
                         "whole tile.")
    ap.add_argument('--img-dir', default=None, help='required for --atoms slic')
    ap.add_argument('--n-segments', type=int, default=600)
    ap.add_argument('--regions', choices=['all', 'oracle'], default='all',
                    help="'all' considers every background-assigned pixel, which is "
                         "what inference can actually see. 'oracle' restricts to "
                         "pixels GT says are real classes -- a supervision leak, "
                         "reportable only as an upper bound.")
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()
    if args.atoms == 'slic' and not args.img_dir:
        raise SystemExit('--atoms slic needs --img-dir (e.g. ~/data/loveda/img_dir/val)')
    atomise = get_atomiser(args.atoms)

    Z = np.load(Path(args.m_pred).expanduser(), allow_pickle=True)
    v = list(Z['valid']); src = np.asarray(Z['pmi_bnd'])
    P = np.zeros((NC, NC))
    for i, ci in enumerate(v):
        for j, cj in enumerate(v):
            P[ci, cj] = src[i, j]
    W = {i: float(np.var(P[i, REAL])) for i in REAL}
    Zr = rowz(P)

    # Two filters the earlier sweeps never applied, both justified by measurements
    # already in hand rather than by guesswork:
    #
    #   SIZE CEILING. prior_ceiling.py stratified by atom size: 64-1k px atoms are
    #   voted correctly 81.7% of the time, >100k px atoms only 33.1%. Small atoms
    #   are the thin boundary seams of 9.1a -- genuine extensions of the confident
    #   region beside them. Large ones are whole dropped regions where the vote is
    #   guessing. The sweep had a min-size FLOOR and no ceiling.
    #
    #   CLASS SUBSET. Vote precision is wildly uneven -- building 86.8%, water
    #   69.8%, forest 21.9%. Committing only to the classes the vote gets right is
    #   a legitimate abstention rule, and unlike a confidence threshold it needs no
    #   signal SAM 3 does not have. Ordered by measured reliability, never by GT
    #   from this split.
    CLASS_LADDER = [
        ('building', [2]),
        ('building+water', [2, 4]),
        ('bld+wat+road', [2, 4, 3]),
        ('bld+wat+road+barren', [2, 4, 3, 5]),
        ('all classes', REAL),
    ]
    settings = [(m, p, ms, cn)
                for p in args.purities for m in args.margins
                for ms in args.max_sizes for cn, _ in CLASS_LADDER]
    CLASS_SETS = {cn: set(cs) for cn, cs in CLASS_LADDER}
    C = {s: np.zeros((NCLS, NCLS), np.int64) for s in settings}
    C['none'] = np.zeros((NCLS, NCLS), np.int64)
    rec = {s: [0, 0] for s in settings}          # [recovered px, correct px]
    disc_total = 0

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    print(f'{len(files)} tiles | τ={args.tau} | β={args.beta} | '
          f'{len(settings)} operating points\n')

    for fi, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.uint8)
        conf = z['conf'].astype(np.float32)
        L = (z['pred'].astype(np.int16) + 1).astype(np.uint8)
        L[conf < args.tau] = 0
        L[gt == 0] = 0

        valid = gt > 0
        base = np.where(L == 0, 1, L).astype(np.uint8)     # unknown -> background
        gi = gt[valid].astype(np.int64) - 1
        bi = base[valid].astype(np.int64) - 1
        C['none'] += np.bincount(gi * NCLS + bi,
                                 minlength=NCLS * NCLS).reshape(NCLS, NCLS)

        # EVERY pixel the baseline assigned to background, not just the ones GT
        # says are real classes. Selecting on `gt >= 2` here would hand the method
        # advance knowledge of which background pixels are worth touching, so it
        # could never damage true background -- precisely the protection
        # tau-relaxation does not get (section 8.2: at tau=0.1 over 70% of true
        # background is misassigned). That is a supervision leak and it inflated
        # an earlier version of this script by +3.47 mIoU. --regions oracle
        # reproduces it deliberately, as an upper bound, never as a result.
        if args.regions == 'oracle':
            disc = (gt >= 2) & (base == 1)
        else:
            disc = (base == 1) & (gt > 0)
        disc_total += int(((gt >= 2) & (base == 1)).sum())
        if not disc.any():
            for s in settings:
                C[s] += np.bincount(gi * NCLS + bi,
                                    minlength=NCLS * NCLS).reshape(NCLS, NCLS)
            continue

        img = load_image(args.img_dir, f.stem) if args.atoms == 'slic' else None
        comp, n = atomise(disc, img, args.n_segments)
        if n == 0:
            for s_ in settings:
                C[s_] += np.bincount(gi * NCLS + bi,
                                     minlength=NCLS * NCLS).reshape(NCLS, NCLS)
            continue
        size = np.bincount(comp.ravel(), minlength=n + 1)
        nb = neighbour_counts(comp, n, L)

        # decide once per component, then apply each operating point as a filter
        assign = np.zeros(n + 1, np.uint8)
        marg = np.zeros(n + 1); pur = np.zeros(n + 1)
        for c in range(1, n + 1):
            cnt = nb[c]
            if size[c] < args.min_size or cnt[REAL].sum() == 0:
                continue
            share = cnt[REAL].astype(float) / cnt[REAL].sum()
            co = np.zeros(len(REAL))
            for i, nl in enumerate(REAL):
                if cnt[nl]:
                    co += share[i] * W[nl] * Zr[nl, REAL]
            sc = (1.0 - args.beta) * zs(share) + args.beta * zs(co)
            o = np.argsort(sc)[::-1]
            assign[c] = REAL[int(o[0])]
            marg[c] = float(sc[o[0]] - sc[o[1]]) if len(o) > 1 else 9.9
            pur[c] = float(share.max())

        for st in settings:
            mth, pth, mxs, cn = st
            allowed = CLASS_SETS[cn]
            ok_cls = np.array([a in allowed for a in assign], dtype=bool)
            ok_sz = (size[:n + 1] <= mxs) if mxs > 0 else np.ones(n + 1, bool)
            take = (assign > 0) & (marg >= mth) & (pur >= pth) & ok_cls & ok_sz
            if take.any():
                newp = base.copy()
                sel = take[comp]
                newp[sel] = assign[comp[sel]]
                pi = newp[valid].astype(np.int64) - 1
                C[st] += np.bincount(gi * NCLS + pi,
                                     minlength=NCLS * NCLS).reshape(NCLS, NCLS)
                got = sel & disc
                rec[st][0] += int(got.sum())
                # correct against GT -- a relabelled TRUE-background pixel counts
                # as wrong, which is the whole point of scoping to `all`
                rec[st][1] += int((newp[got] == gt[got]).sum())
            else:
                C[st] += np.bincount(gi * NCLS + bi,
                                     minlength=NCLS * NCLS).reshape(NCLS, NCLS)

        if (fi + 1) % 250 == 0 or fi + 1 == len(files):
            print(f'  {fi + 1}/{len(files)}')

    base_miou = miou(C['none'])
    md = ['# Week 3 — does selective recovery move mIoU?\n',
          f'- tiles: **{len(files)}**  |  τ: **{args.tau}**  |  β: **{args.beta}**  |  '
          f'min component: **{args.min_size}px**',
          f'- background-assigned real-class pixels: **{disc_total:,}**',
          f'- atoms: **`{args.atoms}`**'
          + ('  (oracle ceiling 92.8%)' if args.atoms == 'slic'
             else '  ⚠️ ceiling 72.8% — ablation row only') + '',
          f'- region scope: **`{args.regions}`**'
          + ('  ⚠️ **ORACLE — uses GT to choose which pixels to touch. Upper bound '
             'only, never a result.**' if args.regions == 'oracle'
             else '  (every background-assigned pixel, as at inference)') + '\n',
          '## Validation gate\n',
          f'| | this run | expected |', '|---|---|---|',
          f'| mIoU, recover nothing | **{base_miou:.2f}** | 47.37 |',
          f'| background-assigned px | **{disc_total:,}** | 323,084,415 |']
    # The expected figures are LoveDA-val-specific, so the gate only means
    # something on the full 1669-tile cache.
    full = len(files) == 1669 and not args.limit
    okgate = (abs(base_miou - 47.37) < 0.05) if full else True
    if not full:
        md.append('\n_(Partial cache — the 47.37 / 323,084,415 gate applies only to the '
                  'full 1669-tile LoveDA val cache. Not checked here.)_\n')
    elif okgate:
        md.append('\n✅ **Gate passed** — this script\'s confusion bookkeeping agrees '
                  'with `measure_discard_rate.py`, so the rows below are trustworthy.\n')
    else:
        md.append('\n⛔ **GATE FAILED** — the confusion bookkeeping disagrees with '
                  '`measure_discard_rate.py`; every number below is void.\n')

    md += ['## Operating points\n',
           'Abstain unless the region clears both thresholds. `margin` = top1−top2 of '
           'the score; `purity` = share of boundary held by the top neighbour class.\n',
           'Sorted by mIoU. `max px` is the atom size ceiling; `classes` is which '
           'labels we are willing to commit to.\n',
           '| classes | max px | margin ≥ | purity ≥ | recovered px | precision | '
           '**mIoU** | Δ |',
           '|---|---|---|---|---|---|---|---|']
    best = None
    rows = []
    for st in settings:
        r, c = rec[st]
        m = miou(C[st])
        rows.append((m, st, r, 100 * c / max(r, 1)))
        if r > 0 and (best is None or m > best[0]):
            best = (m, st, r, 100 * c / max(r, 1))
    for m, st, r, pr in sorted(rows, reverse=True, key=lambda t: t[0])[:20]:
        mth, pth, mxs, cn = st
        md.append(f'| {cn} | {mxs if mxs else "∞"} | {mth:.1f} | {pth:.1f} | '
                  f'{r:,} ({100 * r / max(disc_total, 1):.1f}%) | {pr:.1f}% | '
                  f'**{m:.2f}** | {m - base_miou:+.2f} |')

    # A gain has to be big enough to survive a second dataset and a reviewer.
    # +0.04 on one split is 0.08% relative -- reproducible, since this is
    # deterministic numpy over a fixed cache, but meaningless. An earlier version
    # printed a green tick on exactly that, which is the same failure as the
    # gate-1 verdict: a bar set where the answer already is.
    MEANINGFUL = 0.50
    md += ['\n## Verdict\n']
    if not okgate:
        md.append('⛔ Fix the gate before reading anything above.')
    elif best is None or best[0] <= base_miou + MEANINGFUL:
        got = (best[0] - base_miou) if best else 0.0
        md.append(f'⛔ **No operating point meaningfully beats the baseline** '
                  f'({base_miou:.2f}; best {base_miou + got:.2f}, {got:+.2f}, under the '
                  f'{MEANINGFUL:+.2f} bar). '
                  'Recovering the residual by neighbour propagation costs more in '
                  'background IoU than it returns in real-class recall, at every '
                  'abstention level tested. That is a fourth refuted intervention '
                  'alongside τ-relaxation and presence removal — a strong motivation '
                  'section, but not a method. **Re-scope before Week 4.**')
    else:
        m, (mth, pth, mxs, cn), r, pr = best
        md.append(f'✅ **Best: classes `{cn}`, max atom {mxs if mxs else "∞"} px, '
                  f'margin ≥ {mth:.1f}, purity ≥ {pth:.1f} → mIoU '
                  f'{m:.2f} ({m - base_miou:+.2f})**, recovering {r:,} pixels '
                  f'({100 * r / max(disc_total, 1):.1f}% of the residual) at '
                  f'**{pr:.1f}% precision**.\n\n'
                  'Compare against the alternatives already measured: τ→0.1 gives '
                  '−5.54 at 36.6% precision, presence removal −11.97. **Selective '
                  'recovery is the first intervention that reaches the residual and '
                  'does not cost more than it returns.** The abstention rule is the '
                  'contribution — threshold relaxation and DenseCRF both commit '
                  'everywhere and that is exactly why they lose.')

    md.append('\n> Report recovery rate and precision separately (WEEK1_RESULTS §12). '
              'A headline mIoU gain without the precision column invites the objection '
              'the τ-sweep already answers.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
