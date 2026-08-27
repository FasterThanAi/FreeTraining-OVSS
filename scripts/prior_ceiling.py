"""
Week 3 — the ceiling test. What can the co-occurrence prior ACTUALLY recover?

Gates 1 and 2 asked whether M is accurate and whether it targets the baseline's
confusions. Neither asks the question the project lives or dies on:

    Take a region SAM 3 gave up on. Look only at the confident regions touching
    it. Apply M. How often is the answer right?

That is the method, stripped to its core -- no appearance term, no iteration, no
RAG. If this number is not clearly above the trivial baselines below, the extra
machinery will not save it, and it is far better to learn that in Week 3 than in
Week 9.

WHY THE PRIOR CANNOT BE TESTED ALONE
------------------------------------
A first version of this script scored the co-occurrence term by itself and the
prior came out BELOW chance. That was a defect in the test, and the reason is
worth stating because it constrains the method.

PMI's diagonal is zero by construction -- cooccurrence_gt.py counts only pixel
pairs with DIFFERENT labels, which is right for measuring class-pair structure
and wrong for labelling. A region that SAM 3 partly gave up on is usually the
same class as the confident region beside it: a water body 30% discarded has its
discarded part touching its own confident part. Scored on off-diagonal PMI
alone, "water" gets 0 while every other class gets its (often positive) PMI, so
the correct answer is the one candidate that can never win.

ROADMAP's scoring function already has both halves --

    score(c) = w_emb * sim(...) + w_coc * SUM_n M_eff[label(n), c] * w(n)
                                + w_nbr * vote(N(patch), c)

-- where `vote` is the same-class term and `M_eff` is the cross-class refinement.
So the question is not "does the prior work alone" (it cannot) but "does the
prior add anything ON TOP of the vote". That is what beta sweeps below.

    beta = 0    pure neighbour vote, no co-occurrence at all
    beta = 1    pure co-occurrence, no vote

If accuracy peaks at beta = 0, M earns nothing and should be dropped.

FOUR FAMILIES OF ROW, AND THE COMPARISON IS THE POINT
-----------------------------------------------------
    majority class     always answer `agriculture` (44.7% of real-class pixels).
                       A prior that cannot beat this is worthless.
    neighbour vote     copy the label of the largest confident neighbour. No M at
                       all. This is the honest "do you need co-occurrence?" test:
                       if M does not beat copying a neighbour, delete M.
    prior (mined M)    what the method can actually do, annotation-free.
    prior (GT M)       oracle upper bound -- the same procedure with a perfect
                       matrix. The gap between this and the mined row is the
                       price of ANALYSIS 3.2's circularity, in accuracy rather
                       than in bits.

REACHABILITY IS REPORTED SEPARATELY, AND IT MATTERS AS MUCH AS ACCURACY
----------------------------------------------------------------------
A discarded component with no confident real-class neighbour has nothing to
condition on -- an empty M_image and no seed (ANALYSIS 3.5's seeding problem,
WEEK1_RESULTS 7.7a's catastrophic tiles). Those pixels are UNREACHABLE by this
mechanism no matter how good M is, so they are excluded from the accuracy
figures and counted on their own line. Accuracy over reachable pixels x the
reachable fraction is the honest ceiling.

    python scripts/prior_ceiling.py \
        --cache ~/outputs/week2_tau0.5_instrumented/cache --tau 0.5 \
        --m-pred ~/outputs/week3/M_global_pred_t07.npz \
        --m-gt   ~/outputs/week3/M_global_gt_nobg.npz \
        --md ~/outputs/week3/prior_ceiling.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

LOVEDA = NC = REAL = BG = CLASSES = LB = None   # set by _init_labels()


def connected(mask):
    """Label 4-connected components. scipy if present, else a union-find pass."""
    try:
        from scipy.ndimage import label
        lab, n = label(mask)
        return lab.astype(np.int32), n
    except ImportError:
        pass
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                continue
            up = lab[i - 1, j] if i else 0
            lf = lab[i, j - 1] if j else 0
            if up and lf:
                lab[i, j] = min(up, lf); union(up, lf)
            elif up or lf:
                lab[i, j] = up or lf
            else:
                lab[i, j] = nxt; parent.append(nxt); nxt += 1
    remap = {}
    out = np.zeros_like(lab)
    for i in range(h):
        for j in range(w):
            if lab[i, j]:
                r = find(lab[i, j])
                if r not in remap:
                    remap[r] = len(remap) + 1
                out[i, j] = remap[r]
    return out, len(remap)


def neighbour_counts(comp, ncomp, committed):
    """(ncomp+1, NC) — shared boundary length between each component and each
    confident class. Same 4-connected definition as everywhere else in the repo."""
    acc = np.zeros((ncomp + 1) * NC, np.int64)

    def add(ca, lb):
        m = (ca > 0) & (lb > 0) & (lb != BG)   # touching a confident REAL class
        if m.any():
            acc[:] += np.bincount(ca[m].astype(np.int64) * NC + lb[m].astype(np.int64),
                                  minlength=(ncomp + 1) * NC)

    add(comp[:, :-1], committed[:, 1:]);  add(comp[:, 1:], committed[:, :-1])
    add(comp[:-1, :], committed[1:, :]);  add(comp[1:, :], committed[:-1, :])
    return acc.reshape(ncomp + 1, NC)


def load_pmi(path):
    """PMI matrix expanded to full (NC,NC) LoveDA indexing, missing = 0."""
    Z = np.load(Path(path).expanduser(), allow_pickle=True)
    v = list(Z['valid'])
    P = np.zeros((NC, NC))
    src = np.asarray(Z['pmi_bnd'])
    for i, ci in enumerate(v):
        for j, cj in enumerate(v):
            P[ci, cj] = src[i, j]
    return P


def zs(v):
    """z-score across candidate classes, so the vote term and the co-occurrence
    term are on one scale before they are mixed. Without this, beta would be
    comparing a boundary fraction on [0,1] against PMI in bits."""
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std() if v.std() > 1e-9 else np.zeros_like(v)


def rowz(P):
    """Row z-score. The mined matrix's magnitudes run ~4.2x the GT's, so summing
    raw bits over neighbours lets an over-confident row dominate. Ranking within
    a row is what survived validation; z-scoring keeps that and drops the scale."""
    Q = P.copy()
    for i in REAL:
        r = Q[i, REAL]
        if r.std() > 1e-9:
            Q[i, REAL] = (r - r.mean()) / r.std()
    return Q


def _init_labels(cache):
    """Resolve class names from the cache. See labels.py -- background is located
    BY NAME, never assumed to sit at index 0, because pointing these scripts at a
    dataset with a different class order would otherwise compute nonsense against
    perfectly valid array indices."""
    global LOVEDA, NC, REAL, BG, CLASSES, LB
    LB = labels.from_cache(cache)
    CLASSES = LB.names
    LOVEDA = ['unknown'] + LB.names
    NC = LB.nc
    REAL = LB.real
    BG = LB.bg
    print(f'  classes: {LB}')
    return LB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, default=0.5, help='INFERENCE tau')
    ap.add_argument('--m-pred', required=True)
    ap.add_argument('--m-gt', default=None)
    ap.add_argument('--min-size', type=int, default=64,
                    help='ignore components smaller than this many pixels')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()
    _init_labels(args.cache)

    Ps = {'prior (mined M)': load_pmi(args.m_pred)}
    if args.m_gt:
        Ps['prior (GT M)'] = load_pmi(args.m_gt)
    # discriminability weight per neighbour class, from the matrix being used
    Ws = {k: {i: float(np.var(P[i, REAL])) for i in REAL} for k, P in Ps.items()}
    Zs = {k: rowz(P) for k, P in Ps.items()}

    BETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    methods = ['majority class'] + [f'{k} β={b:.2f}' for k in Ps for b in BETAS]
    hit = {m: 0 for m in methods}
    hit_c = {m: 0 for m in methods}
    per_cls = {m: np.zeros((NC, 2), np.int64) for m in methods}
    # An aggregate can hide a real effect on the subset where the choice is
    # actually hard. A component touching ONE class has nothing to arbitrate --
    # the vote is trivially right or trivially wrong and M cannot help. The
    # decision only exists when two or more classes border the region, and large
    # components are where the pixels are (component accuracy 79% vs pixel 48%
    # says the big ones are the failures). So stratify by both.
    strata_hit = {}
    strata_tot = {}

    def strat_keys(cnt, npx):
        k = [f'{int((cnt[REAL] > 0).sum())} neighbour class'
             + ('' if int((cnt[REAL] > 0).sum()) == 1 else 'es')]
        for lo, hi, nm in [(0, 1000, '64–1k px'), (1000, 10000, '1k–10k px'),
                           (10000, 100000, '10k–100k px'), (100000, 1 << 60, '>100k px')]:
            if lo <= npx < hi:
                k.append(nm)
        return k

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')

    tot_disc = reach_px = reach_c = ncomp_tot = 0
    print(f'{len(files)} tiles | inference τ = {args.tau} | min component {args.min_size}px\n')

    for fi, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.uint8)
        conf = z['conf'].astype(np.float32)
        lbl = (z['pred'].astype(np.int16) + 1).astype(np.uint8)
        lbl[conf < args.tau] = 0
        lbl[gt == 0] = 0

        disc = (gt > 0) & (gt != BG) & ((lbl == 0) | (lbl == BG))
        if not disc.any():
            continue
        tot_disc += int(disc.sum())

        comp, n = connected(disc)
        if n == 0:
            continue
        size = np.bincount(comp.ravel(), minlength=n + 1)
        # majority GT class per component
        gmaj = np.bincount(comp.ravel().astype(np.int64) * NC + gt.ravel(),
                           minlength=(n + 1) * NC).reshape(n + 1, NC)
        gmaj[:, :2] = 0
        truth = gmaj.argmax(1)

        nb = neighbour_counts(comp, n, lbl)

        for c in range(1, n + 1):
            if size[c] < args.min_size:
                continue
            ncomp_tot += 1
            cnt = nb[c]
            if cnt[REAL].sum() == 0:
                continue                              # unreachable: no seed
            reach_px += int(size[c]); reach_c += 1
            t = int(truth[c])
            npx = int(size[c])

            preds = {'majority class': 7}
            share = cnt[REAL].astype(float) / cnt[REAL].sum()
            vote = zs(share)                       # same-class evidence
            for k in Ps:
                co = np.zeros(len(REAL))
                for i, nlab in enumerate(REAL):
                    if cnt[nlab]:
                        co += share[i] * Ws[k][nlab] * Zs[k][nlab, REAL]
                co = zs(co)                        # cross-class evidence
                for b in BETAS:
                    sc = (1.0 - b) * vote + b * co
                    preds[f'{k} β={b:.2f}'] = REAL[int(np.argmax(sc))]

            keys = strat_keys(cnt, npx)
            for m, p in preds.items():
                per_cls[m][t, 1] += npx
                ok = (p == t)
                if ok:
                    hit[m] += npx; hit_c[m] += 1; per_cls[m][t, 0] += npx
                for kk in keys:
                    strata_tot[(kk, m)] = strata_tot.get((kk, m), 0) + npx
                    if ok:
                        strata_hit[(kk, m)] = strata_hit.get((kk, m), 0) + npx

        if (fi + 1) % 250 == 0 or fi + 1 == len(files):
            print(f'  {fi + 1}/{len(files)}')

    md = ['# Week 3 — ceiling test: what can the prior actually recover?\n',
          f'- tiles: **{len(files)}**  |  inference τ: **{args.tau}**  |  '
          f'min component: **{args.min_size}px**',
          f'- pixels assigned to background: **{tot_disc:,}**',
          f'- components ≥ min size: **{ncomp_tot:,}**, of which **{reach_c:,}** have at '
          f'least one confident real-class neighbour',
          f'- **reachable pixels: {reach_px:,}** '
          f'(**{100 * reach_px / max(tot_disc, 1):.1f}%** of the residual)\n',
          'The rest have no confident neighbour to condition on — no seed, empty '
          '`M_image`. Unreachable by this mechanism at any M (ANALYSIS §3.5).\n',
          '| method | pixel accuracy | component accuracy | × majority baseline |',
          '|---|---|---|---|']
    base = hit['majority class'] / max(reach_px, 1)
    k0 = list(Ps)[0]
    vote_key = f'{k0} β=0.00'                     # beta=0 IS the pure neighbour vote
    for m in methods:
        a = hit[m] / max(reach_px, 1)
        c = hit_c[m] / max(reach_c, 1)
        tag = ''
        if m == vote_key:
            tag = '  ← pure neighbour vote, no M'
        elif m.endswith('β=1.00'):
            tag = '  ← pure co-occurrence, no vote'
        md.append(f'| {m}{tag} | **{100 * a:.1f}%** | {100 * c:.1f}% | '
                  f'{a / max(base, 1e-9):.2f}× |')

    md += ['\n## Per class — where does M help, if anywhere?\n',
           '| class | reachable px | majority | neighbour vote (β=0) | best mined β |',
           '|---|---|---|---|---|']
    mined = [m for m in methods if m.startswith(k0)]
    best = max(mined, key=lambda m: hit[m])
    for c in REAL:
        tot = per_cls[best][c, 1]
        if tot == 0:
            continue
        md.append(f'| {LOVEDA[c]} | {tot:,} | '
                  f'{100 * per_cls["majority class"][c, 0] / tot:.1f}% | '
                  f'{100 * per_cls[vote_key][c, 0] / tot:.1f}% | '
                  f'{100 * per_cls[best][c, 0] / tot:.1f}% |')
    md.append(f'\nBest mined variant: **{best}**. The `neighbour vote` column is '
              'β=0.00 — the same procedure with M switched off. **Read the two '
              'rightmost columns against each other: that difference is what the '
              "entire co-occurrence contribution is worth.**\n")

    order = ['1 neighbour class', '2 neighbour classes', '3 neighbour classes',
             '4 neighbour classes', '5 neighbour classes', '6 neighbour classes',
             '64–1k px', '1k–10k px', '10k–100k px', '>100k px']
    md += ['\n## Where the decision is actually hard\n',
           'A region touching one class has nothing to arbitrate. `Δ` is the '
           'co-occurrence contribution **on that stratum** — best mined β minus '
           'β=0. If M has a real effect it must show up here.\n',
           '| stratum | reachable px | β=0 (vote) | best mined β | **Δ** | oracle GT β | Δ oracle |',
           '|---|---|---|---|---|---|---|']
    gt_key = [m for m in methods if m.startswith(list(Ps)[-1])] if len(Ps) > 1 else []
    for kk in order:
        tot = strata_tot.get((kk, vote_key), 0)
        if tot == 0:
            continue
        v = strata_hit.get((kk, vote_key), 0) / tot
        cands = [m for m in methods if m.startswith(k0)]
        bb = max(cands, key=lambda m: strata_hit.get((kk, m), 0))
        b = strata_hit.get((kk, bb), 0) / tot
        row = (f'| {kk} | {tot:,} | {100 * v:.1f}% | {100 * b:.1f}% '
               f'({bb.split("β=")[-1]}) | **{100 * (b - v):+.2f}** |')
        if gt_key:
            gb = max(gt_key, key=lambda m: strata_hit.get((kk, m), 0))
            g = strata_hit.get((kk, gb), 0) / tot
            row += f' {100 * g:.1f}% ({gb.split("β=")[-1]}) | {100 * (g - v):+.2f} |'
        else:
            row += ' — | — |'
        md.append(row)
    md.append('\n`Δ oracle` is the ceiling on the co-occurrence term for that stratum: '
              'what a PERFECT matrix would add. If it is small, better mining cannot '
              'rescue it and the term does not belong in the method.\n')

    pv = hit[vote_key] / max(reach_px, 1)
    pp = hit[best] / max(reach_px, 1)
    md += ['\n## Verdict\n']
    if pp < base * 1.02:
        md.append(f'⛔ **The prior does not beat always answering `agriculture`** '
                  f'({100 * pp:.1f}% vs {100 * base:.1f}%). Co-occurrence over region '
                  'neighbours is not carrying the signal this project assumes. Do not '
                  'build the scoring function on it — re-scope to the RAG-agglomeration '
                  'contribution, or move the prior to a re-ranking role behind an '
                  'appearance term.')
    elif pp < pv * 1.02:
        md.append(f'⚠️ **The prior does not beat copying the largest neighbour** '
                  f'({100 * pp:.1f}% vs {100 * pv:.1f}%). The gain is spatial '
                  'smoothness, not semantics — and a reviewer will say DenseCRF does '
                  'that already (ANALYSIS §6, baseline row 4). Either find where M '
                  'beats the vote (check the per-class table — exclusion-heavy classes '
                  'like `water` are where it should) or drop M and keep the vote.')
    else:
        md.append(f'✅ **The prior beats both trivial baselines** — {100 * pp:.1f}% vs '
                  f'{100 * pv:.1f}% (neighbour vote) and {100 * base:.1f}% (majority). '
                  'Co-occurrence adds something a neighbour vote does not, which is '
                  'exactly the claim the paper has to defend. This table is the '
                  'evidence; keep it and re-run it after every method change.')
    md.append(f'\n> Honest ceiling for a region-level prior at τ={args.tau}: '
              f'**{100 * pp:.1f}% correct on {100 * reach_px / max(tot_disc, 1):.1f}% '
              f'of the residual** ≈ '
              f'**{100 * pp * reach_px / max(tot_disc, 1):.1f}%** of the '
              f'{tot_disc:,} background-assigned pixels. Compare against the '
              'τ-relaxation baseline: 1 correct per 1.73 wrong (WEEK1_RESULTS §8.2).')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
