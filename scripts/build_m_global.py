"""
Week 3 — build M_global, the corpus-level co-occurrence prior.

WHAT THIS IS. A 7x7 table saying, for every ordered class pair, how much more
(or less) than chance those two classes share a boundary. It is the "rule book"
the method conditions on when it tries to label a region SAM 3 gave up on.

Mined from SAM 3's own CONFIDENT predictions -- no ground-truth labels are read
in --source pred. That is what keeps the annotation-free claim honest.

THE ONE DEFINITION THAT MATTERS
-------------------------------
A predicted pixel is used only if the model actually committed to it:

    conf <  tau            -> UNKNOWN (label 0), excluded from every count
    conf >= tau            -> label = pred + 1   (LoveDA numbering, 1..7)

Below-tau pixels are NOT confident background. They are the 323M-pixel residual
this project exists to recover (WEEK1_RESULTS 7.1). Counting them as background
would build the prior out of exactly the mass we are trying to explain, and the
prior would then confidently tell us they are background. The `pred == 0 and
conf >= tau` pixels ARE kept -- there the model confidently chose background,
which is a real observation (mechanism B, 7.7).

Label 0 = unknown mirrors LoveDA's own 0 = no-data, so the resulting matrix is
comparable CELL FOR CELL with the ground-truth matrix from cooccurrence_gt.py
and pmi_permutation_null.py. Same adjacency definition (4-connected shared
boundary length), same class indexing, same PMI code path.

DIRECTEDNESS -- read this before using the output
-------------------------------------------------
Shared-boundary counts are SYMMETRIC by construction. If a water pixel touches
an agricultural pixel that is one boundary; it has no direction, so
M[water, agri] == M[agri, water] always, and no amount of counting changes that.

CLAUDE.md's settled decision "M is directed, not symmetric" was derived from the
CONFUSION matrix (water -> agricultural 19.3M with no reverse, 8.1b), which is a
different object. Directedness is real, but it enters through the ROW-NORMALISED
CONDITIONAL, not the counts:

    P(c | neighbour = n) = M[n, c] / sum_c M[n, c]        != P(n | neighbour = c)

These differ because the class marginals differ -- agricultural is 44.7% of the
dataset, water 18.3%. So "a region bordering water is probably not agricultural"
and its converse genuinely carry different weight. `cond` below is the directed
object; `counts` and `pmi_bnd` are symmetric and that is not a defect.

OUTPUT (npz)
------------
    counts      (8,8) int64    summed adjacency, LoveDA indexing, 0 = unknown
    pix         (8,)  int64    pixel counts per class
    per_image   (T,8,8) int32  each tile's own matrix -> M_image, for the lambda
                               blend, so the sweep is instant and needs no re-read
    per_image_pix (T,8) int32
    names       (T,)  str      tile ids, aligned with per_image
    pmi_bnd     (8,8) float64  signed PMI, BOUNDARY marginals (the corrected one)
    pmi_area    (8,8) float64  same with AREA marginals -- kept only so the
                               superseded ANALYSIS 4 figures stay reproducible
    cond        (8,8) float64  P(column | neighbour = row), Dirichlet-smoothed
    alpha       scalar         smoothing pseudo-count actually used

    python scripts/build_m_global.py --source pred \
        --cache ~/outputs/week2_tau0.5_instrumented/cache \
        --tau 0.5 --out ~/outputs/week3/M_global_pred.npz

    python scripts/build_m_global.py --source gt \
        --cache ~/outputs/week2_tau0.5_instrumented/cache \
        --out ~/outputs/week3/M_global_gt.npz
"""
import argparse
import warnings
from pathlib import Path

import numpy as np

# LoveDA numbering. Index 0 is no-data in GT and "unknown" in predictions --
# excluded from adjacency either way, which is why the two are comparable.
LOVEDA = ['unknown', 'background', 'building', 'road',
          'water', 'barren', 'forest', 'agriculture']
NC = len(LOVEDA)

# measure_discard_rate.py's cache uses its own 0-indexed order for `pred`.
# pred index p corresponds to LoveDA label p + 1.
CACHE_CLASSES = ['background', 'building', 'road', 'water',
                 'barren', 'forest', 'agricultural']


def accumulate_one(lbl):
    """Adjacency counts (NC,NC) and pixel counts (NC,) for one label map.

    Identical to pmi_permutation_null.accumulate_one -- deliberately duplicated
    rather than imported so the two scripts cannot silently drift apart.
    """
    pix = np.bincount(lbl.ravel(), minlength=NC)[:NC]
    M = np.zeros((NC, NC), np.int64)
    a = np.concatenate([lbl[:, :-1].ravel(), lbl[:-1, :].ravel()])
    b = np.concatenate([lbl[:, 1:].ravel(), lbl[1:, :].ravel()])
    keep = (a > 0) & (b > 0) & (a != b)          # drop unknown, drop same-class
    if keep.any():
        c = np.bincount(a[keep] * NC + b[keep],
                        minlength=NC * NC).reshape(NC, NC)
        M = c + c.T                              # a boundary has no direction
    return M, pix


def pmi_from(M, pix, valid, marginal='boundary', alpha=1.0):
    """Signed PMI over the `valid` class indices.

    marginal='boundary'  p_i from the class's own total boundary length (row
                         sums). Same measure on both sides of the ratio, so a
                         class earns nothing for merely having lots of
                         perimeter. THE CORRECTED STATISTIC -- quote this.
    marginal='area'      p_i from pixel counts. What ANALYSIS 4 originally used;
                         mixes a boundary observation with an area expectation
                         and inflates thin classes. Kept for reproducibility
                         only. See CLAUDE.md, settled decisions.

    alpha: the SAME Dirichlet pseudo-count used for the conditional, and it is
    load-bearing rather than cosmetic. An unobserved pair gives P_obs = 0 ->
    log2(0) = -inf, and clamping that to 0.0 would report the strongest possible
    exclusion as "indistinguishable from chance". building-water is exactly such
    a pair (ANALYSIS 4.1: the near-hard constraint), so the clamp would silently
    delete the single most reliable fact in the matrix. With alpha > 0 there are
    no structural zeros and an unobserved pair lands at a large negative value
    bounded by the corpus size, which is the honest reading.

    alpha=0 reproduces cooccurrence_gt.py / ANALYSIS 4 exactly, at the cost of
    that failure mode. Use it only to check reproduction.
    """
    Mv = M[np.ix_(valid, valid)].astype(float)
    if Mv.sum() == 0:
        return np.full((len(valid), len(valid)), np.nan)
    Mv = Mv + alpha
    np.fill_diagonal(Mv, 0.0)              # self-adjacency is never counted
    P_obs = Mv / Mv.sum()
    pv = Mv.sum(1) if marginal == 'boundary' else pix[valid].astype(float)
    if pv.sum() == 0:
        return np.full((len(valid), len(valid)), np.nan)
    p = pv / pv.sum()
    P_exp = np.outer(p, p)
    np.fill_diagonal(P_exp, 0.0)
    if P_exp.sum() > 0:
        P_exp /= P_exp.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        PMI = np.log2(np.where(P_exp > 0, P_obs / P_exp, np.nan))
    PMI[~np.isfinite(PMI)] = 0.0
    return PMI


def conditional(M, valid, alpha):
    """P(column | neighbour = row), with add-alpha Dirichlet smoothing.

    This is the DIRECTED object -- see the module docstring. Smoothing matters:
    an unsmoothed zero says "this pair is impossible", which for a corpus-mined
    prior is never a claim we can support from finitely many tiles.
    """
    Mv = M[np.ix_(valid, valid)].astype(float) + alpha
    np.fill_diagonal(Mv, 0.0)                    # self-adjacency is not counted
    rows = Mv.sum(1, keepdims=True)
    rows[rows == 0] = 1.0
    return Mv / rows


def label_map_from_cache(z, source, tau):
    """Return a LoveDA-numbered label map, 0 = excluded."""
    gt = z['gt'].astype(np.uint8)                # 0 no-data, 1 bg, 2..7 real
    if source == 'gt':
        return gt
    # float16 cache -> float32 before any threshold comparison. At tau=0.5 this
    # is worth ~0.03% of pixels (WEEK1_RESULTS 7.7 caveat).
    conf = z['conf'].astype(np.float32)
    pred = z['pred'].astype(np.int16)
    lbl = (pred + 1).astype(np.uint8)            # cache 0-index -> LoveDA 1-index
    lbl[conf < tau] = 0                          # below tau = UNKNOWN, not bg
    lbl[gt == 0] = 0                             # no-data is never evidence
    return lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True,
                    help='dir of .npz from measure_discard_rate.py')
    ap.add_argument('--source', choices=['pred', 'gt'], default='pred',
                    help="'pred' mines SAM 3's confident output (annotation-free); "
                         "'gt' builds the reference matrix for the validation gate")
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--alpha', type=float, default=1.0,
                    help='Dirichlet pseudo-count for the conditional')
    ap.add_argument('--drop', nargs='+', default=[], metavar='CLASS',
                    help="exclude classes entirely, e.g. --drop background. "
                         "ANALYSIS 4's published PMI_bnd figures are computed "
                         "with background dropped, so use this to compare.")
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--md', default=None, help='also write a markdown summary')
    args = ap.parse_args()

    warnings.filterwarnings('ignore', message='All-NaN slice encountered')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz files under {args.cache}')

    print(f'{len(files)} tiles | source = {args.source}'
          + (f' | tau = {args.tau}' if args.source == 'pred' else '')
          + f' | alpha = {args.alpha}\n')

    counts = np.zeros((NC, NC), np.int64)
    pix = np.zeros(NC, np.int64)
    per_image = np.zeros((len(files), NC, NC), np.int32)
    per_image_pix = np.zeros((len(files), NC), np.int32)
    names = []
    empty = 0

    for i, f in enumerate(files):
        z = np.load(f)
        lbl = label_map_from_cache(z, args.source, args.tau)
        M, p = accumulate_one(lbl)
        counts += M
        pix += p
        per_image[i] = M
        per_image_pix[i] = p
        names.append(f.stem)
        if M.sum() == 0:
            empty += 1                            # no confident boundary at all
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}')

    drop = {d.lower() for d in args.drop}
    bad = [d for d in args.drop if d.lower() not in {x.lower() for x in LOVEDA[1:]}]
    if bad:
        raise SystemExit(f'unknown class(es) {bad}; known: {LOVEDA[1:]}')
    valid = [c for c in range(1, NC)
             if pix[c] > 0 and LOVEDA[c].lower() not in drop]
    if len(valid) < 2:
        raise SystemExit(f'only {len(valid)} class(es) left after --drop')
    nv = [LOVEDA[c] for c in valid]
    pmi_b = pmi_from(counts, pix, valid, 'boundary', args.alpha)
    pmi_a = pmi_from(counts, pix, valid, 'area', args.alpha)
    cond = conditional(counts, valid, args.alpha)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, counts=counts, pix=pix,
        per_image=per_image, per_image_pix=per_image_pix,
        names=np.array(names), valid=np.array(valid),
        class_names=np.array(nv),
        pmi_bnd=pmi_b, pmi_area=pmi_a, cond=cond,
        alpha=np.array(args.alpha), tau=np.array(args.tau),
        source=np.array(args.source),
    )

    w = max(len(x) for x in nv) + 1
    md = [f'# M_global — source `{args.source}`'
          + (f'  (without {", ".join(args.drop)})' if args.drop else '') + '\n',
          f'- tiles: **{len(files)}**'
          + (f'  |  τ: **{args.tau}**' if args.source == 'pred' else ''),
          f'- α (Dirichlet): **{args.alpha}**',
          f'- boundary pixel-pairs counted: **{counts.sum() // 2:,}**',
          f'- tiles with no confident boundary at all: **{empty}**'
          f' ({100 * empty / len(files):.1f}%)\n']

    if args.source == 'pred':
        used = int(pix[1:].sum())
        tot = int(pix.sum())
        md.append(f'- pixels the model committed to: **{used:,}** of {tot:,} '
                  f'(**{100 * used / max(tot, 1):.1f}%**) — the rest are below τ '
                  'and contribute nothing.\n')

    md += ['## Class share of counted boundary\n',
           '| class | boundary share | area share |', '|---|---|---|']
    bshare = counts[np.ix_(valid, valid)].sum(1)
    bshare = bshare / max(bshare.sum(), 1)
    ashare = pix[valid] / max(pix[valid].sum(), 1)
    for k, nm in enumerate(nv):
        md.append(f'| {nm} | {100 * bshare[k]:.1f}% | {100 * ashare[k]:.1f}% |')
    md.append('\nA class whose boundary share far exceeds its area share is thin and '
              'high-perimeter — exactly the confound `PMI_bnd` exists to remove.\n')

    md += ['## Signed PMI, boundary marginals (`PMI_bnd`)\n',
           '| |' + '|'.join(f' {x[:6]} ' for x in nv) + '|',
           '|---|' + '---|' * len(nv)]
    for k, nm in enumerate(nv):
        md.append(f'| **{nm}** |' + '|'.join(
            ' . ' if k == j else f' {pmi_b[k, j]:+.2f} ' for j in range(len(nv))) + '|')

    off = ~np.eye(len(nv), dtype=bool)
    md.append(f'\nMean |PMI_bnd| off-diagonal: **{np.abs(pmi_b[off]).mean():.3f} bits**\n')

    pairs = sorted((pmi_b[i, j], nv[i], nv[j])
                   for i in range(len(nv)) for j in range(i + 1, len(nv)))
    md.append('| strongest attractions | | strongest avoidances | |')
    md.append('|---|---|---|---|')
    for (va, a1, a2), (vd, d1, d2) in zip(pairs[::-1][:5], pairs[:5]):
        md.append(f'| {a1}–{a2} | **{va:+.2f}** | {d1}–{d2} | **{vd:+.2f}** |')

    md += ['\n## Discriminability weights — recomputed on `PMI_bnd`\n',
           'ANALYSIS §4.3 is REFUTED; the old weights were calibrated on `road`, '
           'whose row was a perimeter artefact. `w(n) ∝ Var_c[PMI_bnd(n, c)]`.\n',
           '| neighbour class | row variance | weight (normalised) |', '|---|---|---|']
    rowvar = np.array([np.var(pmi_b[k, off[k]]) for k in range(len(nv))])
    wn = rowvar / max(rowvar.sum(), 1e-12)
    for k in np.argsort(-rowvar):
        md.append(f'| {nv[k]} | {rowvar[k]:.3f} | {wn[k]:.3f} |')
    md.append('\nLow variance = a hub: bordering it barely narrows the vocabulary, '
              'so it should contribute little. High variance = exclusive and '
              'informative.\n')

    md += [f'## Directed conditional `P(column | neighbour = row)`  (α={args.alpha})\n',
           'Counts are symmetric by construction — a shared boundary has no '
           'direction. Directedness lives here, in the row normalisation, because '
           'the class marginals differ.\n',
           '| |' + '|'.join(f' {x[:6]} ' for x in nv) + '|',
           '|---|' + '---|' * len(nv)]
    for k, nm in enumerate(nv):
        md.append(f'| **{nm}** |' + '|'.join(
            ' . ' if k == j else f' {cond[k, j]:.3f} ' for j in range(len(nv))) + '|')
    asym = np.abs(cond - cond.T)[off]
    md.append(f'\nMean |P(c\\|n) − P(n\\|c)|: **{asym.mean():.3f}**, max '
              f'**{asym.max():.3f}** — the measured size of the asymmetry a '
              'symmetric M could not express.\n')

    text = '\n'.join(md)
    print('\n' + text)
    print(f'\nwritten: {out}')
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'written: {p}')


if __name__ == '__main__':
    main()
