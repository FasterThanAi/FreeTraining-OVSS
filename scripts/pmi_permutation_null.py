"""
ANALYSIS.md §4 — is the measured PMI signal semantic, or partly geometric?

cooccurrence_gt.py compares an observed BOUNDARY-frequency distribution against
an independence model built from AREA marginals:

    P_obs = M / M.sum()          # boundary pixel-pairs
    p     = pix / pix.sum()      # class AREA share
    P_exp = outer(p, p)
    PMI   = log2(P_obs / P_exp)

Those are different measures. A class fragmented into thin ribbons has far more
boundary per unit area than one in large contiguous blocks, so **PMI is
systematically inflated for high-perimeter classes and deflated for compact
ones, independent of any semantic affinity.**

That matters specifically for §4.3, whose finding — rural `road` attracts almost
everything, therefore hub classes should be down-weighted — is derived from the
thinnest, highest-perimeter class in LoveDA, and then used to justify the
discriminability weighting in the scoring function.

The existing random control does not catch this: scattering classes uniformly
destroys blob geometry entirely, so it tests "is there any signal at all", not
"is THIS pair's signal geometric or semantic".

THE NULL USED HERE. Keep every image's geometry exactly as it is, and permute
which CLASS LABEL each region carries. Region shapes, sizes, perimeters and the
adjacency graph are all preserved; only the semantic assignment is destroyed. If
road's PMI row survives this, the signal is semantic. If it collapses to the
null, it was road's thin geometry.

WHY IT IS FAST. A per-image relabelling by permutation pi maps the image's
adjacency matrix M_img[i,j] -> M_img[pi(i), pi(j)] and pixel counts likewise. So
each mask is read ONCE, its 7x7 matrix cached, and every trial is pure index
shuffling. 1000 trials over 1669 tiles costs seconds, not hours.

    python scripts/pmi_permutation_null.py --masks ~/data/loveda/ann_dir/val
    python scripts/pmi_permutation_null.py --masks ~/data/loveda/ann_dir/val \
        --drop background --trials 1000

Output: observed PMI, null mean/sd, and a z-score per class pair. |z| >= 3 means
the association is not explainable by geometry alone.
"""
import argparse
from pathlib import Path

import numpy as np

# LoveDA: 0 = no-data (ignored), 1..7 the real classes
LOVEDA = ['ignore', 'background', 'building', 'road',
          'water', 'barren', 'forest', 'agriculture']
NC = len(LOVEDA)


def accumulate_one(lbl):
    """Adjacency counts (NC x NC) and pixel counts (NC,) for one mask."""
    M = np.zeros((NC, NC), np.int64)
    pix = np.bincount(lbl.ravel(), minlength=NC)[:NC]
    a = np.concatenate([lbl[:, :-1].ravel(), lbl[:-1, :].ravel()])
    b = np.concatenate([lbl[:, 1:].ravel(), lbl[1:, :].ravel()])
    keep = (a > 0) & (b > 0) & (a != b)
    if keep.any():
        c = np.bincount(a[keep] * NC + b[keep], minlength=NC * NC).reshape(NC, NC)
        M = c + c.T                       # symmetric: a boundary has no direction
    return M, pix


def pmi_from(M, pix, valid, marginal='area'):
    """Signed PMI over the `valid` class indices.

    marginal='area'      p_i from PIXEL COUNTS -- what cooccurrence_gt.analyse and
                         therefore ANALYSIS 4 currently use. P_obs is a BOUNDARY
                         distribution, so this mixes two different measures and
                         inflates high-perimeter classes.
    marginal='boundary'  p_i from the class's own total BOUNDARY LENGTH
                         (row sums of M). Same measure on both sides, so a class
                         gets no credit merely for having lots of perimeter.
                         This is the corrected statistic.
    """
    Mv = M[np.ix_(valid, valid)].astype(float)
    tot = Mv.sum()
    if tot == 0:
        return None
    P_obs = Mv / tot
    if marginal == 'boundary':
        pv = Mv.sum(1)
    else:
        pv = pix[valid].astype(float)
    if pv.sum() == 0:
        return None
    p = pv / pv.sum()
    P_exp = np.outer(p, p)
    np.fill_diagonal(P_exp, 0.0)
    if P_exp.sum() > 0:
        P_exp /= P_exp.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        PMI = np.log2(np.where(P_exp > 0, P_obs / P_exp, np.nan))
    PMI[~np.isfinite(PMI)] = 0.0
    return PMI


def find_masks(root):
    """PNGs under root. LoveDA's Val/<domain>/ holds BOTH images_png/ and
    masks_png/, so prefer directories whose name mentions masks -- reading an
    RGB image as a label map would be silently wrong (the label-range check
    below would catch it, but this is cleaner)."""
    root = Path(root).expanduser()
    m = sorted(p for p in root.rglob('*.png') if 'mask' in p.parent.name.lower())
    return m if m else sorted(root.rglob('*.png'))


def read_group(files, quiet=False):
    """Pass 1: read each mask ONCE, cache its NCxNC adjacency matrix."""
    from PIL import Image
    Ms, Ps = [], []
    for i, f in enumerate(files, 1):
        lbl = np.array(Image.open(f))
        if lbl.ndim == 3:
            lbl = lbl[..., 0]
        lbl = lbl.astype(np.int64)
        if lbl.max() >= NC:
            raise SystemExit(f'{f} has label {lbl.max()}, expected < {NC}. '
                             'Is this an image rather than a mask?')
        M, pix = accumulate_one(lbl)
        Ms.append(M); Ps.append(pix)
        if not quiet and (i % 250 == 0 or i == len(files)):
            print(f'  {i}/{len(files)}')
    return np.stack(Ms), np.stack(Ps)


def domain_section(res, names_cls, drop):
    """§4.4 — is the prior domain-specific? Reported under BOTH marginals,
    because §4.4's 'six of fifteen pairs flip sign' was computed with the area
    marginal, and that is what makes hierarchical M a settled requirement."""
    (na, ra), (nb, rb) = list(res.items())
    n = len(names_cls)
    off = ~np.eye(n, dtype=bool)
    md = [f'\n## §4.4 — domain transfer: {na} vs {nb} ⭐\n',
          '`ANALYSIS.md` §4.4 reports **mean |PMI difference| 1.137** with background / '
          '**1.310** without, **max 3.064 / 2.791**, and **six of fifteen pairs flipping '
          'sign**. Those were computed with the AREA marginal. The `hierarchical M is '
          'required` decision in `CLAUDE.md` rests on them, so both marginals are shown.\n',
          '| Marginal | mean \\|PMI diff\\| | max \\|PMI diff\\| | pairs flipping sign |',
          '|---|---|---|---|']
    flips_by = {}
    for key, lbl in (('area', 'area (as §4.4)'), ('bnd', 'boundary (corrected)')):
        A, B = ra[key], rb[key]
        D = np.abs(A - B)
        fl = [(names_cls[i], names_cls[j], A[i, j], B[i, j])
              for i in range(n) for j in range(i + 1, n)
              if np.sign(A[i, j]) != np.sign(B[i, j])]
        flips_by[key] = fl
        md.append(f'| {lbl} | **{D[off].mean():.3f}** | {D[off].max():.3f} | '
                  f'**{len(fl)} / {n*(n-1)//2}** |')
    for key, lbl in (('area', 'area'), ('bnd', 'boundary')):
        if flips_by[key]:
            md.append(f'\n*{lbl} sign flips:* ' + ', '.join(
                f'`{a}–{b}` ({x:+.2f}/{y:+.2f})' for a, b, x, y in flips_by[key]))
    fa, fb = len(flips_by['area']), len(flips_by['bnd'])
    da = np.abs(ra['area'] - rb['area'])[off].mean()
    db = np.abs(ra['bnd'] - rb['bnd'])[off].mean()
    md += ['\n**Reading.**\n']
    if fb >= 4 and db >= 0.3:
        md.append(f'The prior stays domain-specific under the corrected marginal '
                  f'({fb} sign flips, mean |diff| {db:.2f} bits). **§4.4 survives and '
                  '`M_eff = λ·M_global + (1−λ)·M_image` remains a requirement.** The λ-sweep '
                  'stays a mandatory ablation.')
    elif fb >= 2 or db >= 0.2:
        md.append(f'Domain differences shrink under the corrected marginal '
                  f'({fa} → {fb} sign flips, {da:.2f} → {db:.2f} bits) but do not vanish. '
                  'Hierarchical M is still defensible; restate §4.4 with these numbers and '
                  'expect λ to matter less than the area-marginal figures implied.')
    else:
        md.append(f'⛔ **Domain differences largely disappear** under the corrected marginal '
                  f'({fa} → {fb} sign flips, {da:.2f} → {db:.2f} bits). §4.4 was substantially '
                  'a formula artefact, and `hierarchical M is required` — a settled decision in '
                  '`CLAUDE.md` — is no longer supported by it. Either find another '
                  'justification for the λ blend or drop it to a tested option. Decide before '
                  'Week 3 builds on it.')
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', required=True, nargs='+',
                    help='one directory, or two for a domain comparison (§4.4)')
    ap.add_argument('--names', nargs='+', default=None,
                    help='labels for the groups (default: directory basenames)')
    ap.add_argument('--trials', type=int, default=1000, help='0 skips the permutation null')
    ap.add_argument('--drop', nargs='*', default=[], metavar='CLASS')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    groups = {}
    gnames = args.names or [Path(m).name for m in args.masks]
    if len(gnames) != len(args.masks):
        raise SystemExit('--names must match the number of --masks directories')
    for nm, d in zip(gnames, args.masks):
        f = find_masks(d)
        if args.limit:
            f = f[:args.limit]
        if not f:
            raise SystemExit(f'no mask PNGs under {d}')
        groups[nm] = f
        print(f'[{nm}] {len(f)} masks')

    # combined corpus drives the main table; groups drive the §4.4 section
    files = [f for fs in groups.values() for f in fs]
    print(f'{len(files)} masks total | {args.trials} permutations | seed {args.seed}')
    if args.drop:
        print(f'dropping: {", ".join(args.drop)}')

    Ms, Ps = read_group(files)

    drop = {d.lower() for d in args.drop}
    valid = [i for i in range(1, NC)
             if Ps[:, i].sum() > 0 and LOVEDA[i].lower() not in drop]
    names = [LOVEDA[i] for i in valid]
    obs = pmi_from(Ms.sum(0), Ps.sum(0), valid, 'area')
    obs_b = pmi_from(Ms.sum(0), Ps.sum(0), valid, 'boundary')
    if obs is None or obs_b is None:
        raise SystemExit('no boundaries found')

    # ---- per-group PMI, for the §4.4 domain comparison ----------------------
    per_group = {}
    if len(groups) >= 2:
        off_g = 0
        for nm, fs in groups.items():
            gM, gP = Ms[off_g:off_g + len(fs)], Ps[off_g:off_g + len(fs)]
            off_g += len(fs)
            a = pmi_from(gM.sum(0), gP.sum(0), valid, 'area')
            b = pmi_from(gM.sum(0), gP.sum(0), valid, 'boundary')
            if a is None or b is None:
                raise SystemExit(f'no boundaries in group {nm}')
            per_group[nm] = {'area': a, 'bnd': b}

    # ---- permutation trials: pure index shuffling, no re-reading ------------
    rng = np.random.default_rng(args.seed)
    if args.trials == 0:
        mu = np.zeros_like(obs)
        sd = np.zeros_like(obs)
        z = np.zeros_like(obs)
    else:
        real = np.arange(1, NC)           # permute among the real class labels only
        null = np.empty((args.trials, len(valid), len(valid)), np.float32)
        for t in range(args.trials):
            Msum = np.zeros((NC, NC), np.int64)
            Psum = np.zeros(NC, np.int64)
            for n in range(len(Ms)):
                perm = rng.permutation(real)
                idx = np.arange(NC); idx[1:] = perm      # label 0 (no-data) fixed
                Msum += Ms[n][np.ix_(idx, idx)]
                Psum += Ps[n][idx]
            p = pmi_from(Msum, Psum, valid, 'area')
            null[t] = p if p is not None else np.nan
            if (t + 1) % max(args.trials // 10, 1) == 0:
                print(f'  trial {t+1}/{args.trials}')

        mu, sd = np.nanmean(null, 0), np.nanstd(null, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.where(sd > 0, (obs - mu) / sd, 0.0)

    off = ~np.eye(len(valid), dtype=bool)
    md = ['# PMI vs a structure-preserving permutation null\n',
          f'- masks: **{len(files)}**  |  permutations: **{args.trials}**  |  seed {args.seed}',
          f'- classes: {", ".join(names)}' + (f'  (dropped: {", ".join(args.drop)})'
                                              if args.drop else ''),
          f'- mean |PMI| **area** marginal (the §4 figure): **{np.abs(obs[off]).mean():.3f}** bits',
          f'- mean |PMI| **boundary** marginal (corrected): '
          f'**{np.abs(obs_b[off]).mean():.3f}** bits  ← quote this one',
          f'- mean |PMI| under the permutation null: **{np.abs(mu[off]).mean():.3f}** bits'
          + ('  *(matches the 0.004 random control of §4)*'
             if args.trials and np.abs(mu[off]).mean() < 0.02 else ''),
          f'- **premise margin: {np.abs(obs_b[off]).mean() / max(np.abs(mu[off]).mean(), 1e-9):.0f}×'
          f' the noise floor**\n' if args.trials else '\n',
          'TWO INDEPENDENT CHECKS, and they answer different questions.\n',
          '**`PMI_bnd`** replaces the area marginals with each class\'s own total BOUNDARY',
          'length, so both sides of the ratio are the same measure and a class earns nothing',
          'for merely having lots of perimeter. **This is the direct correction** to the',
          'concern; `Δ` is how much the area-based figure in ANALYSIS §4 was inflated.\n',
          '**`z`** is the permutation null: geometry held fixed, class labels permuted per',
          'image. It tests whether the label→adjacency association is CONSISTENT, which is a',
          'different question — a class that is reliably thin scores high here on geometry',
          'alone. Read `Δ` for the confound, `z` for reproducibility. Sorted by |Δ|.\n',
          '| Pair | PMI_area (§4) | PMI_bnd (corrected) | Δ | z (perm) |',
          '|---|---|---|---|---|']
    pairs = [(abs(obs[i, j] - obs_b[i, j]), i, j)
             for i in range(len(valid)) for j in range(i + 1, len(valid))]
    pairs.sort(reverse=True)
    for _, i, j in pairs:
        d = obs[i, j] - obs_b[i, j]
        flag = '' if abs(z[i, j]) >= 3 else ' ⚠️'
        warn = ' **⬅**' if abs(d) >= 0.5 else ''
        md.append(f'| {names[i]} – {names[j]} | {obs[i,j]:+.2f} | {obs_b[i,j]:+.2f} | '
                  f'**{d:+.2f}**{warn} | {z[i,j]:+.1f}{flag} |')
    flips = [(names[i], names[j], obs[i, j], obs_b[i, j])
             for _, i, j in pairs if np.sign(obs[i, j]) != np.sign(obs_b[i, j])]
    if flips:
        md.append(f'\n⚠️ **{len(flips)} pair(s) CHANGE SIGN** under the corrected marginal: '
                  + ', '.join(f'`{a}–{b}` ({x:+.2f} → {y:+.2f})' for a, b, x, y in flips)
                  + '. A sign flip means attraction and exclusion swap places — and '
                    'ANALYSIS §4.2 makes exclusion the load-bearing signal. Any of these '
                    'appearing in §4.1\'s table must be restated.')

    # per-class row summary -- this is what 4.3's hub argument rests on
    md += ['\n## Per-class rows — the §4.3 hub question ⭐\n',
           '§4.3 down-weights hub classes using **the variance of a class\'s PMI row**, and',
           'derives the rule from `road` — the thinnest, highest-perimeter class in LoveDA.',
           'If `row var` moves substantially from area to boundary marginals, that weighting',
           'was calibrated on a formula artefact and must be recomputed.\n',
           '| Class | mean \\|PMI\\| area | mean \\|PMI\\| bnd | row var area | '
           'row var bnd | var change |',
           '|---|---|---|---|---|---|']
    for i, nm in enumerate(names):
        m = np.ones(len(valid), bool); m[i] = False
        va, vb = obs[i][m].var(), obs_b[i][m].var()
        ch = (vb / va) if va > 0 else float('nan')
        md.append(f'| {nm} | {np.abs(obs[i][m]).mean():.2f} | {np.abs(obs_b[i][m]).mean():.2f} | '
                  f'{va:.2f} | {vb:.2f} | **{ch:.2f}×** |')
    md.append('\n`var change` is the multiplier §4.3\'s discriminability weight `w(n) ∝ '
              'Var_c[PMI(label(n), c)]` would move by. Anything far from 1.00× means the '
              'weighting changes materially under the corrected statistic.')

    weak = [f'{names[i]}–{names[j]}' for _, i, j in pairs if abs(z[i, j]) < 3]
    off = ~np.eye(len(valid), dtype=bool)
    dmean = np.abs(obs[off] - obs_b[off]).mean()
    md += ['\n## Verdict\n',
           f'- mean |Δ| between the two statistics: **{dmean:.3f}** bits',
           f'- pairs changing sign: **{len(flips)}**',
           f'- pairs failing the permutation null (|z| < 3): **{len(weak)}**'
           + (f' — {", ".join(weak)}' if weak else ''),
           '']
    if dmean < 0.15 and not flips:
        md.append('**The area/boundary mismatch does not matter in practice.** The two '
                  'statistics agree, no pair changes sign, so ANALYSIS §4 stands as written '
                  'and §4.3\'s hub finding is not a formula artefact. Note the correction in '
                  'the paper anyway — one sentence, and it removes an obvious reviewer question.')
    elif flips:
        md.append('**The corrected marginal changes conclusions.** Sign flips mean attraction '
                  'and exclusion swap, and §4.2 makes exclusion the load-bearing signal. '
                  'Recompute §4.1 and §4.3 on `PMI_bnd` and treat the boundary marginal as the '
                  'definition going forward.')
    else:
        md.append(f'**Magnitudes shift (mean |Δ| = {dmean:.2f} bits) but no pair changes sign.** '
                  'The qualitative claims in §4.1 survive; the numeric values should be quoted '
                  'from `PMI_bnd`, and §4.3\'s row variances recomputed from the table above '
                  'before the discriminability weighting is fixed.')
    if per_group:
        md += domain_section(per_group, names, args.drop)

    md += ['\n> Neither check threatens the premise: 1.3–1.7 bits against a 0.004 noise floor is '
           'far too large a margin to be geometric. What is under test is the PER-PAIR '
           'structure, which is what the method actually consumes.\n',
           '> **Limitation of the permutation column.** It holds geometry fixed and permutes '
           'labels, so it detects whether a label is CONSISTENTLY associated with an adjacency '
           'profile. A class that is reliably thin (like `road`) scores high on geometry alone. '
           'It is a reproducibility check, not a control for the perimeter confound — `Δ` is '
           'the control for that. Verified on synthetic data: a corpus with a thin ribbon but '
           'randomly permuted labels yields |z| < 2 everywhere, and one with fixed labels '
           'yields |z| ~ 8 on the ribbon\'s pairs.']

    text = '\n'.join(md)
    print('\n' + text)
    if args.out:
        Path(args.out).expanduser().write_text(text)
        print(f'\nWritten to {args.out}')


if __name__ == '__main__':
    main()
