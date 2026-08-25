"""
Week 3 — at what threshold should M_global be MINED?

WHY THIS EXISTS. Gate 1 failed on the first attempt: M mined from SAM 3's
confident predictions at tau=0.5 ranks class pairs no better than chance against
the ground-truth matrix (Spearman -0.110 over 42 pairs, 18 sign flips). The
diagnosis was structural, not a coding error:

  * A boundary is counted only when BOTH sides are confident. At tau=0.5 that
    captured 4.59M of the GT's 23.37M boundary pixel-pairs -- 19.6% of the
    adjacency graph -- and 28.8% of tiles contributed no boundary at all.
  * `background` is almost absent. P_final(background) = P_fused * S_pres and
    median S_pres(background) is 0.022 (WEEK1_RESULTS 9.2b), so background
    essentially never clears tau. It is 40.8% of GT boundary and ~3.3% of it
    survives into the mined matrix.
  * Losing background REWIRES the graph, it does not merely shrink it. In GT,
    building-road is -3.15 (they avoid: LoveDA labels the pavement between them
    `background`). With background gone the two look directly adjacent and the
    mined value flips to +0.86. Same story for building-forest.

The MINING threshold and the INFERENCE threshold are independent choices --
nothing forces M to be built at the tau the baseline runs at. Lower tau trades
label purity for coverage of the adjacency graph. Which side wins is an
empirical question, and the cache makes every tau free.

    tau -> 0   pure argmax, full coverage, noisier labels
    tau -> 1   very pure labels, almost no observations

This sweeps it and scores each against the GT matrix. The number to read is
Spearman: the scoring function consumes the RANKING of pairs, not absolute bits.

    python scripts/sweep_mining_tau.py \
        --cache ~/outputs/week2_tau0.5_instrumented/cache \
        --gt ~/outputs/week3/M_global_gt.npz \
        --md ~/outputs/week3/mining_tau_sweep.md

Note the two rows are different questions. `all classes` asks whether M is usable
as-is. `real only` (background dropped from BOTH sides) asks whether the real-class
structure is sound once background's absence is taken off the table -- which is
what matters if background ends up handled outside the prior.
"""
import argparse
from pathlib import Path

import numpy as np

from build_m_global import LOVEDA, NC, accumulate_one, pmi_from


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x)
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--gt', required=True, help='M_global_gt.npz from build_m_global.py')
    ap.add_argument('--taus', type=float, nargs='+',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    G = np.load(Path(args.gt).expanduser(), allow_pickle=True)
    gt_counts, gt_pix = G['counts'], G['pix']

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')

    taus = sorted(args.taus)
    counts = {t: np.zeros((NC, NC), np.int64) for t in taus}
    pix = {t: np.zeros(NC, np.int64) for t in taus}
    empty = {t: 0 for t in taus}

    print(f'{len(files)} tiles | mining τ = {taus}\n')
    for i, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.uint8)
        conf = z['conf'].astype(np.float32)      # float16 cache -> f32 before compare
        base = (z['pred'].astype(np.int16) + 1).astype(np.uint8)
        nodata = gt == 0
        for t in taus:
            lbl = base.copy()
            if t > 0:
                lbl[conf < t] = 0
            lbl[nodata] = 0
            M, p = accumulate_one(lbl)
            counts[t] += M
            pix[t] += p
            if M.sum() == 0:
                empty[t] += 1
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}')

    def score(t, drop_bg):
        vg = [c for c in range(1, NC) if gt_pix[c] > 0]
        vp = [c for c in range(1, NC) if pix[t][c] > 0]
        v = [c for c in vg if c in vp]
        if drop_bg:
            v = [c for c in v if LOVEDA[c] != 'background']
        if len(v) < 3:
            return None
        a = pmi_from(counts[t], pix[t], v, 'boundary', args.alpha)
        b = pmi_from(gt_counts, gt_pix, v, 'boundary', args.alpha)
        off = ~np.eye(len(v), dtype=bool)
        return dict(rho=spearman(a[off], b[off]),
                    flips=int(((a[off] > 0) != (b[off] > 0)).sum()),
                    npairs=int(off.sum()),
                    mad=float(np.abs(a[off] - b[off]).mean()),
                    nclass=len(v))

    gt_bnd = int(gt_counts.sum() // 2)
    md = ['# Week 3 — at what τ should `M_global` be mined?\n',
          f'- tiles: **{len(files)}**  |  α: **{args.alpha}**',
          f'- GT reference: **{gt_bnd:,}** boundary pixel-pairs\n',
          'Mining τ and inference τ are independent. Lower τ buys coverage of the '
          'adjacency graph at the cost of label purity.\n',
          '| mining τ | coverage of GT boundary | tiles w/ no boundary | '
          'bg share of boundary | ρ all classes | ρ real only | sign flips (real) |',
          '|---|---|---|---|---|---|---|']

    gt_bg = gt_counts[1].sum() / max(gt_counts.sum(), 1)
    rows = []
    for t in taus:
        tot = int(counts[t].sum() // 2)
        cov = 100 * tot / max(gt_bnd, 1)
        bg = 100 * counts[t][1].sum() / max(counts[t].sum(), 1)
        sa, sr = score(t, False), score(t, True)
        rows.append((t, cov, sa, sr))
        md.append(
            f'| {t:.2f} | {cov:.1f}% ({tot:,}) | {empty[t]} '
            f'({100 * empty[t] / len(files):.1f}%) | {bg:.1f}% | '
            + (f'**{sa["rho"]:+.3f}**' if sa else '—') + ' | '
            + (f'**{sr["rho"]:+.3f}**' if sr else '—') + ' | '
            + (f'{sr["flips"]}/{sr["npairs"]}' if sr else '—') + ' |')

    md.append(f'\nGT background share of boundary: **{100 * gt_bg:.1f}%** — the column '
              'above shows how much of that survives mining. ρ is Spearman against the '
              'GT matrix over class pairs; ≥0.7 is a usable prior, ≤0.35 is not.\n')

    ok = [r for r in rows if r[3] and r[3]['rho'] >= 0.7]
    best_r = max((r for r in rows if r[3]), key=lambda r: r[3]['rho'], default=None)
    best_a = max((r for r in rows if r[2]), key=lambda r: r[2]['rho'], default=None)

    md.append('## Verdict\n')
    if best_a and best_a[2]['rho'] >= 0.7:
        md.append(f'✅ **Mine at τ = {best_a[0]:.2f}** — ρ = {best_a[2]["rho"]:+.3f} '
                  'across all classes including background. Threshold starvation was '
                  'the whole problem; use this τ for mining and keep τ=0.5 at '
                  'inference. Say explicitly in the paper that the two thresholds are '
                  'separate hyperparameters.')
    elif best_r and best_r[3]['rho'] >= 0.7:
        md.append(f'⚠️ **Real-class structure is recoverable at τ = {best_r[0]:.2f}** '
                  f'(ρ = {best_r[3]["rho"]:+.3f}), but background never becomes '
                  'mineable at any τ — `P_final(bg) ≤ S_pres(bg) ≈ 0.022`, so it '
                  'cannot clear a meaningful threshold. **Consequence: background '
                  'must be handled OUTSIDE the co-occurrence prior** — as an '
                  '"unknown/none-of-the-above" outcome rather than a class with '
                  'adjacency statistics. That is a method decision, and this table '
                  'is the evidence for it.')
    elif best_r:
        md.append(f'⛔ **No τ yields a usable prior.** Best is τ = {best_r[0]:.2f} at '
                  f'ρ = {best_r[3]["rho"]:+.3f} over real classes. Mining M from '
                  "SAM 3's own thresholded output does not reproduce the adjacency "
                  'structure at pixel level — ANALYSIS §3.2\'s circularity concern, '
                  'confirmed and worse than expected. Options, in order of cost: '
                  '(a) bridge unknown gaps so two confident regions separated by a '
                  'band of discarded pixels still count as adjacent; (b) mine at '
                  'REGION level over SAM 3 masks rather than pixel level; '
                  '(c) accept a weaker `M_global` and lean on `M_image` + the λ '
                  'blend. Decide before writing the scoring function.')
    md.append('\n> Coverage is not the target — fidelity is. A τ that counts more '
              'boundary but ranks pairs worse is the wrong choice, so read ρ, not '
              'the coverage column.\n')

    if ok:
        md.append(f'Usable (ρ ≥ 0.7, real classes): '
                  f'{", ".join(f"τ={r[0]:.2f}" for r in ok)}')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
