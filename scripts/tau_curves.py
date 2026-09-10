"""
IoU against its OWN threshold, one curve per class -- and the proof that the
six curves are independent.

WHY. Every threshold result in this project is reported as a fitted vector and a
delta. Neither shows the reader what the optimiser was looking at. The question
that prompted this (a project review, 10 Sep) was exactly right: if the other six
thresholds are held constant while one is swept, what does that curve look like,
is it skewed, and can the search be made cheaper?

⭐ THE ANSWER, WHICH IS SHARPER THAN A PICTURE. With the argmax fixed, lowering
tau_c moves pixels between class c and the catch-all and nowhere else. Class d's
TP, FP and FN are all untouched, so:

    IoU_d is EXACTLY invariant to tau_c, for every real class d != c.

Two consequences the paper should carry:

  (1) the fit is SIX ONE-VS-BACKGROUND problems, not one seven-way problem. They
      are coupled only through the single catch-all class.
  (2) under `--objective real` -- the objective tau_cv/tau_deploy actually deploy
      -- the catch-all is excluded, so the objective separates and ONE sweep is
      already the global optimum: O(C*T) = 6*201 evaluations, no iteration.
      Against brute force over the joint grid (201^6 ~ 6.6e13) that is the whole
      complexity question, answered exactly rather than heuristically.

⚠️ Under FULL mIoU the catch-all is in the mean, so the classes genuinely do
couple and the iteration earns its place. `tau_oracle.py`'s +1.46 bound uses full
mIoU; `tau_cv.py`, `tau_deploy.py` and `precision_proxy.py` use `real`. WEEK3
§9d's stated reason for the label-free bound ("the objective is coupled") is
therefore right for the oracle and WRONG for the deployed fit -- there the reason
is simply that the peak of an IoU curve cannot be found without labels. This
script checks the claim on real data instead of asserting it.

WHAT IT DRAWS, per real class, on the SAME calibration/held-out split as the
deployment run:

    dashed  IoU_c on the CALIBRATION tiles      <- what the fit sees
    solid   IoU_c on the HELD-OUT tiles         <- what the reader gets
    lines   the published tau, the fitted tau, and the held-out argmax

⭐ The gap between the dashed peak and the solid peak IS the generalisation
error, per class. On LoveDA it is what `road` losing 0.53 in the deployment run
looks like (WEEK3 §9c), and it is visible rather than argued.

Plus two panels for the coupling that does exist: the catch-all's IoU, and full
mIoU, each against every class's threshold.

CPU only, no GPU, reads the cache. ⚠️ `per_tile_hists` holds one histogram per
tile (~650 MB for LoveDA val); it is reused rather than reimplemented so the
binning is bit-identical to the fit being illustrated.

    python scripts/tau_curves.py --cache ~/outputs/week3_fused/cache --tau 0.5 \
        --calib 200 --seed 0 --outdir docs --md ~/outputs/week3/tau_curves.md
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, miou, per_class_iou, NBINS   # noqa: E402
from tau_cv import per_tile_hists, fit                            # noqa: E402

# WEEK3 §9c -- the deployment fit that eval.py reproduced end to end (LoveDA,
# seed 0, 200 calibration tiles, objective `real`). Checked, not assumed: a
# figure that has drifted from the table it illustrates is worse than no figure.
REFERENCE_LOVEDA = {'water': 0.175, 'building': 0.190, 'barren': 0.375,
                    'forest': 0.410, 'agricultural': 0.565, 'road': 0.675}


def sweep(H, taus, c, bg, grid, nc):
    """IoU of every class, and full mIoU, as tau_c walks the grid.

    Returns (per_class[T, nc], miou_full[T]). Only tau[c] moves; the rest stay
    at `taus`, which is the whole point of the exercise.
    """
    pc = np.full((len(grid), nc), np.nan)
    mf = np.zeros(len(grid))
    t = np.asarray(taus, float).copy()
    for i, v in enumerate(grid):
        t[c] = v
        C = confusion_at(H, t, bg, NBINS)
        pc[i] = per_class_iou(C)
        mf[i] = miou(C)
    return pc, mf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True, help='the published τ')
    ap.add_argument('--calib', type=int, default=200,
                    help='calibration tiles; 200 matches the deployment run')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--objective', choices=['all', 'real'], default='real',
                    help='what the FIT maximises. The curves are drawn either way; '
                         'the separability result applies to `real`.')
    ap.add_argument('--outdir', default='docs')
    ap.add_argument('--name', default='fig10_tau_curves')
    ap.add_argument('--md', default=None)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    real = [c for c in range(nc) if c != bg]
    print(f'  classes: {LB}')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    n = len(files)
    if args.calib >= n:
        sys.exit(f'!! --calib {args.calib} but only {n} tiles in the cache')
    print(f'{n} tiles | published τ = {args.tau} | {NBINS} bins | '
          f'objective `{args.objective}`\n')

    PT = per_tile_hists(files, nc, NBINS)
    idx = np.random.default_rng(args.seed).permutation(n)
    cal, hel = idx[:args.calib], idx[args.calib:]
    Hcal = PT[cal].sum(0).astype(np.int64)
    Hhel = PT[hel].sum(0).astype(np.int64)
    print(f'\ncalibration {len(cal)} tiles | held out {len(hel)} tiles '
          f'(disjoint, seed {args.seed})')

    grid = np.arange(NBINS + 1) / NBINS
    taus = fit(Hcal, bg, NBINS, objective=args.objective)
    base = np.full(nc, args.tau)

    # ---------------------------------------------------------------- the sweeps
    # Every curve is computed with the OTHER thresholds at the fitted vector, so
    # the picture is a slice through the solution the method actually deploys.
    curves = {}
    for c in real:
        pc_cal, _ = sweep(Hcal, taus, c, bg, grid, nc)
        pc_hel, mf_hel = sweep(Hhel, taus, c, bg, grid, nc)
        curves[c] = (pc_cal, pc_hel, mf_hel)

    # ------------------------------------------------- CHECK 1: separability
    # For each swept class, how far does any OTHER real class's IoU move? The
    # algebra says exactly zero. This is that claim, on real data.
    print('\nseparability — max |Δ IoU| of any OTHER real class while τ_c sweeps '
          'the whole grid:')
    worst = 0.0
    sep_rows = []
    for c in real:
        _, pc_hel, _ = curves[c]
        others = [d for d in real if d != c]
        dev = 0.0
        for d in others:
            col = pc_hel[:, d]
            if np.isfinite(col).any():
                dev = max(dev, float(np.nanmax(col) - np.nanmin(col)))
        bgdev = float(np.nanmax(pc_hel[:, bg]) - np.nanmin(pc_hel[:, bg]))
        sep_rows.append((LB.names[c], dev, bgdev))
        worst = max(worst, dev)
        print(f'  {LB.names[c]:<14} other real classes {dev:.6f}   '
              f'catch-all moves {bgdev:.3f}')
    verdict_sep = ('EXACT — the six real classes are independent given the argmax; '
                   'all coupling runs through the catch-all'
                   if worst < 1e-9 else
                   f'⛔ NOT exact — {worst:.6f}. The separability argument is wrong on '
                   f'this cache and WEEK3 §9d\'s coupling wording stands as written.')
    print(f'  → {verdict_sep}')

    # --------------------------------------------- CHECK 2: does iteration pay?
    # If the objective separates, one sweep is already the global optimum and the
    # extra rounds cannot move a single threshold.
    t1 = fit(Hcal, bg, NBINS, rounds=1, objective=args.objective)
    t6 = fit(Hcal, bg, NBINS, rounds=6, objective=args.objective)
    t1a = fit(Hcal, bg, NBINS, rounds=1, objective='all')
    t6a = fit(Hcal, bg, NBINS, rounds=6, objective='all')
    d_obj = float(np.abs(t1 - t6).max())
    d_all = float(np.abs(t1a - t6a).max())
    print(f'\nrounds=1 vs rounds=6:  objective `{args.objective}` max |Δτ| = '
          f'{d_obj:.4f}   |   objective `all` max |Δτ| = {d_all:.4f}')
    print(f'  one sweep costs {len(real) * len(grid)} evaluations; '
          f'the joint grid would cost {len(grid)}^{len(real)}')

    # ------------------------------------------------------- per-class summary
    rows = []
    for c in real:
        pc_cal, pc_hel, _ = curves[c]
        i_cal = int(np.nanargmax(pc_cal[:, c])) if np.isfinite(pc_cal[:, c]).any() else 0
        i_hel = int(np.nanargmax(pc_hel[:, c])) if np.isfinite(pc_hel[:, c]).any() else 0
        j_pub = int(np.rint(args.tau * NBINS))
        j_fit = int(np.rint(taus[c] * NBINS))
        rows.append(dict(name=LB.names[c], c=c,
                         t_fit=taus[c], t_cal=grid[i_cal], t_hel=grid[i_hel],
                         iou_pub=pc_hel[j_pub, c], iou_fit=pc_hel[j_fit, c],
                         iou_hel=pc_hel[i_hel, c]))

    print('\n| class | published τ | fitted τ | held-out best τ | IoU @pub | '
          '@fitted | @best |')
    for r in rows:
        print(f"  {r['name']:<14} {args.tau:.3f}   {r['t_fit']:.3f}   "
              f"{r['t_hel']:.3f}   {r['iou_pub']:6.2f}  {r['iou_fit']:6.2f}  "
              f"{r['iou_hel']:6.2f}")

    # A named reference check, where one exists for this protocol.
    ref_note = None
    if (set(REFERENCE_LOVEDA) <= set(LB.names) and abs(args.tau - 0.5) < 1e-9
            and args.calib == 200 and args.seed == 0 and args.objective == 'real'):
        diffs = {k: taus[LB.names.index(k)] - v for k, v in REFERENCE_LOVEDA.items()}
        worst_k = max(diffs, key=lambda k: abs(diffs[k]))
        ok = abs(diffs[worst_k]) < 1e-9
        ref_note = (f'reference check vs WEEK3 §9c: '
                    + ('✅ every threshold identical'
                       if ok else
                       f'⚠️ largest disagreement `{worst_k}` {diffs[worst_k]:+.3f} — '
                       f'the cache or the split is not the one §9c used'))
        print(f'\n{ref_note}')

    # -------------------------------------------------------------- the figure
    panels = len(real) + 2
    ncols = min(4, panels)
    nrows = int(np.ceil(panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap('tab10')

    for k, c in enumerate(real):
        ax = axes[k]
        pc_cal, pc_hel, _ = curves[c]
        col = cmap(k % 10)
        ax.plot(grid, pc_cal[:, c], ls='--', lw=1.2, color=col, alpha=0.75,
                label=f'calibration ({len(cal)})')
        ax.plot(grid, pc_hel[:, c], ls='-', lw=1.8, color=col,
                label=f'held out ({len(hel)})')
        ax.axvline(args.tau, color='0.45', ls=':', lw=1.2)
        ax.axvline(taus[c], color=col, ls='-', lw=1.0, alpha=0.6)
        r = rows[k]
        ax.plot([r['t_hel']], [r['iou_hel']], marker='v', ms=6, color='0.25')
        ax.set_title(f"{LB.names[c]}   fitted {taus[c]:.3f}", fontsize=10)
        ax.set_xlabel(r'$\tau_c$'); ax.set_ylabel('IoU (%)')
        ax.set_xlim(0, 1); ax.grid(alpha=0.25)
        if k == 0:
            ax.legend(fontsize=7, loc='lower center')

    # coupling panel 1 -- the catch-all is the only class that moves
    ax = axes[len(real)]
    for k, c in enumerate(real):
        _, pc_hel, _ = curves[c]
        ax.plot(grid, pc_hel[:, bg], lw=1.3, color=cmap(k % 10), label=LB.names[c])
    ax.axvline(args.tau, color='0.45', ls=':', lw=1.2)
    ax.set_title(f'{LB.names[bg]} (catch-all) IoU\nvs each class\'s τ', fontsize=10)
    ax.set_xlabel(r'$\tau_c$'); ax.set_ylabel('IoU (%)')
    ax.set_xlim(0, 1); ax.grid(alpha=0.25); ax.legend(fontsize=6, ncol=2)

    # coupling panel 2 -- and therefore the reported metric moves too
    ax = axes[len(real) + 1]
    for k, c in enumerate(real):
        _, _, mf = curves[c]
        ax.plot(grid, mf, lw=1.3, color=cmap(k % 10), label=LB.names[c])
    ax.axvline(args.tau, color='0.45', ls=':', lw=1.2)
    ax.set_title('full mIoU vs each class\'s τ\n(couples only via the catch-all)',
                 fontsize=10)
    ax.set_xlabel(r'$\tau_c$'); ax.set_ylabel('mIoU')
    ax.set_xlim(0, 1); ax.grid(alpha=0.25)

    for ax in axes[panels:]:
        ax.axis('off')
    fig.suptitle('Per-class IoU against its own threshold, other thresholds held '
                 'at the fitted vector', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.outdir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = out / f'{args.name}.{ext}'
        fig.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f'\nwrote {p}')

    # ------------------------------------------------------------------ the md
    if args.md:
        md = [f'# Per-class IoU vs its own threshold\n',
              f'- cache: `{args.cache}`  |  tiles: **{n}**  |  bins: **{NBINS}**  '
              f'|  grid: **{len(grid)}** values',
              f'- calibration **{len(cal)}** / held out **{len(hel)}**, disjoint, '
              f'seed **{args.seed}**  |  published τ **{args.tau}**  |  fit '
              f'objective **`{args.objective}`**\n',
              'Each curve sweeps one class\'s threshold with the other classes held '
              'at the fitted vector.\n',
              '## Separability — does τ_c touch any other real class?\n',
              '| swept class | max \\|Δ IoU\\| over other real classes | catch-all '
              'IoU range |', '|---|---|---|']
        for name, dev, bgdev in sep_rows:
            md.append(f'| {name} | {dev:.6f} | {bgdev:.2f} |')
        md += [f'\n**{verdict_sep}**\n',
               f'One sweep costs **{len(real) * len(grid)}** confusion evaluations; '
               f'the joint grid would cost {len(grid)}^{len(real)}. '
               f'rounds=1 vs rounds=6 under `{args.objective}`: max |Δτ| '
               f'**{d_obj:.4f}**; under `all`: **{d_all:.4f}**.\n',
               '## Per class, on the held-out tiles\n',
               '| class | published τ | fitted τ | calib best τ | held-out best τ | '
               'IoU @ published | @ fitted | @ held-out best |',
               '|---|---|---|---|---|---|---|---|']
        for r in rows:
            md.append(f"| {r['name']} | {args.tau:.3f} | **{r['t_fit']:.3f}** | "
                      f"{r['t_cal']:.3f} | {r['t_hel']:.3f} | {r['iou_pub']:.2f} | "
                      f"**{r['iou_fit']:.2f}** | {r['iou_hel']:.2f} |")
        md.append('\n⚠️ `calib best τ` and `held-out best τ` differing is the '
                  'generalisation error, per class, and it is why a class can lose '
                  'IoU under a fit that improved the mean (WEEK3 §9c, `road`).\n')
        if ref_note:
            md.append(f'{ref_note}\n')
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('\n'.join(md))
        print(f'wrote {p}')


if __name__ == '__main__':
    main()
