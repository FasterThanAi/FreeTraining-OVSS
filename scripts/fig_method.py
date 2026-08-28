"""
Figure 7 — the method. Three panels, one argument.

    (a) the fitted thresholds against the single global tau      <- WHY it works
    (b) per-class IoU delta, two protocols side by side          <- WHAT it buys
    (c) how many labelled tiles the calibration costs            <- WHETHER it is practical

Panel (b) deliberately plots BOTH protocols. They agree on the shape -- water
dominates, road is the one real loss -- and they disagree on `background`
(-0.01 under 5-fold, +0.85 in the single end-to-end fit). Showing one and hiding
the other would be the same mistake as quoting OpenEarthMap's +2.28 without its
per-class table.

EVERY NUMBER IS CITED to WEEK3_RESULTS.md and printed on render, so a figure that
has drifted from its source table is visible rather than silent.

    python scripts/fig_method.py --outdir docs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

C_DOWN, C_UP, C_GOOD, C_BAD, C_BG = '#1f6f8b', '#c47f17', '#2e8b57', '#c0392b', '#8e44ad'
TAU_PUB = 0.5

# WEEK3_RESULTS.md 9c -- the deployment fit (200 calibration tiles, seed 0),
# the one confirmed end-to-end by eval.py. Precision/recall are the baseline's
# own, WEEK1_RESULTS.md 5, full val split.
# (class, fitted tau, precision, recall)
THRESHOLDS = [
    ('water',        0.175, 89.5, 54.7),
    ('building',     0.190, 77.2, 78.6),
    ('barren',       0.375, 51.5, 53.9),
    ('forest',       0.410, 57.9, 44.8),
    ('agricultural', 0.565, 66.9, 62.0),
    ('road',         0.675, 69.6, 70.5),
]

# (class, 5-fold mean delta [9b], end-to-end measured delta [9c])
PER_CLASS = [
    ('water',        +6.78, +6.54),
    ('barren',       +0.98, +1.08),
    ('forest',       +0.37, +0.44),
    ('building',     +0.28, +0.17),
    ('road',         +0.10, -0.53),
    ('agricultural', -0.21, -0.26),
    ('background',   -0.01, +0.85),
]

# WEEK3_RESULTS.md 9b -- (calibration tiles, mean delta, sd, worst draw)
CURVE = [(10, -2.14, 1.99, -5.59), (50, +0.08, 0.97, -0.99), (100, +0.54, 0.71, -0.57),
         (200, +0.79, 0.35, +0.43), (400, +1.21, 0.16, +1.04)]

FOLDS = [+1.37, +1.89, +0.88, +0.80, +0.98]      # 9b, 5-fold
FOLD_MEAN, FOLD_SD, ORACLE = +1.18, 0.45, +1.46


def panel_a(ax):
    names = [c for c, _, _, _ in THRESHOLDS]
    tau = np.array([t for _, t, _, _ in THRESHOLDS])
    gap = np.array([p - r for _, _, p, r in THRESHOLDS])
    x = np.arange(len(names))

    ax.axhline(TAU_PUB, color='black', lw=1.6, ls='--', zorder=2)
    ax.annotate('published global τ = 0.5', xy=(len(x) - 0.45, TAU_PUB), xytext=(0, 6),
                textcoords='offset points', ha='right', va='bottom', fontsize=8.5,
                style='italic')
    for xi, t in zip(x, tau):
        c = C_DOWN if t < TAU_PUB else C_UP
        ax.plot([xi, xi], [TAU_PUB, t], color=c, lw=2.6, zorder=3, solid_capstyle='round')
        ax.plot(xi, t, 'o', color=c, ms=11, zorder=4, mec='black', mew=0.8)
        ax.annotate(f'{t:.3f}', xy=(xi, t), xytext=(0, -16 if t < TAU_PUB else 12),
                    textcoords='offset points', ha='center', fontsize=8.6, weight='bold',
                    color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n{"P−R " if i == 0 else ""}{g:+.0f}'.replace('-', '−')
                        for i, (n, g) in enumerate(zip(names, gap))], fontsize=8.8)
    ax.set_ylim(0.0, 0.82)
    ax.set_ylabel('confidence threshold τ', fontsize=9.5)
    ax.set_title('(a)  one global τ is wrong in opposite directions',
                 fontsize=10.4, weight='bold', loc='left')
    ax.text(0.02, 0.965, 'spread 0.175 – 0.675', transform=ax.transAxes,
            fontsize=8.6, va='top', style='italic', color='#444')
    ax.grid(axis='y', color='#dddddd', lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def panel_b(ax):
    names = [c for c, _, _ in PER_CLASS]
    a = np.array([v for _, v, _ in PER_CLASS])
    b = np.array([v for _, _, v in PER_CLASS])
    y = np.arange(len(names))[::-1]
    h = 0.36
    ax.barh(y + h / 2, a, height=h, color=[C_BG if n == 'background' else
                                           (C_GOOD if v > 0 else C_BAD)
                                           for n, v in zip(names, a)],
            edgecolor='black', lw=0.6, zorder=3, label='5-fold mean (§9b)')
    ax.barh(y - h / 2, b, height=h, color=[C_BG if n == 'background' else
                                           (C_GOOD if v > 0 else C_BAD)
                                           for n, v in zip(names, b)],
            edgecolor='black', lw=0.6, zorder=3, alpha=0.45,
            label='end-to-end run (§9c)')
    ax.axvline(0, color='black', lw=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f'$\\bf{{{n}}}$' if n == 'water' else n for n in names],
                       fontsize=9)
    for yi, n, v, w in zip(y, names, a, b):
        ax.annotate(f'{v:+.2f}', xy=(v, yi + h / 2), xytext=(4 if v >= 0 else -4, 0),
                    textcoords='offset points', va='center',
                    ha='left' if v >= 0 else 'right', fontsize=7.8)
        # label the faded series only where the two protocols genuinely disagree --
        # `road` and `background` are exactly the two the text calls out, and a
        # reader who sees only the solid bars would miss both
        if abs(w - v) > 0.3:
            ax.annotate(f'{w:+.2f}', xy=(w, yi - h / 2), xytext=(4 if w >= 0 else -4, 0),
                        textcoords='offset points', va='center', style='italic',
                        ha='left' if w >= 0 else 'right', fontsize=7.6, color='#555')
    ax.set_xlim(-1.6, 8.1)
    ax.set_xlabel('Δ IoU vs published τ', fontsize=9.5)
    ax.set_title('(b)  the gain is land cover, not the catch-all',
                 fontsize=10.4, weight='bold', loc='left')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#777', edgecolor='black', lw=0.6,
                             label='5-fold mean (§9b)'),
                       Patch(facecolor='#777', edgecolor='black', lw=0.6, alpha=0.45,
                             label='end-to-end run (§9c)')],
              fontsize=8, loc='lower right', framealpha=0.95,
              title='fill = protocol; colour = sign', title_fontsize=7.4)
    ax.grid(axis='x', color='#dddddd', lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)


def panel_c(ax):
    n = np.array([r[0] for r in CURVE], float)
    m = np.array([r[1] for r in CURVE])
    sd = np.array([r[2] for r in CURVE])
    worst = np.array([r[3] for r in CURVE])

    ax.axhspan(-6.2, 0, color='#c0392b', alpha=0.055, zorder=0)
    ax.axhline(0, color='black', lw=1.0, zorder=2)
    ax.axhline(ORACLE, color='#555', lw=1.3, ls=':', zorder=2)
    ax.annotate(f'oracle bound {ORACLE:+.2f}', xy=(n[0], ORACLE), xytext=(2, 4),
                textcoords='offset points', fontsize=8.2, style='italic', color='#555')
    ax.fill_between(n, m - sd, m + sd, color=C_DOWN, alpha=0.18, zorder=2)
    ax.plot(n, m, '-o', color=C_DOWN, lw=2.0, ms=6, zorder=4, label='mean Δ ± sd')
    ax.plot(n, worst, '--s', color=C_BAD, lw=1.3, ms=4.5, zorder=4, label='worst draw')

    i200 = list(n).index(200)
    ax.plot(200, worst[i200], 's', color=C_GOOD, ms=9, zorder=5, mec='black', mew=0.8)
    ax.annotate('200 tiles — every\ndraw turns positive', xy=(200, worst[i200]),
                xytext=(4, -36), textcoords='offset points', fontsize=8.3,
                ha='center', weight='bold', color=C_GOOD)
    ax.set_xscale('log')
    ax.set_xticks(n)
    ax.set_xticklabels([f'{int(v)}' for v in n], fontsize=8.8)
    ax.minorticks_off()
    ax.set_xlabel('labelled calibration tiles', fontsize=9.5)
    ax.set_ylabel('Δ mIoU on held-out tiles', fontsize=9.5)
    ax.set_ylim(-6.2, 2.3)
    ax.set_title('(c)  what the calibration costs', fontsize=10.4, weight='bold', loc='left')
    ax.legend(fontsize=8, loc='lower right', framealpha=0.95)
    ax.grid(color='#dddddd', lw=0.7, zorder=1)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='docs')
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()
    out = Path(args.outdir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print('Sanity check against WEEK3_RESULTS.md — every plotted number:')
    print(f'  §9b  5-fold {FOLD_MEAN:+.2f} ± {FOLD_SD:.2f}, folds '
          + ', '.join(f'{f:+.2f}' for f in FOLDS)
          + f'  -> mean {np.mean(FOLDS):+.3f}, sd {np.std(FOLDS, ddof=1):.3f}')
    lc = sum(v for _, v, _ in PER_CLASS)
    print(f'  §9b  per-class sum {lc:+.2f}  (land cover '
          f'{lc - dict((c, v) for c, v, _ in PER_CLASS)["background"]:+.2f}, '
          f'reported +8.30)')
    e2e = [v for _, _, v in PER_CLASS]
    print(f'  §9c  end-to-end mean over 7 classes {np.mean(e2e):+.3f}  (reported +1.18)')
    print(f'  §9c  thresholds ' + ', '.join(f'{c} {t:.3f}' for c, t, _, _ in THRESHOLDS))
    print(f'  §9b  curve      ' + ', '.join(f'n={int(a)}:{b:+.2f}' for a, b, _, _ in CURVE))

    fig = plt.figure(figsize=(13.6, 4.5))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 0.95], wspace=0.30,
                          left=0.055, right=0.985, top=0.86, bottom=0.17)
    panel_a(fig.add_subplot(gs[0]))
    panel_b(fig.add_subplot(gs[1]))
    panel_c(fig.add_subplot(gs[2]))
    fig.suptitle('Per-class thresholds: +1.18 ± 0.45 mIoU on LoveDA, no weights trained',
                 fontsize=12.2, weight='bold', x=0.055, ha='left', y=0.975)

    stem = out / 'fig7_method'
    fig.savefig(stem.with_suffix('.png'), dpi=args.dpi, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    print(f'\nwritten: {stem}.png  +  .pdf')


if __name__ == '__main__':
    main()
