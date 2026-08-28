"""
Figures 3-5 for the paper. One script so the numbers live in one place.

    3  OEM per-class IoU delta      -- why +2.28 is not a segmentation improvement
    4  detection AUC across signals -- both datasets, one axis
    5  atom purity, cc vs SLIC      -- why atomisation dominated the prior

EVERY NUMBER IS CITED to WEEK3_RESULTS.md and must match it. The script prints
all of them on render, so drift between a figure and its source table is visible
rather than silent.

    python scripts/fig_results.py --outdir docs
    python scripts/fig_results.py --only 3
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

C_LOVE, C_OEM = '#c0392b', '#1f6f8b'
C_BAD, C_GOOD, C_BG = '#c0392b', '#2e8b57', '#8e44ad'


# ------------------------------------------------------------------ figure 3
# WEEK3_RESULTS.md 8.1. OEM, best honest operating point, SLIC atoms.
# (class, IoU before, IoU after, REPORTED delta). The delta is carried explicitly
# rather than recomputed: before/after are rounded to 2dp in the source table, so
# 29.94 - 27.88 gives +2.06 where the run reported +2.05. Small, but a figure that
# disagrees with its own table by 0.01 is a figure nobody can check.
OEM_PER_CLASS = [
    ('background', 17.13, 39.80, +22.67),
    ('bareland',   13.77, 13.40,  -0.36),
    ('grass',      42.92, 43.15,  +0.23),
    ('pavement',   27.88, 29.94,  +2.05),
    ('road',       45.88, 45.32,  -0.56),
    ('tree',       63.91, 63.26,  -0.64),
    ('water',      66.57, 67.60,  +1.02),
    ('cropland',   44.08, 43.98,  -0.10),
    ('building',   75.32, 71.57,  -3.75),
]


def fig3(outdir, dpi):
    names = [c for c, _, _, _ in OEM_PER_CLASS]
    d = np.array([delta for _, _, _, delta in OEM_PER_CLASS])
    order = np.argsort(d)                       # most negative at the bottom
    names = [names[i] for i in order]
    d = d[order]
    is_bg = np.array([n == 'background' for n in names])

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.patch.set_facecolor('white')
    colors = [C_BG if b else (C_GOOD if v > 0 else C_BAD)
              for b, v in zip(is_bg, d)]
    y = np.arange(len(d))
    ax.barh(y, d, color=colors, edgecolor='black', linewidth=0.7, height=0.68, zorder=3)
    ax.axvline(0, color='black', lw=1.0, zorder=4)

    for i, v in enumerate(d):
        ax.text(v + (0.5 if v >= 0 else -0.5), i, f'{v:+.2f}',
                va='center', ha='left' if v >= 0 else 'right',
                fontsize=10, fontweight='bold', zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel('Δ IoU after recovering the residual', fontsize=10.5)
    ax.set_xlim(-7.5, 27.5)
    ax.grid(axis='x', alpha=0.22, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)

    real = d[~is_bg].sum()
    ax.text(0.985, 0.06,
            f'`background`  {d[is_bg][0]:+.2f}\n'
            f'8 real classes  {real:+.2f}\n'
            f'{"─" * 22}\n'
            f'mean over 9 classes  {(d.sum() / 9):+.2f} mIoU',
            transform=ax.transAxes, fontsize=9.6, family='monospace',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='#fbfbfb', ec='#999', lw=0.8))

    ax.set_title('OpenEarthMap: the +2.28 mIoU gain is `background` alone',
                 fontsize=13.5, fontweight='bold', pad=42)
    ax.text(0.5, 1.012,
            'Recovery relabels 13.8M background-assigned pixels at 27.5% precision. '
            'Those pixels were already wrong,\nso `background` sheds false positives '
            'either way — while land-cover classification gets worse.',
            transform=ax.transAxes, fontsize=9, color='#444', ha='center', va='bottom')

    fig.tight_layout()
    _save(fig, outdir / 'fig3_oem_per_class', dpi)
    print(f'  fig 3  background {d[is_bg][0]:+.2f}   real classes {real:+.2f}   '
          f'mIoU {(d.sum() / 9):+.2f}   [WEEK3 8.1]')


# ------------------------------------------------------------------ figure 4
# WEEK3_RESULTS.md 9. None = not measured on that dataset.
SIGNALS = [   # (label, LoveDA AUC, OEM AUC, level)
    ('conf  (= P_final)',            0.582, 0.794, 'pixel'),
    ('conf2  (runner-up class)',     0.541, 0.913, 'pixel'),
    ('gap  = conf − conf2',          0.558, 0.601, 'pixel'),
    ('fconf  (= P_fused, ungated)',  0.559, 0.781, 'pixel'),
    ('fgap  = fconf − conf',         0.447, 0.796, 'pixel'),
    ('S_pres  (max over classes)',   0.434, 0.703, 'pixel'),
    ('S_pres  (argmax class)',       0.520, 0.681, 'pixel'),
    ('mean conf  per atom',          0.576, 0.798, 'region'),
    ('max conf  per atom',           0.516, 0.768, 'region'),
    ('novelty vs prototypes',        0.528, None,  'appearance'),
    ('mean colour',                  0.586, None,  'appearance'),
    ('gradient energy (texture)',    0.622, None,  'appearance'),
]
FLOOR = 0.53          # empirical, WEEK3 9.2


def fig4(outdir, dpi):
    labels = [s[0] for s in SIGNALS]
    lv = np.array([s[1] if s[1] is not None else np.nan for s in SIGNALS])
    oe = np.array([s[2] if s[2] is not None else np.nan for s in SIGNALS])
    y = np.arange(len(SIGNALS))[::-1]

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    fig.patch.set_facecolor('white')

    ax.axvspan(0.40, FLOOR, color='#bbb', alpha=0.32, zorder=0)
    ax.axvline(0.5, color='#777', lw=1.0, ls=':', zorder=1)
    ax.axvline(FLOOR, color='#777', lw=1.2, ls='--', zorder=1)

    h = 0.36
    ax.barh(y + h / 2, lv, height=h, color=C_LOVE, edgecolor='black',
            linewidth=0.6, label='LoveDA  (background 36.1% — catch-all)', zorder=3)
    ax.barh(y - h / 2, oe, height=h, color=C_OEM, edgecolor='black',
            linewidth=0.6, label='OpenEarthMap  (background 0.84%)', zorder=3)

    for yy, v in zip(y + h / 2, lv):
        if np.isfinite(v):
            ax.text(v + 0.006, yy, f'{v:.3f}', va='center', fontsize=8.2, zorder=4)
    for yy, v in zip(y - h / 2, oe):
        if np.isfinite(v):
            ax.text(v + 0.006, yy, f'{v:.3f}', va='center', fontsize=8.2,
                    fontweight='bold', zorder=4)
        else:
            ax.text(0.408, yy, 'not measured', va='center', fontsize=7.4,
                    color='#888', style='italic', zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.4)
    ax.set_xlim(0.40, 1.0)
    ax.set_xlabel('AUC — separating recoverable pixels from genuine background',
                  fontsize=10.5)
    ax.set_ylim(-1.5, y.max() + 0.7)
    ax.text(FLOOR + 0.005, -1.42, 'empirical floor 0.53', fontsize=8,
            color='#555', va='bottom', ha='left')
    ax.text(0.499, -1.42, 'chance 0.50', fontsize=8, color='#777',
            va='bottom', ha='right')
    ax.grid(axis='x', alpha=0.22, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95,
              bbox_to_anchor=(1.0, 0.045))

    ax.set_title('The same signals detect the residual on one dataset and not the other',
                 fontsize=13.5, fontweight='bold', pad=40)
    ax.text(0.5, 1.010,
            'On LoveDA every signal lands between 0.434 and 0.622 — at most 0.09 above '
            'the floor. On OpenEarthMap the same\nrunner-up class score reaches 0.913: '
            'where `background` is rare, a strong second place means a real class was '
            'suppressed.',
            transform=ax.transAxes, fontsize=9, color='#444', ha='center', va='bottom')

    fig.tight_layout()
    _save(fig, outdir / 'fig4_detection_auc', dpi)
    print(f'  fig 4  LoveDA best {np.nanmax(lv):.3f}   OEM best {np.nanmax(oe):.3f}   '
          f'floor {FLOOR}   [WEEK3 9]')


# ------------------------------------------------------------------ figure 5
# WEEK3_RESULTS.md 4. Cumulative share of pixels in atoms at or below a purity.
PURITY_X = [0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
PURITY_CC = [12.5, 28.5, 45.3, 61.4, 78.8, 94.7]
PURITY_SLIC = [1.1, 5.9, 10.8, 16.2, 23.0, 35.1]


def fig5(outdir, dpi):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9),
                                  gridspec_kw={'width_ratios': [1.55, 1]})
    fig.patch.set_facecolor('white')

    ax.plot(PURITY_X, PURITY_CC, 'o-', color=C_BAD, lw=2.2, ms=7,
            label='connected components', zorder=3)
    ax.plot(PURITY_X, PURITY_SLIC, 's-', color=C_OEM, lw=2.2, ms=7,
            label='SLIC superpixels', zorder=3)
    ax.fill_between(PURITY_X, PURITY_SLIC, PURITY_CC, color='#999', alpha=0.16, zorder=2)

    ax.annotate('12.5% of pixels sit in atoms\nno better than a coin flip',
                xy=(0.503, 12.5), xytext=(0.525, 72),
                fontsize=8.8, color='#8a1a1a', ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='#8a1a1a', lw=1.1,
                                shrinkA=2, shrinkB=3,
                                connectionstyle='arc3,rad=0.18'))
    ax.annotate('1.1% with SLIC', xy=(0.503, 1.1), xytext=(0.60, 14),
                fontsize=8.8, color=C_OEM, fontweight='bold', ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color=C_OEM, lw=1.1,
                                shrinkA=2, shrinkB=3,
                                connectionstyle='arc3,rad=-0.18'))

    ax.set_xlabel('atom purity  (share of an atom’s pixels in its majority GT class)',
                  fontsize=9.8)
    ax.set_ylabel('% of pixels in atoms at or below', fontsize=9.8)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.22, zorder=0)
    ax.legend(fontsize=9.4, loc='upper left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title('Lower is better — fewer pixels in unlabelable atoms',
                 fontsize=10.5, pad=8)

    bars = ax2.bar([0, 1], [72.8, 92.8], width=0.55,
                   color=[C_BAD, C_OEM], edgecolor='black', linewidth=0.8, zorder=3)
    for x, v in zip([0, 1], [72.8, 92.8]):
        ax2.text(x, v + 1.2, f'{v:.1f}%', ha='center', fontsize=12,
                 fontweight='bold', zorder=4)
    # the gap, drawn between the bars so it crosses neither of them
    ax2.hlines(72.8, -0.30, 0.50, color='#333', lw=1.0, ls='--', zorder=4)
    ax2.annotate('', xy=(0.50, 92.8), xytext=(0.50, 72.8),
                 arrowprops=dict(arrowstyle='<->', color='#333', lw=1.4))
    ax2.text(0.44, 82.8, '+20.0\npoints', ha='right', va='center', fontsize=10,
             fontweight='bold', color='#333')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['connected\ncomponents', 'SLIC'], fontsize=9.6)
    ax2.set_ylabel('oracle-labeller accuracy', fontsize=9.8)
    ax2.set_ylim(0, 108)
    ax2.grid(axis='y', alpha=0.22, zorder=0)
    for s in ('top', 'right'):
        ax2.spines[s].set_visible(False)
    ax2.set_title('Hard ceiling on ANY region-level method', fontsize=10.5, pad=8)

    fig.suptitle('Atomisation, not the co-occurrence prior, set the ceiling',
                 fontsize=13.5, fontweight='bold', y=0.995)
    fig.text(0.5, 0.925,
             'A perfect co-occurrence matrix was worth +0.3 points (WEEK3 §6). '
             'Choosing the right atoms was worth 20.',
             fontsize=9, color='#444', ha='center', va='top')

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, outdir / 'fig5_atom_purity', dpi)
    print('  fig 5  cc 0.728 / ceiling 72.8%   slic 0.928 / ceiling 92.8%   [WEEK3 4]')


def _save(fig, stem, dpi):
    fig.savefig(stem.with_suffix('.png'), dpi=dpi, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    print(f'written: {stem}.png  +  .pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='docs')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--only', choices=['3', '4', '5'], default=None)
    args = ap.parse_args()
    out = Path(args.outdir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print('Sanity check against WEEK3_RESULTS.md:')
    for k, fn in (('3', fig3), ('4', fig4), ('5', fig5)):
        if args.only in (None, k):
            fn(out, args.dpi)


if __name__ == '__main__':
    main()
