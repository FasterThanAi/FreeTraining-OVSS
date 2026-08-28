"""
Figure 2 — the mechanism. Label design governs the residual and its detectability.

THE ARGUMENT. LoveDA and OpenEarthMap disagree about almost everything the
project measures, and one variable orders all of it: the share of ground truth
occupied by a catch-all `background` class. Where it is a third of the scene,
SAM 3 has a plausible answer everywhere, so nearly a third of real land cover is
discarded and no signal distinguishes that residual from genuine background.
Where the vocabulary covers the scene, the residual is small and the runner-up
class score finds it easily.

WHY SMALL MULTIPLES AND NOT A SCATTER. With two datasets a scatter plot of
"background share vs outcome" has two points, and two points are collinear by
construction -- it would assert a trend the data cannot support. Paired bars
state exactly what was measured: six quantities, each moving in the direction
the mechanism predicts, with no interpolation implied. The caption says n=2 out
loud, and panel 1 is drawn as the cause with the rest as consequences.

EVERY NUMBER IS CITED to WEEK3_RESULTS.md below and must match it. If that file
changes, change this and re-render -- a figure that has drifted from its source
table is worse than no figure.

    python scripts/fig_mechanism.py --out docs/fig2_mechanism.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

LOVEDA = 'LoveDA\n1669 tiles, τ=0.5'
OEM = 'OpenEarthMap\n384 tiles, τ=0.1'

C_LOVE = '#c0392b'      # catch-all vocabulary
C_OEM = '#1f6f8b'       # covering vocabulary

# ---------------------------------------------------------------- the data
# All from WEEK3_RESULTS.md §7 and §8. Provenance per row.
PANELS = [
    dict(title='① `background` share of GT',
         sub='the cause: is the vocabulary a catch-all?',
         unit='% of labelled pixels', love=36.1, oem=0.84,
         fmt='{:.2f}%', src='§7'),
    dict(title='② real-class pixels discarded',
         sub='how much land cover is assigned to background',
         unit='% of real-class pixels', love=29.68, oem=3.78,
         fmt='{:.2f}%', src='§7'),
    dict(title='③ catastrophic tiles',
         sub='tiles losing ≥99% of their real-class pixels',
         unit='% of tiles', love=100 * 198 / 1669, oem=0.0,
         fmt='{:.1f}%', src='§7 (198/1669 vs 0/384)'),
    dict(title='④ corr(presence, discard)',
         sub='does presence collapse track failure?',
         unit='Pearson r', love=-0.750, oem=0.094,
         fmt='{:+.3f}', src='§7', zero=True),
    dict(title='⑤ best detection AUC',
         sub='can the residual be told from real background?',
         unit='AUC', love=0.622, oem=0.913,
         fmt='{:.3f}', src='§9', chance=0.53),
    dict(title='⑥ honest recovery',
         sub='does recovering it improve segmentation?',
         unit='Δ mIoU', love=0.04, oem=2.28,
         fmt='{:+.2f}', src='§8', zero=True, caveat_oem=True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/fig2_mechanism.png')
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    fig.patch.set_facecolor('white')

    for ax, p in zip(axes.ravel(), PANELS):
        vals = [p['love'], p['oem']]
        bars = ax.bar([0, 1], vals, width=0.55,
                      color=[C_LOVE, C_OEM], edgecolor='black', linewidth=0.8, zorder=3)

        # the caveated bar is hatched -- see the annotation below
        if p.get('caveat_oem'):
            bars[1].set_hatch('////')
            bars[1].set_edgecolor('black')

        lo = min(0, min(vals)); hi = max(0, max(vals))
        pad = 0.30 * (hi - lo if hi > lo else 1)
        ax.set_ylim(lo - pad * 0.55, hi + pad)

        for x, v in zip([0, 1], vals):
            off = pad * 0.10 * (1 if v >= 0 else -1)
            ax.text(x, v + off, p['fmt'].format(v), ha='center',
                    va='bottom' if v >= 0 else 'top',
                    fontsize=11.5, fontweight='bold', zorder=4)

        if p.get('zero'):
            ax.axhline(0, color='black', lw=0.9, zorder=2)
        if p.get('chance'):
            ax.axhline(p['chance'], color='#666', lw=1.2, ls='--', zorder=2)
            ax.text(-0.58, p['chance'], 'chance (0.53)', fontsize=7.5,
                    color='#555', va='center', ha='left',
                    bbox=dict(boxstyle='square,pad=0.18', fc='white', ec='none'))

        ax.set_title(p['title'], fontsize=11.5, fontweight='bold', pad=22)
        ax.text(0.5, 1.028, p['sub'], transform=ax.transAxes, fontsize=8.4,
                color='#444', ha='center', style='italic')
        ax.set_ylabel(p['unit'], fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([LOVEDA, OEM], fontsize=8.2)
        ax.set_xlim(-0.62, 1.62)
        ax.grid(axis='y', alpha=0.22, zorder=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.text(0.985, 0.955, p['src'], transform=ax.transAxes, fontsize=6.8,
                color='#999', ha='right', va='top')

    # Panel 6 needs its asterisk ON the figure. A reader who takes +2.28 at face
    # value has learned the opposite of what the experiment showed.
    ax6 = axes.ravel()[5]
    ax6.annotate('hatched: +22.67 of this\nis `background` alone.\n'
                 'Real classes net −2.11.',
                 xy=(0.72, 1.95), xytext=(0.10, 1.20),
                 fontsize=7.8, color='#8a1a1a', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.35', fc='#fdf0ef',
                           ec='#8a1a1a', lw=0.7),
                 arrowprops=dict(arrowstyle='->', color='#8a1a1a', lw=1.0))

    fig.suptitle('Label design, not the model, governs the background residual',
                 fontsize=15, fontweight='bold', y=0.985)
    fig.text(0.5, 0.938,
             'A catch-all `background` class gives SAM 3 a plausible answer everywhere: '
             'the residual is large and indistinguishable from genuine background. '
             'Where the vocabulary covers the scene, it is small and easily detected.\n'
             'Panel ① is the cause; ②–⑥ are consequences, each moving as it predicts. '
             'n = 2 datasets — the direction is consistent, the functional form is not '
             'claimed.',
             fontsize=8.8, color='#333', ha='center', va='top')

    fig.tight_layout(rect=[0, 0.005, 1, 0.905])
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor='white')
    fig.savefig(out.with_suffix('.pdf'), facecolor='white')   # vector, for LaTeX
    print(f'written: {out}\nwritten: {out.with_suffix(".pdf")}')

    print('\nSanity check against WEEK3_RESULTS.md §7/§8:')
    for p in PANELS:
        print(f'  {p["title"]:34s} LoveDA {p["fmt"].format(p["love"]):>9s}   '
              f'OEM {p["fmt"].format(p["oem"]):>9s}   [{p["src"]}]')


if __name__ == '__main__':
    main()
