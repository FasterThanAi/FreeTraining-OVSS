"""
Qualitative figure -- the method, running, on four tiles chosen to argue rather
than to flatter.

WHY THIS FIGURE EXISTS. The paper carries four figures and not one of them shows
a segmentation result. Reviewers look at figures before tables, and a method
whose whole claim is "same weights, different decision rule" is far more
convincing shown side by side than described.

WHY THESE FOUR TILES. Two wins, one no-op and one failure, in that order:

  1. a normal urban scene         -- what it does, and what it leaves alone
  2. the mechanism at full strength -- a whole missed canopy recovered
  3. a tile it barely touches     -- it does not churn when nothing is wrong
  4. a tile where it LOSES        -- the per-class weight is one number for the
                                     whole dataset, and on an ambiguous tile it
                                     over-fires

⭐ Tile 4 is not optional. The five-fold Potsdam result is +4.86 +- 0.35 with 5/5
folds positive; that is an average, and a figure of four wins would misrepresent
it. A reviewer who cannot find the failure case assumes it was hidden.

⛔ Tiles whose gain is mostly the CATCH-ALL are deliberately excluded. Two of the
strongest-looking candidates (+8.65 and +6.66 full mIoU) are +2.01 and +0.95 once
`clutter` is removed. Putting those here would reproduce SS8.1's error -- a metric
moving without segmentation quality moving -- in the figure of the paper that
documents it.

Panels and per-tile numbers come from demo_app, unmodified, so the figure and the
demo cannot drift apart. Every number plotted is printed on render.

Runs on the workstation (needs SAM 3 + the segov3 env), from ~/SegEarth-OV-3:

    python ~/FreeTraining-OVSS/scripts/fig_qualitative.py \
      --config configs/cfg_potsdam.py \
      --preset configs/cfg_potsdam_reorder.py \
      --outdir ~/FreeTraining-OVSS/docs
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_app as D                     # noqa: E402

# The four, in presentation order, with what each is there to say. Paths are
# relative to the dataset's image directory.
TILES = [
    ('2_13_0_2560_512_3072',  'a normal scene'),
    ('2_13_0_4096_512_4608',  'the mechanism at full strength'),
    ('2_13_0_5120_512_5632',  'nothing to fix, nothing changed'),
    ('2_13_1024_0_1536_512',  'where it fails'),
]
COLS = [('input', 'input'), ('truth', 'ground truth'),
        ('baseline', 'SegEarth-OV3'), ('calibrated', 'ours')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--preset', required=True)
    ap.add_argument('--img-dir', default='data/Potsdam/img_dir/val')
    ap.add_argument('--tiles', nargs='*', default=[t for t, _ in TILES])
    ap.add_argument('--outdir', default='docs')
    ap.add_argument('--name', default='fig8_qualitative')
    args = ap.parse_args()

    preset = D.read_preset(args.preset)
    eng = D.Engine(args.config)
    vp = eng.classname_path
    if not (vp and Path(vp).expanduser().exists()):
        raise SystemExit(f'⛔ class list not found ({vp!r})')
    vocab = Path(vp).expanduser().read_text()

    rows = []
    for stem in args.tiles:
        f = Path(args.img_dir) / f'{stem}.png'
        if not f.exists():
            cands = sorted(Path(args.img_dir).glob(stem + '.*'))
            if not cands:
                raise SystemExit(f'⛔ {f} not found')
            f = cands[0]
        print(f'  {f.name}')
        names, pan, note, st = D.panels(eng, f, vocab, preset)
        if 'calibrated' not in pan:
            raise SystemExit(f'⛔ the preset did not apply to {f.name}; the figure '
                             'would show a baseline column twice')
        if 'u0' not in st:
            raise SystemExit(f'⛔ no ground truth for {f.name}; the figure needs '
                             'the truth column and the IoU numbers')
        rows.append((stem, pan, st, names))

    n = len(rows[0][3])
    bg = eng.bg
    fig, axes = plt.subplots(len(rows), len(COLS),
                             figsize=(2.25 * len(COLS), 2.62 * len(rows)))
    axes = np.atleast_2d(axes)

    print('\n  numbers plotted (each is recomputed here, not transcribed):')
    for r, (stem, pan, st, names) in enumerate(rows):
        ib, if_ = D.ratio(st['i0'], st['u0']), D.ratio(st['i1'], st['u1'])
        pres = st['present'] > 0
        m0, m1, idx = D.miou(ib, if_, pres)
        e0, e1, _ = D.miou(ib, if_, pres, keep=lambda c: c != bg)
        # the class that moves most, among those actually in the truth
        movers = [(c, if_[c] - ib[c]) for c in range(n)
                  if pres[c] and not (np.isnan(ib[c]) or np.isnan(if_[c]))]
        cls, d = max(movers, key=lambda t: abs(t[1])) if movers else (None, 0.0)

        for c, (key, _) in enumerate(COLS):
            ax = axes[r, c]
            ax.imshow(pan[key])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor('#bbb'); s.set_linewidth(0.6)
            if r == 0:
                ax.set_title(COLS[c][1], fontsize=9, pad=5)

        lab = (f'{TILES[r][1] if r < len(TILES) else stem}\n'
               f'mIoU {m0:.1f}→{m1:.1f} ({m1 - m0:+.1f})\n'
               f'excl. {names[bg]} ({e1 - e0:+.1f})'
               + (f'\n{names[cls]} {d:+.1f}' if cls is not None else ''))
        axes[r, 0].set_ylabel(lab, fontsize=7.5, labelpad=6,
                              color='#c0392b' if m1 < m0 else '#222')
        print(f'    {stem:26s} mIoU {m0:6.2f} -> {m1:6.2f} ({m1 - m0:+6.2f})  '
              f'excl {e0:6.2f} -> {e1:6.2f} ({e1 - e0:+6.2f})  '
              f'{len(idx)} classes  '
              + (f'{names[cls]} {d:+.1f}' if cls is not None else ''))

    # one legend for the whole figure
    handles = [plt.Rectangle((0, 0), 1, 1,
                             fc=np.array(D.PALETTE[c % len(D.PALETTE)]) / 255,
                             ec='#0003') for c in range(n)]
    fig.legend(handles, names, loc='lower center', ncol=min(n, 7), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = Path(args.outdir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = out / f'{args.name}.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'  wrote {p}')
    print('\n  ⚠️  Row 4 is a LOSS and belongs in the figure. The five-fold Potsdam\n'
          '      result is +4.86 ± 0.35 (5/5 folds); that is an average, and four\n'
          '      wins would misrepresent it.')


if __name__ == '__main__':
    main()
