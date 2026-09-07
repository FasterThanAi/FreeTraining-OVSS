"""
The open-vocabulary figure: one scene, two vocabularies, no retraining.

WHY IT EXISTS. Every other figure in the paper shows one fixed class list, so
none of them demonstrates the property the setting is named for. This one does:
the same tiles, the same weights, the same forward pass, segmented under two
different class lists supplied as plain words at inference.

⭐ THE ARGUMENT IS THE DISAGREEMENT. On imagery from a region no benchmark here
covers, a blue-painted metal roof is confidently labelled `water` under a
vocabulary that offers no better word, and is separated correctly once `blue
roof` is available. Naming a class is the whole intervention.

⚠️ THE FAILURES BELONG IN THE CAPTION. Where a vocabulary has no word for what
is present the model assigns the nearest one it has -- dense canopy becomes
`bare earth` when the only vegetation word is `banana tree, palm`. That is
vocabulary incompleteness rather than model failure, and it is the honest cost
of an open class list.

⛔ NO GROUND TRUTH EXISTS FOR THIS IMAGERY, so nothing here is an accuracy
claim. Per-class PIXEL SHARES are printed instead, so what the caption asserts
rests on counts rather than on what the eye reads off a colour overlay.

⚠️ Prompts were revised ONCE, after looking at the first run. Tuning a class
list until the picture looks right is not a result; say plainly that one
revision happened and stop.

Runs on the workstation, from ~/SegEarth-OV-3:

    python ~/FreeTraining-OVSS/scripts/fig_vocabulary.py \
      --config configs/cfg_potsdam.py \
      --vocab-a ~/FreeTraining-OVSS/configs/cls_oam_south_asia.txt \
      --vocab-b /tmp/vocab_b.txt \
      --tiles ~/data/oam/img_dir/val/6a9e75688cb6541dae9ebee6_20992_0.png \
              ~/data/oam/img_dir/val/6a9e75688cb6541dae9ebee6_22528_26624.png \
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


def run(eng, tile, vocab_text):
    names, pan, note, _ = D.panels(eng, tile, vocab_text, None)
    img = pan['input']
    lg = eng.logits(tile)
    pred = D.apply_rule(lg, eng.base_tau, eng.bg)
    nod = img[..., :3].max(axis=2) == 0
    share = np.array([100.0 * ((pred == c) & ~nod).sum() / max((~nod).sum(), 1)
                      for c in range(len(names))])
    return names, pan['baseline'], share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--vocab-a', required=True)
    ap.add_argument('--vocab-b', required=True)
    ap.add_argument('--tiles', nargs='+', required=True)
    ap.add_argument('--outdir', default='docs')
    ap.add_argument('--name', default='fig9_vocabulary')
    args = ap.parse_args()

    va = Path(args.vocab_a).expanduser().read_text()
    vb = Path(args.vocab_b).expanduser().read_text()
    eng = D.Engine(args.config)

    rows = []
    for t in args.tiles:
        t = Path(t).expanduser()
        if not t.exists():
            raise SystemExit(f'⛔ {t} not found')
        print(f'  {t.name}')
        na, pa, sa = run(eng, t, va)
        nb, pb, sb = run(eng, t, vb)
        import matplotlib.image as mpimg
        rows.append((t.stem, mpimg.imread(str(t)), (na, pa, sa), (nb, pb, sb)))

    print('\n  per-class pixel share, nodata excluded '
          '(what the caption may assert):')
    for stem, _, (na, _, sa), (nb, _, sb) in rows:
        print(f'    {stem}')
        for tag, nm, sh in (('A', na, sa), ('B', nb, sb)):
            top = sorted(range(len(nm)), key=lambda c: -sh[c])[:4]
            print(f'      vocab {tag}: ' + ',  '.join(
                f'{nm[c]} {sh[c]:.1f}%' for c in top if sh[c] >= 0.5))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9.6, 3.35 * len(rows)))
    axes = np.atleast_2d(axes)
    heads = ['input', 'vocabulary A', 'vocabulary B']
    for r, (stem, img, (na, pa, _), (nb, pb, _)) in enumerate(rows):
        for c, im in enumerate([img, pa, pb]):
            ax = axes[r, c]
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor('#bbb'); s.set_linewidth(0.6)
            if r == 0:
                ax.set_title(heads[c], fontsize=9, pad=5)

    def legend(names, ax, title):
        h = [plt.Rectangle((0, 0), 1, 1,
                           fc=np.array(D.PALETTE[c % len(D.PALETTE)]) / 255,
                           ec='#0003') for c in range(len(names))]
        ax.legend(h, names, loc='upper center', ncol=min(len(names), 4),
                  frameon=False, fontsize=7.2, title=title,
                  title_fontsize=7.6, bbox_to_anchor=(0.5, 0.0))
    legend(rows[0][2][0], axes[-1, 1], 'vocabulary A')
    legend(rows[0][3][0], axes[-1, 2], 'vocabulary B')

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = Path(args.outdir).expanduser(); out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = out / f'{args.name}.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'  wrote {p}')
    print('\n  ⚠️  No ground truth exists here. Nothing in this figure is an')
    print('      accuracy claim; the shares above are what may be asserted.')


if __name__ == '__main__':
    main()
