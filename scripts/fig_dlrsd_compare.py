"""Side-by-side panels on DLRSD: image | ground truth | baseline | ours | what changed.

⭐ NO GPU. The cache holds the full 17-channel `logits` stack, and both rungs are
pure arithmetic on it:

    baseline   pred = argmax(logits),        discard where max < 0.1
    ours       pred = argmax(w * logits),    discard where the RAW winning score
                                             is below that class's own threshold

So these panels are the SAME arithmetic that produced 37.27 -> 44.42, not a
re-run that could drift from it. `w` and the threshold vector are read out of
the deployed config, never retyped -- a permuted vector passes a length check
and produces a perfectly plausible picture.

⛔ TILES ARE CHOSEN TO ARGUE, NOT TO FLATTER, and `--auto` enforces it: the
biggest win, a tile the method barely touches, a tile it LOSES on, and a tile
containing a class that scores zero however it is prompted. @DLRSD_RESULTS.md
§9-10 -- 13 classes improve, 3 lose and 2 are dead, so a sheet of four wins
would misrepresent it. `fig_qualitative.py` makes the same argument for Potsdam
and keeps a tile that loses 7.72.

⭐ THE FIFTH PANEL IS THE POINT. An output panel shows what the method produced;
the changed panel shows what it DECIDED -- green where it corrected a pixel, red
where it broke one, grey where it swapped one wrong label for another. A method
whose whole claim is "same weights, different decision rule" is argued by that
panel and merely illustrated by the others.

    python scripts/fig_dlrsd_compare.py \\
        --cache ~/outputs/dlrsd_full/cache \\
        --img-dir ~/data/dlrsd/img_dir/val \\
        --deploy-cfg ~/SegEarth-OV-3/configs/cfg_dlrsd_perclass.py \\
        --auto --out docs/dlrsd_compare
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as LB_MOD  # noqa: E402

# DLRSD's own palette, read off its label PNGs (inspect_dataset.py --palette)
PALETTE = [(166, 202, 240), (128, 128, 0), (0, 0, 128), (255, 0, 0),
           (0, 128, 0), (128, 0, 0), (255, 233, 233), (160, 160, 164),
           (0, 128, 128), (90, 87, 255), (255, 255, 0), (255, 192, 0),
           (0, 0, 255), (255, 0, 192), (128, 0, 128), (0, 255, 0),
           (0, 255, 255)]
SINK_RGB = (20, 20, 20)          # discarded: no class, and not scored as one


def read_vectors(cfg_path, n):
    """Pull `class_scale` and `prob_thd` out of the generated config.

    ⛔ Read, never retyped. The segmentor validates the LENGTH of these vectors
    and cannot detect a PERMUTATION, so a transposed pair would render a picture
    that looks entirely reasonable and is several points wrong.
    """
    src = Path(cfg_path).expanduser().read_text()
    out = {}
    for key in ('class_scale', 'prob_thd'):
        m = re.search(rf'{key}\s*=\s*\[(.*?)\]', src, re.S)
        if not m:
            out[key] = None
            continue
        vals = [float(x) for x in m.group(1).replace('\n', ' ').split(',')
                if x.strip()]
        if len(vals) != n:
            raise SystemExit(f'⛔ {key} in {cfg_path} has {len(vals)} entries, '
                             f'the cache has {n} classes.')
        out[key] = np.asarray(vals, np.float32)
    if out['prob_thd'] is None:
        raise SystemExit(f'⛔ no prob_thd found in {cfg_path}')
    return out['class_scale'], out['prob_thd']


def predict(logits, w, tau, sink):
    """The segmentor's rule, exactly: scale the argmax, threshold the RAW score."""
    s = logits if w is None else logits * w[:, None, None]
    pred = np.argmax(s, 0)
    raw = np.take_along_axis(logits, pred[None], 0)[0]    # RAW, not scaled
    out = pred.copy()
    out[raw < tau[pred]] = sink
    return out


def colourise(idx, sink):
    h, w = idx.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[...] = SINK_RGB
    for k, c in enumerate(PALETTE):
        rgb[idx == k] = c
    return rgb


def per_class_iou(pred, gt, n, sink):
    """IoU over the classes this tile actually contains, either side."""
    out = {}
    for k in range(n):
        g, p = gt == k, pred == k
        u = (g | p).sum()
        if u:
            out[k] = 100.0 * (g & p).sum() / u
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--deploy-cfg', required=True,
                    help='the config written by reorder_deploy.py --cfg-out')
    ap.add_argument('--tau', type=float, default=0.1, help='baseline threshold')
    ap.add_argument('--tiles', nargs='*', default=None, help='cache stems')
    ap.add_argument('--auto', action='store_true',
                    help='choose four tiles that argue: best win, near no-op, a '
                         'LOSS, and one holding a class that scores zero')
    ap.add_argument('--extra-dir', default=None,
                    help='optional 5th column: a directory of masks from another '
                         'method, named by tile stem. ⚠️ See the warning printed '
                         'when it is used.')
    ap.add_argument('--extra-label', default='other method')
    ap.add_argument('--out', default='docs/dlrsd_compare')
    ap.add_argument('--scan', type=int, default=300,
                    help='tiles to score when choosing with --auto')
    args = ap.parse_args()

    L = LB_MOD.from_cache(args.cache)
    n, sink = L.n, L.bg - 1
    if not L.sink:
        print(f'  ⚠️ this cache has a catch-all (`{L.catch_all}`); the panels '
              f'will still render but the "discarded" colour is a real class.')
    w, tau = read_vectors(args.deploy_cfg, n)
    print(f'  {n} classes, sink index {sink}')
    print(f'  scale  : {"none" if w is None else ", ".join(f"{x:.3f}" for x in w)}')
    print(f'  thresh : {", ".join(f"{x:.3f}" for x in tau)}')

    files = {f.stem: f for f in sorted(Path(args.cache).expanduser().glob('*.npz'))}
    base_tau = np.full(n, args.tau, np.float32)

    def load(stem):
        z = np.load(files[stem])
        if 'logits' not in z.files:
            raise SystemExit(f'⛔ {stem}.npz has no `logits`. Re-cache with '
                             f'--cache-full; the scale needs the whole stack.')
        return z['logits'].astype(np.float32), z['gt'].astype(np.int32) - 1

    # ---- choose the tiles
    if args.tiles:
        chosen = [(t, '') for t in args.tiles]
    elif args.auto:
        print(f'\n  scoring {min(args.scan, len(files))} tiles to choose four...')
        rows = []
        for i, stem in enumerate(list(files)[:args.scan]):
            lg, gt = load(stem)
            a = predict(lg, None, base_tau, sink)
            c = predict(lg, w, tau, sink)
            va = per_class_iou(a, gt, n, sink)
            vc = per_class_iou(c, gt, n, sink)
            ks = set(va) | set(vc)
            d = np.mean([vc.get(k, 0) - va.get(k, 0) for k in ks]) if ks else 0.0
            dead = any(vc.get(k, 0) == 0 and (gt == k).sum() > 200
                       for k in (4, 9))            # chaparral, mobile home
            rows.append((stem, d, dead))
            if (i + 1) % 100 == 0:
                print(f'    {i + 1}/{min(args.scan, len(files))}')
        rows.sort(key=lambda r: -r[1])
        # ⛔ The four roles must land on FOUR DIFFERENT tiles. On a small or
        # uniformly-signed set the best tile can also be the one nearest zero,
        # and the figure would then show the same scene twice under two
        # contradictory captions -- which is worse than showing three panels.
        # Each pick is taken from what is still unused.
        used, chosen = set(), []

        def take(cand, why):
            for r in cand:
                if r[0] not in used:
                    used.add(r[0])
                    chosen.append((r[0], why(r)))
                    return
            print(f'  ⚠️ no distinct tile left for: {why(("", 0.0, False))}')

        take(rows,                                   lambda r: f'biggest gain {r[1]:+.1f}')
        take(rows[::-1],                             lambda r: f'⛔ a LOSS {r[1]:+.1f}')
        take(sorted(rows, key=lambda r: abs(r[1])),  lambda r: f'near no-op {r[1]:+.1f}')
        take([r for r in rows if r[2]],              lambda r: '⛔ holds a class that scores zero')
        if len(chosen) < 4:
            print(f'  ⚠️ only {len(chosen)} distinct tiles available — raise '
                  f'--scan, or the cache is very small.')
        print('\n  chosen:')
        for st, why in chosen:
            print(f'    {st:22s} {why}')
    else:
        raise SystemExit('pass --tiles or --auto')

    # ---- render
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ncol = 5 + (1 if args.extra_dir else 0)
    fig, axes = plt.subplots(len(chosen), ncol,
                             figsize=(2.6 * ncol, 2.7 * len(chosen)))
    axes = np.atleast_2d(axes)

    print('\n  per-tile mean IoU over the classes present:')
    for r, (stem, why) in enumerate(chosen):
        lg, gt = load(stem)
        a = predict(lg, None, base_tau, sink)
        c = predict(lg, w, tau, sink)
        va, vc = per_class_iou(a, gt, n, sink), per_class_iou(c, gt, n, sink)
        ks = set(va) | set(vc)
        ma = np.mean([va.get(k, 0) for k in ks])
        mc = np.mean([vc.get(k, 0) for k in ks])

        img_p = Path(args.img_dir).expanduser() / f'{stem}.png'
        panels = [(np.array(Image.open(img_p).convert('RGB')), f'image  {stem}'),
                  (colourise(gt, sink), 'ground truth'),
                  (colourise(a, sink), f'baseline τ={args.tau}   {ma:.1f}'),
                  (colourise(c, sink), f'ours (both levers)   {mc:.1f}')]

        if args.extra_dir:
            ep = Path(args.extra_dir).expanduser() / f'{stem}.png'
            if ep.is_file():
                ea = np.array(Image.open(ep))
                if ea.ndim == 2:
                    ea = colourise(ea.astype(np.int32) - 1, sink)
                panels.insert(3, (ea, args.extra_label))
            else:
                panels.insert(3, (np.zeros_like(panels[1][0]),
                                  f'{args.extra_label}\n(missing)'))

        # ---- the panel that argues
        changed = a != c
        ch = np.full(gt.shape + (3,), 245, np.uint8)
        ch[changed & (c == gt)] = (40, 160, 60)      # fixed
        ch[changed & (a == gt)] = (200, 50, 50)      # broken
        ch[changed & (c != gt) & (a != gt)] = (190, 190, 190)
        fixed = int((changed & (c == gt)).sum())
        broke = int((changed & (a == gt)).sum())
        tot = int(changed.sum())
        panels.append((ch, f'changed {100*tot/gt.size:.1f}%\n'
                           f'fixed {fixed:,} · broke {broke:,}'))

        for cidx, (im, title) in enumerate(panels):
            ax = axes[r, cidx]
            ax.imshow(im)
            ax.set_title(title, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        print(f'    {stem:22s} {ma:5.1f} -> {mc:5.1f}  ({mc-ma:+5.1f})   '
              f'changed {100*tot/gt.size:4.1f}%  fixed {fixed:,}  broke {broke:,}'
              f'   {why}')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = out / f'dlrsd_compare.{ext}'
        fig.savefig(p, dpi=160, bbox_inches='tight')
        print(f'\n  written: {p}')

    if args.extra_dir:
        print('\n  ⚠️ THE EXTRA COLUMN IS NOT A CONTROLLED COMPARISON unless it '
              'came from the same\n     backbone, checkpoint, input resolution '
              'and vocabulary. Ours is SAM 3 at\n     1008x1008 with DLRSD\'s 17 '
              'published prompts. If the other column differs in\n     any of '
              'those, the panels differ for reasons that have nothing to do with '
              'the\n     decision rule, and the figure will be read as if they '
              'do. Label it, or drop it.')


if __name__ == '__main__':
    main()
