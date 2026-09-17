"""Right / wrong maps, pixel by pixel: image | ground truth | baseline | ours | changed.

Each pixel of the two middle panels is coloured by whether it was RIGHT:
    green      correct
    red        labelled as a different class
    dark grey  discarded by the threshold, and that was wrong
    white      no ground truth (ignored by every metric)
and the last panel by what the method CHANGED:
    green  fixed (wrong -> right)   red  broken (right -> wrong)
    yellow one error swapped for another            light grey  unchanged

Rung A (baseline) = published τ; rung C (ours) = per-class scale + per-class τ,
read from the deployed config. Same arithmetic as pixel_accuracy.py.
⭐ On Potsdam a discard IS a `clutter` prediction, so it is green where the truth
is clutter. On DLRSD a discard goes to an unscored sink and is never correct.

    # 4 argued tiles on one sheet (biggest gain, biggest loss, no-op, median)
    python scripts/fig_pixel_maps.py --preset dlrsd --auto --out docs/pixel_maps_dlrsd
    # every held-out tile + index.csv  (large; do not commit)
    python scripts/fig_pixel_maps.py --preset dlrsd --all --out ~/outputs/dlrsd_pixel_maps
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as LB                                          # noqa: E402
from pixel_accuracy import read_vector, decide               # noqa: E402

PRESETS = {
    'dlrsd': dict(cache='~/outputs/dlrsd_full/cache', tau=0.1,
                  split='~/splits/dlrsd_heldout.txt',
                  scale_cfg='~/SegEarth-OV-3/configs/cfg_dlrsd_perclass.py',
                  img_dir='~/data/dlrsd/img_dir/val'),
    'potsdam': dict(cache='~/outputs/potsdam_full/cache', tau=0.1,
                    split='~/splits/potsdam_reorder_heldout.txt',
                    scale_cfg='~/SegEarth-OV-3/configs/cfg_potsdam_reorder.py',
                    img_dir='~/SegEarth-OV-3/data/Potsdam/img_dir/val'),
}
GREEN, RED, GREY, WHITE = (46, 160, 67), (214, 39, 40), (80, 80, 80), (255, 255, 255)
YELLOW, LIGHT = (240, 190, 0), (228, 228, 228)
# ISPRS Potsdam's own colours, and DLRSD's read off its label PNGs.
PAL_POTSDAM = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0),
               (255, 255, 0), (255, 0, 0)]
PAL_DLRSD = [(166, 202, 240), (128, 128, 0), (0, 0, 128), (255, 0, 0),
             (0, 128, 0), (128, 0, 0), (255, 233, 233), (160, 160, 164),
             (0, 128, 128), (90, 87, 255), (255, 255, 0), (255, 192, 0),
             (0, 0, 255), (255, 0, 192), (128, 0, 128), (0, 255, 0),
             (0, 255, 255)]


def palette(n):
    if n == 6:
        return PAL_POTSDAM
    if n == 17:
        return PAL_DLRSD
    rng = np.random.default_rng(0)
    return [tuple(int(v) for v in rng.integers(40, 255, 3)) for _ in range(n)]


def tile_maps(L, gt, n, bg, w, tau_c, tau_a):
    """Per-pixel outcome codes for A and C, plus counts. 0 ignore, 1 correct,
    2 wrong label, 3 discarded (and wrong)."""
    m = gt > 0
    g = gt[m] - 1
    flat = L[:, m]
    codes, preds = [], []
    for wv, thr in ((None, tau_a), (w, tau_c)):
        out, fired = decide(flat, wv, thr, bg)
        c = np.where(out == g, 1, np.where(fired, 3, 2))
        full = np.zeros(gt.shape, np.uint8)
        full[m] = c
        codes.append(full)
        p = np.full(gt.shape, -1, np.int64)
        p[m] = out
        preds.append(p)
    a, c = codes
    lab = int(m.sum())
    st = dict(labelled=lab,
              acc_A=100 * (a == 1).sum() / max(lab, 1),
              acc_C=100 * (c == 1).sum() / max(lab, 1),
              fixed=int(((a > 1) & (c == 1)).sum()),
              broken=int(((a == 1) & (c > 1)).sum()),
              discarded_A=int((a == 3).sum()), discarded_C=int((c == 3).sum()),
              n_gt_classes=int(np.unique(g).size))
    st['delta'] = st['acc_C'] - st['acc_A']
    st['changed_pct'] = 100 * ((preds[0] != preds[1]) & m).sum() / max(lab, 1)
    return a, c, preds, st


def colour_codes(code):
    rgb = np.empty(code.shape + (3,), np.uint8)
    rgb[:] = WHITE
    rgb[code == 1], rgb[code == 2], rgb[code == 3] = GREEN, RED, GREY
    return rgb


def colour_changes(a, c, preds, gt):
    rgb = np.empty(gt.shape + (3,), np.uint8)
    rgb[:] = WHITE
    m = gt > 0
    rgb[m] = LIGHT
    rgb[(a > 1) & (c == 1)] = GREEN
    rgb[(a == 1) & (c > 1)] = RED
    rgb[(a > 1) & (c > 1) & (preds[0] != preds[1])] = YELLOW
    return rgb


def colour_gt(gt, pal):
    rgb = np.zeros(gt.shape + (3,), np.uint8)
    for k, col in enumerate(pal):
        rgb[gt == k + 1] = col
    return rgb


def load_image(img_dir, stem, shape):
    cands = sorted(Path(img_dir).expanduser().glob(stem + '.*'))
    if not cands:
        return np.full(shape + (3,), 200, np.uint8), False
    im = Image.open(cands[0])
    if im.mode not in ('RGB', 'L', 'RGBA', 'P'):
        a = np.asarray(im).astype(np.float64)
        a = 255 * (a - a.min()) / max(a.max() - a.min(), 1e-9)
        im = Image.fromarray(a.astype(np.uint8))
    im = im.convert('RGB')
    if im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im), True


def strip(panels, size, pad=6, header=16):
    """Panels side by side with a clipped title each (PIL default font ~6 px/char)."""
    W = len(panels) * size + (len(panels) + 1) * pad
    out = Image.new('RGB', (W, size + header + 2 * pad), WHITE)
    d = ImageDraw.Draw(out)
    cap = max(6, size // 6)
    for i, (arr, title) in enumerate(panels):
        im = Image.fromarray(arr)
        if im.size != (size, size):
            im = im.resize((size, size), Image.NEAREST)
        x = pad + i * (size + pad)
        out.paste(im, (x, header + pad))
        d.text((x, 3), title[:cap], fill=(0, 0, 0))
    return out


def legend(width):
    items = [(GREEN, 'correct'), (RED, 'wrong class'), (GREY, 'discarded (wrong)'),
             (WHITE, 'no ground truth'), (GREEN, 'changed: fixed'),
             (RED, 'changed: broken'), (YELLOW, 'changed: one error for another'),
             (LIGHT, 'unchanged')]
    im = Image.new('RGB', (max(width, 700), 44), WHITE)
    d = ImageDraw.Draw(im)
    x, y = 8, 6
    for k, (col, txt) in enumerate(items):
        if k == 4:
            x, y = 8, 26
        d.rectangle([x, y, x + 12, y + 12], fill=col, outline=(0, 0, 0))
        d.text((x + 17, y + 1), txt, fill=(0, 0, 0))
        x += 30 + 6 * len(txt)
    return im


def render(stem, L, gt, n, bg, w, tau_c, tau_a, pal, img_dir, size):
    a, c, preds, st = tile_maps(L, gt, n, bg, w, tau_c, tau_a)
    img, found = load_image(img_dir, stem, gt.shape)
    panels = [(img, stem if found else f'{stem} (no image)'),
              (colour_gt(gt, pal), 'ground truth'),
              (colour_codes(a), f'baseline {st["acc_A"]:.1f}%'),
              (colour_codes(c), f'ours {st["acc_C"]:.1f}% ({st["delta"]:+.1f})'),
              (colour_changes(a, c, preds, gt), f'+{st["fixed"]} -{st["broken"]}')]
    return strip(panels, size), st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=sorted(PRESETS))
    ap.add_argument('--cache')
    ap.add_argument('--tau', type=float)
    ap.add_argument('--split')
    ap.add_argument('--scale-cfg')
    ap.add_argument('--img-dir')
    ap.add_argument('--out', required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--auto', action='store_true', help='4 argued tiles on one sheet')
    g.add_argument('--all', action='store_true', help='every held-out tile + index.csv')
    g.add_argument('--tiles', help='comma-separated stems')
    ap.add_argument('--size', type=int, default=256, help='panel size in px')
    ap.add_argument('--limit', type=int, default=0, help='smoke test only')
    args = ap.parse_args()
    if args.preset:
        for k, v in PRESETS[args.preset].items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    for k in ('cache', 'tau', 'split', 'scale_cfg', 'img_dir'):
        if getattr(args, k) is None:
            raise SystemExit(f'⛔ --{k.replace("_", "-")} is required (or use --preset)')

    cache = Path(args.cache).expanduser()
    lab = LB.from_cache(cache)
    n, bg = lab.n, lab.bg - 1
    w = read_vector(args.scale_cfg, 'class_scale', n)
    tau_c = read_vector(args.scale_cfg, 'prob_thd', n)
    if w is None or tau_c is None:
        raise SystemExit(f'⛔ class_scale / prob_thd missing from {args.scale_cfg}')
    tau_a = np.float32(args.tau)
    pal = palette(n)
    stems = [s.strip() for s in Path(args.split).expanduser().read_text().splitlines()
             if s.strip()]
    if args.tiles:
        stems = [s.strip() for s in args.tiles.split(',') if s.strip()]
    if args.limit:
        stems = stems[:args.limit]
    missing = [s for s in stems if not (cache / f'{s}.npz').is_file()]
    if missing:
        raise SystemExit(f'⛔ {len(missing)} tiles have no cache file, e.g. {missing[:3]}')
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    def load(stem):
        z = np.load(cache / f'{stem}.npz')
        return z['logits'].astype(np.float32), z['gt'].astype(np.int64)

    legend(5 * args.size + 36).save(out / 'legend.png')

    if args.tiles or args.all:
        tdir = out / 'tiles' if args.all else out
        tdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, s in enumerate(stems, 1):
            L, gt = load(s)
            if not (gt > 0).any():
                continue
            im, st = render(s, L, gt, n, bg, w, tau_c, tau_a, pal, args.img_dir, args.size)
            im.save(tdir / f'{s}.png')
            rows.append(dict(tile=s, **{k: (round(v, 2) if isinstance(v, float) else v)
                                        for k, v in st.items()}))
            if i % 200 == 0 or i == len(stems):
                print(f'    {i}/{len(stems)}', flush=True)
        if args.all:
            rows.sort(key=lambda r: r['delta'])
            with open(out / 'index.csv', 'w', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
            (out / 'README.md').write_text(
                '# Right/wrong maps, every held-out tile\n\n'
                'Panels: image | ground truth | baseline | ours | changed.  '
                'Colours in `legend.png`. `index.csv` is sorted WORST first by the '
                'change in per-tile pixel accuracy (delta = ours − baseline).\n')
            print(f'  wrote {len(rows)} tiles, index.csv, README.md -> {out}')
        return

    # --auto: one stats pass, then render four tiles chosen to argue, not flatter
    stats = []
    for i, s in enumerate(stems, 1):
        L, gt = load(s)
        if (gt > 0).any():
            stats.append((s, tile_maps(L, gt, n, bg, w, tau_c, tau_a)[3]))
        if i % 200 == 0 or i == len(stems):
            print(f'    stats {i}/{len(stems)}', flush=True)
    moved = [x for x in stats if x[1]['n_gt_classes'] >= 2 and x[1]['changed_pct'] >= 1.0]
    pool = moved or stats
    picks = []

    def take(cands, why):
        for s, st in cands:
            if s not in [p[0] for p in picks]:
                picks.append((s, why))
                print(f'  {why}: {s}  baseline {st["acc_A"]:.1f}% -> ours '
                      f'{st["acc_C"]:.1f}% ({st["delta"]:+.1f})')
                return
    take(sorted(pool, key=lambda x: -x[1]['delta']), 'biggest gain')
    take(sorted(pool, key=lambda x: x[1]['delta']), 'biggest loss')
    take(sorted([x for x in stats if x[1]['acc_A'] >= 80],
                key=lambda x: abs(x[1]['delta'])), 'barely touched')
    med = sorted(stats, key=lambda x: x[1]['delta'])
    take(med[len(med) // 2:] + med[:len(med) // 2], 'median tile')

    strips = []
    for s, why in picks:
        L, gt = load(s)
        im, _ = render(s, L, gt, n, bg, w, tau_c, tau_a, pal, args.img_dir, args.size)
        im.save(out / f'{why.replace(" ", "_")}_{s}.png')
        lab_im = Image.new('RGB', (im.width, 18), WHITE)
        ImageDraw.Draw(lab_im).text((6, 3), why, fill=(0, 0, 0))
        strips += [lab_im, im]
    leg = legend(strips[0].width)
    strips.append(leg)
    sheet = Image.new('RGB', (max(x.width for x in strips), sum(x.height for x in strips)), WHITE)
    y = 0
    for x in strips:
        sheet.paste(x, (0, y))
        y += x.height
    sheet.save(out / 'sheet.png')
    sheet.save(out / 'sheet.pdf')
    print(f'  wrote sheet.png / sheet.pdf and {len(picks)} strips -> {out}')


if __name__ == '__main__':
    main()
