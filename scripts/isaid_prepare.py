"""
iSAID: convert RGB colour-coded semantic masks to integer class indices.

iSAID ships `*_instance_color_RGB.png` -- masks where the class is encoded as a
COLOUR, not an index. Every script in this repo reads a mask as an integer class
map, and PIL would happily hand them a 3-channel array that numpy indexes without
complaint. That is the silent-corruption failure mode this project has already
hit four times, so this converter refuses to guess: any colour not in the palette
aborts the run rather than falling through to class 0.

THE PALETTE was read off the data, not recalled. The 15 sampled val masks
contained exactly twelve distinct colours, all of which match the standard iSAID
assignment against cls_iSAID.txt's class order. The four absent from that sample
(basketball court, bridge, helicopter, plane) are included below and are verified
on the full split by --check, which reports any colour it cannot place.

    # verify the palette covers every colour present, without writing anything
    python scripts/isaid_prepare.py --src <Semantic_masks/images/images> --check

    # convert
    python scripts/isaid_prepare.py \
        --src ~/data/isaid_raw/ValidationData/val/Semantic_masks/images/images \
        --out ~/data/isaid/ann_dir/val
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

# index -> (R, G, B), in cls_iSAID.txt order. Index 0 is background.
PALETTE = [
    ('background',          (0, 0, 0)),
    ('ship',                (0, 0, 63)),
    ('store tank',          (0, 63, 63)),
    ('baseball diamond',    (0, 63, 0)),
    ('tennis court',        (0, 63, 127)),
    ('basketball court',    (0, 63, 191)),
    ('ground track field',  (0, 63, 255)),
    ('bridge',              (0, 127, 63)),
    ('large vehicle',       (0, 127, 127)),
    ('small vehicle',       (0, 0, 127)),
    ('helicopter',          (0, 0, 191)),
    ('swimming pool',       (0, 0, 255)),
    ('roundabout',          (0, 191, 127)),
    ('soccer ball field',   (0, 127, 191)),
    ('plane',               (0, 127, 255)),
    ('harbor',              (0, 100, 155)),
]
LUT = {rgb: i for i, (_, rgb) in enumerate(PALETTE)}


def key(a):
    """Pack an HxWx3 uint8 array into one int32 per pixel, for fast lookup."""
    a = a.astype(np.int32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='dir of *_instance_color_RGB.png')
    ap.add_argument('--out', default=None, help='where to write index masks')
    ap.add_argument('--check', action='store_true',
                    help='scan for unmapped colours and exit without writing')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None                # iSAID tiles exceed PIL's bomb guard

    files = sorted(Path(args.src).expanduser().glob('*.png'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .png in {args.src}')
    print(f'{len(files)} masks | {"CHECK ONLY" if args.check else "converting"}\n')

    packed = {(r << 16) | (g << 8) | b: i for i, (_, (r, g, b)) in enumerate(PALETTE)}
    keys = np.array(sorted(packed))
    vals = np.array([packed[k] for k in keys], np.uint8)

    unknown = Counter()
    counts = np.zeros(len(PALETTE), np.int64)
    out = Path(args.out).expanduser() if args.out else None
    if out and not args.check:
        out.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files):
        a = np.array(Image.open(f).convert('RGB'))
        k = key(a)
        pos = np.searchsorted(keys, k)
        pos_c = np.clip(pos, 0, len(keys) - 1)
        hit = keys[pos_c] == k
        if not hit.all():
            for u, c in zip(*np.unique(k[~hit], return_counts=True)):
                unknown[(int(u) >> 16 & 255, int(u) >> 8 & 255, int(u) & 255)] += int(c)
        idx = vals[pos_c]
        idx[~hit] = 0                            # only reached if --check lets it pass
        counts += np.bincount(idx.ravel(), minlength=len(PALETTE))

        if out and not args.check:
            # iSAID names masks P0003_instance_color_RGB.png; the image is P0003.png
            stem = f.stem.replace('_instance_color_RGB', '')
            Image.fromarray(idx).save(out / f'{stem}.png')
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}')

    if unknown:
        print('\n⛔ COLOURS NOT IN THE PALETTE — conversion is NOT safe:\n')
        for c, n in unknown.most_common(20):
            print(f'    {c}  {n:,} px')
        raise SystemExit(
            'Every unmapped colour would silently become `background`, inflating the '
            'one class this project measures. Add them to PALETTE (in cls_iSAID.txt '
            'order) and re-run.')

    tot = counts.sum()
    print(f'\n✅ every colour mapped. {tot:,} pixels over {len(files)} masks\n')
    print(f'{"class":22s} {"pixels":>16s}   share')
    for i, (name, _) in enumerate(PALETTE):
        print(f'{name:22s} {counts[i]:>16,}   {100 * counts[i] / max(tot, 1):6.3f}%')
    print(f'\n`background` share of GT: {100 * counts[0] / max(tot, 1):.2f}%')
    if out and not args.check:
        print(f'\nwritten to {out}')
    elif args.check:
        print('\ncheck only — nothing written. Re-run with --out to convert.')


if __name__ == '__main__':
    main()
