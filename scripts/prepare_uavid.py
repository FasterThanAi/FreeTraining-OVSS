"""
Prepare UAVid for SegEarth-OV3's loader.

Three things stand between the download and a runnable split, and each one fails
SILENTLY if it is got wrong -- which is the failure mode WEEK3 11 lists four
times over.

1. ⛔ THE TEST SPLIT HAS NO LABELS. `cfg_uavid.py` points at img_dir/test and
   ann_dir/test, but UAVid withholds test annotations (it is a live benchmark);
   `uavid_test/seq*/` contains only `Images`. Their published number must come
   from `val`, and this script prepares val.

2. ⛔ FILENAMES REPEAT ACROSS SEQUENCES. Every sequence has 000000.png,
   000100.png and so on. Flattening seq*/Images into one directory without
   renaming overwrites most of the dataset and leaves a smaller split that still
   evaluates cleanly. Files become seq16_000000.png.

3. ⛔ LABELS ARE RGB, NOT CLASS INDICES. UAVid ships (128,0,0) for building and
   so on. mmseg expects indices. ⭐ The mapping is derived from UAVidDataset's
   own METAINFO palette rather than hardcoded here, so it cannot drift from what
   the loader expects -- and every pixel colour is checked against that palette,
   because an unmapped colour silently becomes whatever the fallback is.

⚠️ UAVid frames are 3840x2160 and SAM 3 resizes every input to 1008x1008, so a
16:9 frame is squashed. Their config adds no Resize, so we inherit that
distortion; it is the baseline's choice and reproducing it is the point.

    python scripts/prepare_uavid.py --src ~/Downloads/uvaid \\
        --dst ~/data/uavid --repo ~/SegEarth-OV-3
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


# ⭐ UAVidDataset's own map, copied verbatim from custom_datasets.py, plus the
# merge its docstring names. Hardcoded rather than parsed: METAINFO is written
# as dict(...) with parentheses and a parser looking for a closing brace finds
# nothing. verify_meta() below re-checks it against the file on every run, so it
# cannot drift silently.
CLASSES = ('background', 'building', 'road', 'car', 'tree', 'vegetation', 'human')
PALETTE = [(0, 0, 0), (128, 0, 0), (128, 64, 128), (192, 0, 192),
           (0, 128, 0), (128, 128, 0), (64, 64, 0)]
# ⛔ THE MERGE THAT IS NOT IN THE PALETTE. UAVidDataset's docstring says
# "convert Moving_Car to Static_Car", and UAVid ships TWO car colours: static
# (192,0,192), which is in the palette, and moving (64,0,128), which is not.
# Without this line every moving car becomes 255/ignore -- a silent partial
# deletion of a class that still evaluates cleanly.
EXTRA = {(64, 0, 128): 3}          # moving car -> `car`


def verify_meta(repo):
    """Re-check the hardcoded map against custom_datasets.py."""
    src = (Path(repo).expanduser() / 'custom_datasets.py').read_text()
    i = src.index('class UAVidDataset')
    block = src[i:i + 2000]
    for c in CLASSES:
        if f"'{c}'" not in block:
            raise SystemExit(f'class {c!r} is no longer in UAVidDataset; '
                             f'the hardcoded map is stale.')
    for r, g, b in PALETTE:
        if f'[{r}, {g}, {b}]' not in block:
            raise SystemExit(f'palette colour {(r, g, b)} is no longer in '
                             f'UAVidDataset; the hardcoded map is stale.')
    return list(CLASSES), list(PALETTE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='the unpacked download')
    ap.add_argument('--dst', required=True, help='where the mmseg split goes')
    ap.add_argument('--repo', default='~/SegEarth-OV-3')
    ap.add_argument('--split', default='val', choices=['val', 'train'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    classes, palette = verify_meta(args.repo)
    print(f'  UAVidDataset: {len(classes)} classes')
    for c, p in zip(classes, palette):
        print(f'    {c:22} {p}')

    src = Path(args.src).expanduser() / f'uavid_{args.split}'
    if not src.is_dir():
        raise SystemExit(f'{src} does not exist')
    seqs = sorted(d for d in src.iterdir() if d.is_dir())
    print(f'\n  {len(seqs)} sequences under {src}')

    dst = Path(args.dst).expanduser()
    img_out, ann_out = dst / 'img_dir' / args.split, dst / 'ann_dir' / args.split

    # ⭐ colour -> index, straight off the loader's own palette
    lut = {tuple(p): i for i, p in enumerate(palette)}
    lut.update(EXTRA)                       # <<< the moving-car merge
    print(f'\n  colour map: {len(lut)} colours -> {len(palette)} classes '
          f'({len(EXTRA)} merged)')
    if len(lut) != len(palette):
        raise SystemExit('the palette contains duplicate colours; the mapping '
                         'would be ambiguous')

    pairs = []
    for s in seqs:
        imgs = sorted((s / 'Images').glob('*.png'))
        if not imgs:
            print(f'  ⚠️ {s.name}: no Images, skipped')
            continue
        labs = sorted((s / 'Labels').glob('*.png'))
        if len(labs) != len(imgs):
            raise SystemExit(
                f'{s.name}: {len(imgs)} images but {len(labs)} labels. '
                f'A split that does not pair up produces a plausible table from '
                f'mismatched data.')
        for im, la in zip(imgs, labs):
            if im.name != la.name:
                raise SystemExit(f'{s.name}: {im.name} pairs with {la.name}')
            pairs.append((s.name, im, la))
    print(f'  {len(pairs)} image/label pairs')
    if args.dry_run:
        for n, im, _ in pairs[:3]:
            print(f'    would write {n}_{im.name}')
        return

    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)
    unmapped = Counter()
    for i, (seq, im, la) in enumerate(pairs, 1):
        stem = f'{seq}_{im.stem}'
        Image.open(im).convert('RGB').save(img_out / f'{stem}.png')

        a = np.array(Image.open(la).convert('RGB'))
        idx = np.full(a.shape[:2], 255, np.uint8)          # 255 = mmseg ignore
        for colour, k in lut.items():
            idx[(a == np.array(colour, np.uint8)).all(-1)] = k
        miss = idx == 255
        if miss.any():
            for c in np.unique(a[miss].reshape(-1, 3), axis=0):
                unmapped[tuple(int(x) for x in c)] += int(
                    ((a == c).all(-1) & miss).sum())
        Image.fromarray(idx).save(ann_out / f'{stem}.png')
        if i % 20 == 0 or i == len(pairs):
            print(f'  {i}/{len(pairs)}', flush=True)

    if unmapped:
        print('\n⛔ COLOURS NOT IN THE PALETTE — the mapping is incomplete:')
        for c, n in unmapped.most_common(12):
            print(f'    {c}: {n:,} px')
        print('   Those pixels are written as 255 (ignore). ⚠️ UAVid has TWO car '
              'colours, static (192,0,192) and moving (64,0,128), which collapse '
              'into one `car` class -- if the palette carries only one of them, '
              'add the other by hand and re-run.')
        raise SystemExit(1)

    print(f'\n✅ {len(pairs)} pairs written to {dst}')
    print(f'\nNext:\n  ln -sfn {dst} {Path(args.repo).expanduser()}/data/UAVid')
    print(f'  # ⚠️ cfg_uavid.py points at test/, which has no labels. Repoint it '
          f'to {args.split}/ before running.')


if __name__ == '__main__':
    main()
