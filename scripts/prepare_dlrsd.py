"""Lay DLRSD out for mmseg, and prove nothing was lost doing it.

⭐ THIS IS THE EASIEST PREPARATION IN THE PROJECT and the script is short for a
reason: `inspect_dataset.py` already showed two flat directories, 2100 stems
matching exactly, and labels that are ALREADY class indices. There is no
sequence flattening (UAVid), no doubled nesting (LoveDA), no RGB palette to
invert, no augmented copies to exclude. So this mostly builds directories.

It does exactly one real conversion, and it is insurance rather than necessity:

⛔ THE LABELS ARE MODE-P (PALETTE) PNGs, AND THAT IS BACKEND-DEPENDENT. Read
with Pillow, a mode-P PNG yields the INDEX plane -- which is what we want, and
what mmseg's LoadAnnotations does by default. Read through OpenCV with
IMREAD_UNCHANGED, the SAME FILE yields a 3-channel BGR image, because OpenCV
applies the palette. Every label would silently become colour data. Nothing
crashes; the numbers are simply wrong.

Rather than depend on which backend mmseg happens to use, the labels are
rewritten as mode-L (plain 8-bit grayscale), where every backend agrees. It
costs ~4 MB and removes the failure mode entirely.

⭐ And the conversion is CHECKED, not trusted: the index plane is compared
before and after, byte for byte, and the run aborts on a single differing
pixel. The class totals must also reproduce 137,625,600 = 2100 x 256 x 256
exactly -- the same self-verifying arithmetic that confirmed UAVid's
597,196,800.

    python scripts/prepare_dlrsd.py --src ~/Downloads/DLRSD --dst ~/data/dlrsd
"""
import argparse
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

NCLS = 17
EXPECT_PX = 2100 * 256 * 256          # 137,625,600
GROUP_RE = re.compile(r'^([A-Za-z_\-]+?)[\-_]?\d+$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--images', default='all_images')
    ap.add_argument('--labels', default='all_labels')
    ap.add_argument('--split', default='val')
    ap.add_argument('--copy-images', action='store_true',
                    help='copy rather than symlink (241 MB). Symlinks are the '
                         'default because disk is the binding constraint on '
                         'this machine, but they break if the download moves.')
    ap.add_argument('--repo', default='~/SegEarth-OV-3')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    img_src, lab_src = src / args.images, src / args.labels
    for d in (img_src, lab_src):
        if not d.is_dir():
            raise SystemExit(f'{d} does not exist')

    imgs = sorted(img_src.glob('*.png'))
    labs = sorted(lab_src.glob('*.png'))
    si, sl = {p.stem for p in imgs}, {p.stem for p in labs}
    if si != sl:
        raise SystemExit(
            f'stems do not match: {len(si - sl)} images without a label, '
            f'{len(sl - si)} labels without an image. A sorted zip would pair '
            f'an image with the wrong label and still evaluate cleanly.')
    print(f'  {len(imgs)} pairs, stems match exactly')

    cats = Counter(GROUP_RE.match(p.stem).group(1) for p in imgs
                   if GROUP_RE.match(p.stem))
    print(f'  {len(cats)} scene categories, sizes '
          f'{min(cats.values())}-{max(cats.values())}')

    dst = Path(args.dst).expanduser()
    img_out = dst / 'img_dir' / args.split
    ann_out = dst / 'ann_dir' / args.split
    if args.dry_run:
        print(f'  would write {img_out} and {ann_out}')
        return
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    # ---- labels: mode-P -> mode-L, verified pixel for pixel
    totals = np.zeros(256, np.int64)
    for i, p in enumerate(labs, 1):
        with Image.open(p) as im:
            if im.mode not in ('P', 'L'):
                raise SystemExit(f'{p.name}: mode {im.mode}, expected P or L. '
                                 f'An RGB label needs a colour map, not this '
                                 f'script.')
            a = np.array(im)
        if a.ndim != 2:
            raise SystemExit(f'{p.name}: got shape {a.shape}, expected 2-D. '
                             f'The palette was expanded -- check the image '
                             f'backend.')
        bad = np.unique(a[(a < 1) | (a > NCLS)])
        if bad.size:
            raise SystemExit(
                f'{p.name}: values outside 1..{NCLS}: {bad.tolist()}. '
                f'Every DLRSD pixel should carry one of {NCLS} real classes '
                f'and 0 should never appear.')
        out = ann_out / p.name
        Image.fromarray(a, mode='L').save(out)

        # ⭐ verify the round trip rather than assume it
        with Image.open(out) as im2:
            b = np.array(im2)
        if not np.array_equal(a, b):
            raise SystemExit(f'{p.name}: the mode-L copy differs from the '
                             f'original index plane. Conversion is not safe; '
                             f'stop.')
        totals += np.bincount(a.ravel(), minlength=256)
        if i % 500 == 0 or i == len(labs):
            print(f'  labels {i}/{len(labs)}', flush=True)

    # ---- images: symlink (or copy)
    import shutil
    for i, p in enumerate(imgs, 1):
        out = img_out / p.name
        if out.exists() or out.is_symlink():
            out.unlink()
        if args.copy_images:
            shutil.copy2(p, out)
        else:
            os.symlink(p.resolve(), out)      # never `ln -s` into an existing dir
        if i % 500 == 0 or i == len(imgs):
            print(f'  images {i}/{len(imgs)}', flush=True)

    # ---- the accounting that makes this trustworthy without a reproduction gate
    tot = int(totals.sum())
    print(f'\n  labelled pixels : {tot:,}')
    print(f'  2100 x 256 x 256: {EXPECT_PX:,}')
    if tot != EXPECT_PX:
        raise SystemExit(f'⛔ pixel total is off by {EXPECT_PX - tot:,}. '
                         f'Something was dropped.')
    if totals[0]:
        raise SystemExit(f'⛔ {totals[0]:,} pixels at value 0. The no-data '
                         f'reading in prereg/predict_dlrsd.md is wrong and the '
                         f'design decisions built on it need revisiting.')
    print('  ✅ exact, and zero pixels at value 0 — no catch-all, no no-data')

    print(f'\n  class shares (mask value -> share of all pixels):')
    for k in range(1, NCLS + 1):
        print(f'    {k:2d}  {100 * totals[k] / tot:6.2f}%')

    repo = Path(args.repo).expanduser()
    print(f'\n✅ written to {dst}')
    print(f'\nNext:\n  bash scripts/install_dlrsd.sh --data {dst} --repo {repo}')


if __name__ == '__main__':
    main()
