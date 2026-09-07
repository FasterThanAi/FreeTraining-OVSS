"""
Cut large OpenAerialMap GeoTIFFs into square tiles the pipeline can consume.

WHY SQUARE. SAM 3 resizes every input to 1008x1008 (ANALYSIS SS4.5). A
non-square tile therefore arrives with its aspect ratio distorted, silently --
buildings become rectangles, the model still returns a plausible mask, and
nothing in the output says the geometry was wrong. Tiles are square here and the
script refuses to emit anything else.

WHY REJECT TILES. OAM scenes are flight strips inside a bounding box, so a large
share of every file is nodata -- black, or alpha 0. A tile that is 80% black is
not a hard example, it is an empty one, and putting it in a demo wastes a GPU
pass and misleads a viewer.

⚠️ SELECTION IS RANDOM, WITH A SEED, AFTER FILTERING. Ranking tiles by variance
or content would be cherry-picking dressed as preprocessing: the interesting
tiles would be chosen by a rule correlated with what the model finds easy. The
seed is printed and recorded so the draw is reproducible.

⚠️ NO GROUND TRUTH EXISTS FOR THIS IMAGERY, so anything run on it is
QUALITATIVE. The fitted presets are also for other datasets' vocabularies and
will switch themselves off. Both are correct and both must be stated wherever
these tiles appear.

Provenance: a manifest CSV records, per tile, the source OAM image id (which is
the filename), the pixel offset, the tile size and the ground sample distance
where the GeoTIFF carries one. ⚠️ The LICENCE is not in the file -- record it
per image id from the OAM page, into the licence column, before publishing any
figure that uses these tiles.

    python scripts/make_tiles.py ~/Downloads/*.tif --out ~/data/oam --size 512 -n 8
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np


def open_reader(path, max_gib):
    """rasterio if present (windowed reads, no full decode), else PIL.

    A 700 MB compressed GeoTIFF can be many gigabytes decoded, so the PIL path
    is a fallback that says so rather than a silent default.
    """
    try:
        import rasterio
        ds = rasterio.open(str(path))
        gsd = abs(ds.transform.a) if ds.transform else None
        def read(x, y, s):
            from rasterio.windows import Window
            a = ds.read(window=Window(x, y, s, s))          # (bands, s, s)
            return np.transpose(a, (1, 2, 0))
        return ds.width, ds.height, gsd, read, 'rasterio', ds.close
    except ImportError:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None                        # these are legitimately huge
        im = Image.open(str(path))
        w, h = im.size
        need = w * h * len(im.getbands()) / 2**30
        if need > max_gib:
            raise MemoryError(
                f'{w}x{h}x{len(im.getbands())} decodes to about {need:.1f} GiB and '
                f'PIL has no windowed read.\n'
                f'       Either raise --max-gib (and risk the OOM killer), or use '
                f'rasterio, which reads\n'
                f'       one window at a time and also recovers the GSD:\n'
                f'         conda create -n tiles -c conda-forge python=3.11 rasterio '
                f'pillow numpy -y\n'
                f'         conda activate tiles && python scripts/make_tiles.py ...\n'
                f'       ⛔ Do NOT install rasterio into segov3.')
        print(f'    ⚠️  rasterio not available — decoding the whole image with PIL '
              f'({w}x{h}, ~{need:.1f} GiB).')
        arr = np.asarray(im)
        if arr.ndim == 2:
            arr = arr[..., None]
        return w, h, None, (lambda x, y, s: arr[y:y + s, x:x + s]), 'pil', (lambda: None)


def usable(tile, max_nodata):
    """Fraction of the tile that is real imagery, and whether it clears the bar."""
    if tile.shape[2] >= 4:                      # alpha band marks nodata explicitly
        valid = tile[..., 3] > 0
    else:
        valid = tile[..., :3].max(axis=2) > 0   # OAM pads with pure black
    frac = float(valid.mean())
    return frac, frac >= (1.0 - max_nodata)


def rebuild(out, size):
    """Reconstruct provenance from the tiles themselves.

    The filename carries everything that cannot be recomputed: `<oam id>_<x>_<y>`.
    The valid fraction is remeasured from the pixels. GSD and licence cannot be
    recovered from a PNG and are left blank for the OAM page to supply.
    """
    from PIL import Image
    d = out / 'img_dir' / 'val'
    tiles = sorted(d.glob('*.png'))
    if not tiles:
        raise SystemExit(f'⛔ no tiles in {d}')
    rows = []
    for t in tiles:
        src, _, rest = t.stem.rpartition('_')       # rest = y
        sid, _, x = src.rpartition('_')
        if not (x.isdigit() and rest.isdigit()):
            print(f'  ⚠️  {t.name} does not follow <id>_<x>_<y>.png, skipped')
            continue
        a = np.asarray(Image.open(t))
        frac = float((a[..., :3].max(axis=2) > 0).mean())
        rows.append(dict(tile=t.name, source_id=sid, x=int(x), y=int(rest),
                         size=a.shape[0], gsd_m='', valid_frac=f'{frac:.3f}',
                         seed='', licence=''))
    man = out / 'manifest.csv'
    with man.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ids = sorted({r['source_id'] for r in rows})
    print(f'  rebuilt {man} from {len(rows)} tiles, {len(ids)} source images:')
    for i in ids:
        print(f'    {i}  ({sum(1 for r in rows if r["source_id"] == i)} tiles)')
    print('\n  ⚠️  `gsd_m` and `licence` are EMPTY and cannot be recovered from a')
    print('      PNG. Fill both from each image id\'s OpenAerialMap page before any')
    print('      figure using these tiles is published.')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sources', nargs='*', help='GeoTIFF files (omit with --rebuild)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--per-image', '-n', type=int, default=8)
    ap.add_argument('--max-nodata', type=float, default=0.05,
                    help='reject a tile with more than this fraction of nodata')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rebuild', action='store_true',
                    help='reconstruct manifest.csv from tiles already in --out, '
                         'for a run that was interrupted before it wrote one')
    ap.add_argument('--max-gib', type=float, default=6.0,
                    help='refuse the PIL path above this decoded size; rasterio, '
                         'which reads windows, is unaffected')
    args = ap.parse_args()

    if args.rebuild:
        return rebuild(Path(args.out).expanduser(), args.size)
    if not args.sources:
        raise SystemExit('⛔ no GeoTIFFs given (and --rebuild not set)')
    if args.size < 64:
        raise SystemExit('⛔ --size below 64 is not a tile, it is a patch')
    out = Path(args.out).expanduser()
    (out / 'img_dir' / 'val').mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows, total = [], 0
    # ⚠️ Written incrementally. An earlier version wrote it only after every file
    # was processed, so an interrupted run left tiles with NO provenance -- which
    # is the one record that cannot be reconstructed later if the source files
    # are deleted. Provenance is flushed as soon as it exists.
    man = out / 'manifest.csv'
    FIELDS = ['tile', 'source_id', 'x', 'y', 'size', 'gsd_m', 'valid_frac',
              'seed', 'licence']
    fresh = not man.exists()
    mf = man.open('a', newline='')
    wcsv = csv.DictWriter(mf, fieldnames=FIELDS)
    if fresh:
        wcsv.writeheader(); mf.flush()

    from PIL import Image
    for src in args.sources:
        src = Path(src).expanduser()
        if not src.exists():
            print(f'  ⛔ {src} not found'); continue
        print(f'  {src.name}  ({src.stat().st_size / 2**20:.0f} MiB)')
        try:
            w, h, gsd, read, backend, close = open_reader(src, args.max_gib)
        except MemoryError as e:
            print(f'    ⛔ skipped: {e}')
            continue
        s = args.size
        print(f'    {w}x{h}, backend {backend}'
              + (f', GSD {gsd * 100:.1f} cm' if gsd else ', GSD unknown'))
        if gsd and not (0.02 <= gsd <= 1.0):
            print(f'    ⚠️  {gsd * 100:.0f} cm is outside everything this pipeline has '
                  f'been evaluated on (4–50 cm). Results here are not comparable.')
        if w < s or h < s:
            print('    ⛔ smaller than one tile, skipped'); close(); continue

        grid = [(x, y) for y in range(0, h - s + 1, s) for x in range(0, w - s + 1, s)]
        rng.shuffle(grid)
        kept = 0
        for x, y in grid:
            if kept >= args.per_image:
                break
            t = read(x, y, s)
            if t.shape[0] != s or t.shape[1] != s:
                continue                                   # ragged edge
            frac, ok = usable(t, args.max_nodata)
            if not ok:
                continue
            name = f'{src.stem}_{x}_{y}.png'
            Image.fromarray(np.ascontiguousarray(t[..., :3])).save(
                out / 'img_dir' / 'val' / name)
            rows.append(dict(tile=name, source_id=src.stem, x=x, y=y, size=s,
                             gsd_m=f'{gsd:.4f}' if gsd else '',
                             valid_frac=f'{frac:.3f}', seed=args.seed, licence=''))
            kept += 1
        wcsv.writerows(rows[total:]); mf.flush()      # provenance, before the next file
        total += kept
        print(f'    wrote {kept} tiles')
        close()

    mf.close()
    if not rows:
        raise SystemExit('⛔ no tiles written — try a larger --max-nodata or a '
                         'smaller --size')
    print(f'\n  {total} tiles -> {out / "img_dir" / "val"}')
    print(f'  manifest      -> {man}')
    print('\n  ⚠️  NEXT, BEFORE ANY FIGURE IS PUBLISHED:')
    print('      fill the `licence` column from each image id\'s OpenAerialMap page.')
    if any(not r['gsd_m'] for r in rows):
        n = sum(1 for r in rows if not r['gsd_m'])
        print(f'      ⚠️  {n} tiles have NO GSD (the PIL path cannot read the GeoTIFF')
        print('          transform). The OAM page states it per image — fill `gsd_m`')
        print('          from there, in metres, or the imagery cannot be placed')
        print('          against the 4-50 cm band the benchmarks occupy.')
    print('      ⚠️  There is no ground truth here. The demo will show baseline vs')
    print('      calibrated with NO IoU, and a preset fitted for another dataset\'s')
    print('      vocabulary will switch itself off. Both are correct.')


if __name__ == '__main__':
    main()
