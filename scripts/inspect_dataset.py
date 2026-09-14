"""What is actually in this download? Answer before writing a prep script.

Every dataset added to this project has cost at least one wrong number to a
SILENT structural surprise, never to a crash:

  LoveDA   doubled `Val/Val/` nesting (WEEK1 3)
  UAVid    filenames repeat across sequences; flattening overwrites most of the
           split and the remainder still evaluates cleanly (prepare_uavid.py 2)
  UAVid    labels are RGB, not class indices, and the moving-car colour is
           absent from the palette -- a silent partial deletion of a class
  UAVid    "train" is 200 real frames + 400 augmented copies (UAVID 8)
  Potsdam  the catch-all is the HIGHEST class value, not the lowest, so
           `real = g > BACKGROUND` matched nothing and still produced a
           plausible 5.55% (WEEK3 11)
  OEM      `reduce_zero_label` differs from LoveDA's, and `--limit` is
           rural-only because the ID ranges are disjoint

⭐ The common shape: the pipeline runs, the table looks reasonable, and the
number is wrong. So this script REPORTS rather than assumes, and it is loud
about the things that have historically been silent.

It writes nothing and changes nothing. Point it at a raw download.

    python scripts/inspect_dataset.py --root ~/Downloads/DLRSD
    python scripts/inspect_dataset.py --root ~/Downloads/DLRSD --md ~/outputs/dlrsd_inspect.md
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # these are aerial tiles, not decompression bombs

IMG_EXT = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}
# Filenames of the form <word><digits> carry a group. UCM/DLRSD is
# `airplane00.tif` ... `airplane99.tif`, 21 categories x 100 -- and images
# within a category are cut from the same source photograph, so a fold split
# that ignores this leaks exactly the way UAVid's flight sequences did
# (worth +0.54 mIoU and two folds there).
GROUP_RE = re.compile(r'^([A-Za-z_\-]+?)[\-_]?\d+$')


def human(n):
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.1f} {u}'
        n /= 1024
    return f'{n:.1f} PB'


def walk(root, max_depth=3):
    """Directory shape, depth-limited. Catches the doubled-nesting trap."""
    out = []
    root = Path(root)
    for d in sorted(p for p in root.rglob('*') if p.is_dir()):
        rel = d.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        files = [f for f in d.iterdir() if f.is_file()]
        exts = Counter(f.suffix.lower() for f in files)
        out.append((rel, len(files), exts))
    return out


def probe_image(p, n_colours=64):
    """Open one file and report what it really is."""
    with Image.open(p) as im:
        mode, size = im.mode, im.size
        pal = im.getpalette()
        a = np.array(im)
    info = {'mode': mode, 'size': size, 'shape': a.shape,
            'dtype': str(a.dtype), 'has_palette': pal is not None}
    if pal:
        # A mode-P PNG already stores class INDICES; np.array gives the index
        # plane, not RGB. The palette is still worth dumping, because the
        # colours are how a dataset's published legend names each index -- and
        # WEEK3 11 records a near miss where a hardcoded ladder would have
        # labelled `grass` as `building` and printed a clean table with wrong
        # row names. Verify index -> name against the legend, never assume it.
        rgb = [tuple(pal[i * 3:i * 3 + 3]) for i in range(len(pal) // 3)]
        info['palette'] = rgb
    if a.ndim == 2:
        vals, cnt = np.unique(a, return_counts=True)
        info['kind'] = 'index'
        info['values'] = list(zip(vals.tolist(), cnt.tolist()))[:n_colours]
        info['n_values'] = int(vals.size)
    elif a.ndim == 3:
        flat = a.reshape(-1, a.shape[2])
        uniq, cnt = np.unique(flat, axis=0, return_counts=True)
        info['kind'] = 'rgb' if a.shape[2] >= 3 else 'multi'
        order = np.argsort(-cnt)
        info['values'] = [(tuple(int(x) for x in uniq[i]), int(cnt[i]))
                          for i in order[:n_colours]]
        info['n_values'] = int(uniq.shape[0])
    return info


def scan_dir(files, sample, n_colours):
    """Sizes, modes, and the union of label values over a sample."""
    sizes, modes, union = Counter(), Counter(), Counter()
    step = max(1, len(files) // sample) if sample else 1
    probed = files[::step][:sample] if sample else files
    kind, dtype = None, None
    for p in probed:
        try:
            info = probe_image(p, n_colours)
        except Exception as e:                      # noqa: BLE001
            print(f'  !! cannot open {p.name}: {e}')
            continue
        sizes[info['size']] += 1
        modes[info['mode']] += 1
        kind = kind or info['kind']
        dtype = dtype or info['dtype']
        for v, c in info['values']:
            union[v] += c
    return sizes, modes, union, kind, dtype, len(probed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='the unpacked download')
    ap.add_argument('--sample', type=int, default=60,
                    help='how many files per directory to open (0 = all). '
                         'The value union is over THIS SAMPLE, so a rare class '
                         'can be missed -- raise it before trusting a class list.')
    ap.add_argument('--colours', type=int, default=64)
    ap.add_argument('--palette', action='store_true',
                    help='dump the PNG palette of mode-P labels, so index -> '
                         'colour can be checked against the published legend '
                         'instead of assumed')
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        raise SystemExit(f'{root} does not exist')

    L = [f'# What is in `{root}`\n']
    print(f'\n=== {root} ===\n')

    # ---- 1. shape of the tree
    dirs = walk(root)
    L += ['## Directory shape\n', '| path | files | extensions |', '|---|---|---|']
    L.append(f'| `.` (root) | {sum(1 for f in root.iterdir() if f.is_file())} | '
             f'{dict(Counter(f.suffix.lower() for f in root.iterdir() if f.is_file()))} |')
    for rel, n, exts in dirs:
        L.append(f'| `{rel}` | {n} | {dict(exts)} |')
    L.append('')
    if len(dirs) > 40:
        L.append(f'⚠️ **{len(dirs)} directories.** If images are split across many '
                 f'subdirectories, check for repeated filenames before flattening — '
                 f'UAVid lost most of a split that way and still evaluated cleanly.\n')

    # ---- 2. which directories hold images
    buckets = defaultdict(list)
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            buckets[p.parent].append(p)
    for k in buckets:
        buckets[k].sort()

    L += ['## Image-bearing directories\n']
    if not buckets:
        L.append('⛔ **No image files found at any depth.** Check the unpack.\n')
        print('\n'.join(L))
        return

    order = sorted(buckets, key=lambda d: -len(buckets[d]))
    for d in order:
        files = buckets[d]
        rel = d.relative_to(root)
        total = sum(f.stat().st_size for f in files)
        sizes, modes, union, kind, dtype, nprobed = scan_dir(files, args.sample,
                                                             args.colours)
        L += [f'### `{rel}` — **{len(files)} files**, {human(total)}\n',
              f'- probed **{nprobed}**  |  kind: **{kind}**  |  dtype: `{dtype}`',
              f'- modes: {dict(modes)}',
              f'- sizes: {dict(list(sizes.items())[:8])}'
              + ('  ⚠️ **more than 8 distinct sizes**' if len(sizes) > 8 else '')]

        if len(sizes) == 1:
            (w, h), = sizes
            if w != h:
                L.append(f'- ⛔ **non-square ({w}x{h}).** SAM 3 resizes every input '
                         f'to 1008x1008, so the aspect ratio is distorted, and `d4` '
                         f'TTA refuses non-square input by design.')
            if max(w, h) < 1008:
                L.append(f'- ⭐ **{w}x{h} is SMALLER than SAM 3\'s 1008x1008 working '
                         f'resolution, so every tile is UPSAMPLED ~{1008 / max(w, h):.1f}x. '
                         f'No dataset in this project has run in that regime** — LoveDA '
                         f'is ~1:1 and UAVid is downsampled. Treat inference cost, '
                         f'discard rate and per-class behaviour as unmeasured here.')

        L.append(f'- distinct values over the sample: **{len(union)}**')
        if args.palette:
            try:
                pinfo = probe_image(files[0], args.colours)
            except Exception:                            # noqa: BLE001
                pinfo = {}
            if pinfo.get('palette'):
                rgb = pinfo['palette']
                used = sorted(union) if kind == 'index' else []
                nz = max(used) + 1 if used else len(rgb)
                L += ['', f'#### Palette of `{files[0].name}` '
                          f'(first {nz} of {len(rgb)} entries)', '',
                      '| index | RGB | in this sample |', '|---|---|---|']
                for i, c in enumerate(rgb[:nz]):
                    L.append(f'| {i} | `{c}` | '
                             f'{"yes" if i in union else "—"} |')
                L.append('')
        if kind == 'rgb' and len(union) <= args.colours:
            L += ['', '| colour | pixels | share |', '|---|---|---|']
            tot = sum(union.values())
            for v, c in union.most_common():
                L.append(f'| `{v}` | {c:,} | {100 * c / tot:.2f}% |')
            L.append('')
            L.append('⛔ **RGB labels are NOT class indices.** mmseg needs indices; a '
                     'colour missing from the mapping becomes 255/ignore and silently '
                     'deletes part of a class. UAVid ships two car colours and only one '
                     'is in the palette.\n')
        elif kind == 'index':
            L += ['', '| value | pixels | share |', '|---|---|---|']
            tot = sum(union.values())
            for v, c in sorted(union.items()):
                L.append(f'| {v} | {c:,} | {100 * c / tot:.2f}% |')
            L.append('')
            if 0 in union:
                L.append('⚠️ Value `0` IS present. Decide whether it is '
                         '**no-data/ignore** (LoveDA, `reduce_zero_label=True`) or a '
                         '**real class** (OpenEarthMap, `reduce_zero_label=False`). '
                         'Getting this backwards deletes a class or shifts every '
                         'label by one, and neither crashes.\n')
            else:
                nprobe = 'ALL files' if not args.sample else f'{nprobed} sampled files'
                L.append(f'- ⭐ **Value `0` never appears** across {nprobe}, and the '
                         f'values run 1..{max(union)} with no gap' +
                         ('' if sorted(union) == list(range(1, max(union) + 1))
                          else ' ⚠️ **(there ARE gaps — check)**') +
                         f'. That matches this repo\'s own convention '
                         f'(`labels.py`: mask value 0 = no-data, i+1 = classes[i]), '
                         f'so `reduce_zero_label=True` is the likely setting — but '
                         f'confirm it, because here 0 means *no pixel is unlabelled* '
                         f'rather than *some are*.')
                L.append(f'- ⛔⭐ **Every pixel carries a real class, so this dataset '
                         f'may have NO CATCH-ALL.** Check the legend: if none of the '
                         f'{max(union)} names is background/clutter/unlabelled, then '
                         f'(a) `labels.py` will not find a catch-all and will guess '
                         f'mask value 1 — pass `bg_name=` or fix it explicitly; and '
                         f'(b) full mIoU equals catch-all-excluded mIoU by '
                         f'construction, so no result here can be a repaired '
                         f'catch-all (WEEK3 9h).\n')
        else:
            L.append(f'- ⚠️ more than {args.colours} distinct values; raise `--colours`\n')

        # ---- group structure hiding in the filenames
        groups = Counter()
        ungrouped = 0
        for f in files:
            m = GROUP_RE.match(f.stem)
            if m:
                groups[m.group(1)] += 1
            else:
                ungrouped += 1
        if groups and ungrouped < len(files) * 0.5:
            L += [f'- ⭐ **filenames imply {len(groups)} groups** '
                  f'(e.g. {", ".join(list(groups)[:6])}), sizes '
                  f'{min(groups.values())}–{max(groups.values())}, '
                  f'{ungrouped} ungrouped',
                  f'- ⛔ **Fold splits must be group-disjoint.** On UAVid a '
                  f'frame-level 5-fold straddled every group and reported '
                  f'+1.05 ± 0.86 (5/5) where the truth was +0.51 ± 0.95 (3/5) — '
                  f'the leak was worth **+0.54 mIoU and two folds**. Use '
                  f'`tau_cv.py --group-re`.\n']
        else:
            L.append('- no group structure detected in the filenames\n')

    # ---- 3. do the two largest directories pair up?
    if len(order) >= 2:
        a, b = order[0], order[1]
        sa = {f.stem for f in buckets[a]}
        sb = {f.stem for f in buckets[b]}
        L += ['## Pairing\n',
              f'- `{a.relative_to(root)}`: {len(sa)} stems',
              f'- `{b.relative_to(root)}`: {len(sb)} stems',
              f'- **shared stems: {len(sa & sb)}**',
              f'- only in the first: {len(sa - sb)}  |  only in the second: {len(sb - sa)}']
        if sa == sb:
            L.append('\n✅ **Stems match exactly**, so image/label pairing is by name '
                     'and a sorted zip is safe.\n')
        else:
            ex_a = sorted(sa - sb)[:5]
            ex_b = sorted(sb - sa)[:5]
            L += ['', '⛔ **Stems do NOT match.** A sorted `zip` would pair an image '
                  'with the wrong label and produce a plausible table from mismatched '
                  'data — pair by name explicitly and fail on a miss.',
                  f'- unmatched examples: {ex_a} / {ex_b}\n']
        # duplicate stems across directories anywhere
        allstems = Counter(f.stem for fs in buckets.values() for f in fs)
        dup = [s for s, n in allstems.items() if n > 2]
        if dup:
            L.append(f'⚠️ **{len(dup)} stems appear in more than two directories** '
                     f'({dup[:5]}). Flattening would overwrite.\n')

    text = '\n'.join(L)
    print(text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
