"""Pin DLRSD's index -> class name from the DATA, and decide the fold protocol.

TWO QUESTIONS, ONE PASS over the labels. Both are cheap, CPU-only, and both
have to be answered before a single GPU second is spent.

1. WHICH INDEX IS WHICH CLASS. The labels are mode-P PNGs carrying 1..17 and
   the published class list is alphabetical -- airplane, bare soil, buildings,
   cars, chaparral, court, dock, field, grass, mobile home, pavement, sand,
   sea, ship, tanks, trees, water -- so index i == the i-th name is the obvious
   reading. ⛔ It is also a GUESS, and WEEK3 11 records what that costs: a
   hardcoded ladder would have labelled `grass` as `building` on OpenEarthMap
   and printed a clean table with the wrong row names. Nothing crashes.

   ⭐ DLRSD can check itself, because it is built on UC Merced: 21 scene
   categories x 100 images, and the category is in the filename. A class that
   is named `airplane` must occur overwhelmingly in `airplane*` files. That
   fingerprint pins the mapping with no external legend and no trust in
   anybody's memory.

2. ARE THE CLASSES CATEGORY-LOCKED. This decides the fold protocol and it is
   not a detail. Group-disjoint folds are mandatory wherever groups exist --
   on UAVid a frame-level split leaked +0.54 mIoU and two folds. But grouping
   DLRSD by category makes every fold a CROSS-CATEGORY TRANSFER test, and if
   `airplane` pixels live only in the `airplane` category then holding that
   category out leaves the class with no calibration data at all. The fit
   would then be asked to do something impossible and the failure would look
   like the method's.

       classes spread across many categories -> group-disjoint folds are fine
       classes locked to one category         -> stratify WITHIN category
                                                 instead, and say why

    python scripts/dlrsd_class_map.py --root ~/Downloads/DLRSD
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Published class list, Shao et al. -- DLRSD is the dense-labelling extension of
# UC Merced. Order as published (alphabetical). This is the HYPOTHESIS that the
# category fingerprint below either confirms or refutes; it is never assumed.
DLRSD_CLASSES = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral',
                 'court', 'dock', 'field', 'grass', 'mobile home', 'pavement',
                 'sand', 'sea', 'ship', 'tanks', 'trees', 'water']

# A class name that is also a UCM category name gives a checkable prediction.
# Left = our hypothesised class, right = the category its pixels should
# concentrate in. Only the unambiguous ones; `buildings`, `pavement`, `trees`
# and `grass` are everywhere by nature and predict nothing.
FINGERPRINT = {
    'airplane': ['airplane'],
    'chaparral': ['chaparral'],
    'mobile home': ['mobilehomepark'],
    'tanks': ['storagetanks'],
    'ship': ['harbor'],
    'dock': ['harbor'],
    'court': ['tenniscourt', 'baseballdiamond'],
    'sea': ['beach', 'harbor'],
    'field': ['agricultural', 'golfcourse'],
}
GROUP_RE = re.compile(r'^([A-Za-z_\-]+?)[\-_]?\d+$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--labels', default='all_labels')
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    lab_dir = Path(args.root).expanduser() / args.labels
    files = sorted(lab_dir.glob('*.png'))
    if not files:
        raise SystemExit(f'no label PNGs in {lab_dir}')

    nc = len(DLRSD_CLASSES)
    cats = sorted({GROUP_RE.match(f.stem).group(1) for f in files
                   if GROUP_RE.match(f.stem)})
    ci = {c: i for i, c in enumerate(cats)}
    # px[category, class_index]  and  img[category, class_index] = images containing it
    px = np.zeros((len(cats), nc + 1), np.int64)
    img = np.zeros((len(cats), nc + 1), np.int64)

    for n, f in enumerate(files, 1):
        m = GROUP_RE.match(f.stem)
        if not m:
            continue
        r = ci[m.group(1)]
        a = np.array(Image.open(f))
        v, c = np.unique(a, return_counts=True)
        keep = v <= nc
        px[r, v[keep]] += c[keep]
        img[r, v[keep]] += 1
        if n % 500 == 0 or n == len(files):
            print(f'  {n}/{len(files)}', flush=True)

    tot = px.sum(0)
    L = ['# DLRSD — index → class, and the fold protocol\n',
         f'- labels: `{lab_dir}`  |  files: **{len(files)}**  |  '
         f'categories: **{len(cats)}**  |  pixels: **{px.sum():,}**\n']

    # ---- 1. the fingerprint
    L += ['## 1. Does index *i* really mean the *i*-th published class?\n',
          'Each row is a class whose name is also a UC Merced scene category, so it '
          'makes a checkable prediction: its pixels must concentrate in that '
          'category. ⭐ This pins the mapping from the data — no external legend, '
          'no trust in anyone\'s recollection.\n',
          '| idx | hypothesised class | expected category | share of its pixels there | top category actually seen | |',
          '|---|---|---|---|---|---|']
    passes = fails = 0
    for i, name in enumerate(DLRSD_CLASSES, start=1):
        if name not in FINGERPRINT:
            continue
        want = [w for w in FINGERPRINT[name] if w in ci]
        if not want or tot[i] == 0:
            continue
        got = px[[ci[w] for w in want], i].sum() / tot[i]
        top = cats[int(np.argmax(px[:, i]))]
        ok = got >= 0.5
        passes += ok
        fails += not ok
        L.append(f'| {i} | `{name}` | {", ".join(want)} | **{100 * got:.1f}%** | '
                 f'`{top}` | {"✅" if ok else "⛔"} |')
    L.append('')
    if fails == 0 and passes >= 5:
        L.append(f'✅ **All {passes} checkable classes concentrate where the '
                 f'alphabetical mapping says they should.** Index *i* = the *i*-th '
                 f'published class is confirmed by the data, not assumed.\n')
    else:
        L.append(f'⛔ **{fails} of {passes + fails} checks FAILED.** The '
                 f'alphabetical mapping is wrong or the palette is ordered '
                 f'differently. **Do not run anything** until this resolves — a '
                 f'wrong ladder produces a clean table with wrong row names.\n')

    # ---- 2. class totals
    L += ['## 2. Class sizes\n',
          '| idx | class | pixels | share | categories present | images present |',
          '|---|---|---|---|---|---|']
    for i, name in enumerate(DLRSD_CLASSES, start=1):
        ncat = int((px[:, i] > 0).sum())
        nimg = int(img[:, i].sum())
        L.append(f'| {i} | `{name}` | {tot[i]:,} | {100 * tot[i] / px.sum():.2f}% | '
                 f'**{ncat}**/{len(cats)} | {nimg} |')
    L.append('')
    if tot[0] > 0:
        L.append(f'⚠️ **Index 0 carries {tot[0]:,} pixels** — the no-data reading '
                 f'needs revisiting.\n')
    else:
        L.append('⭐ **Index 0 is empty and no class is a catch-all**, so full mIoU '
                 'equals catch-all-excluded mIoU by construction. No result on this '
                 'dataset can be a repaired catch-all (WEEK3 §9h) — the objection '
                 'that qualified OEM +2.28, ConInfer +6.01 and Potsdam\'s −8.18 '
                 'baseline distortion cannot arise here.\n')

    # ---- 3. the protocol decision
    locked = [(DLRSD_CLASSES[i - 1], int((px[:, i] > 0).sum()))
              for i in range(1, nc + 1) if 0 < (px[:, i] > 0).sum() <= 2]
    conc = []
    for i in range(1, nc + 1):
        if tot[i] == 0:
            continue
        conc.append((DLRSD_CLASSES[i - 1], px[:, i].max() / tot[i],
                     int((px[:, i] > 0).sum())))
    conc.sort(key=lambda r: -r[1])

    L += ['## 3. Fold protocol — group-disjoint, or stratified?\n',
          'A 5-fold over 21 categories holds out ~4 categories per fold. That is only '
          'legitimate if the held-out categories\' classes also appear in the '
          'calibration categories.\n',
          '| class | categories present | largest category\'s share of its pixels |',
          '|---|---|---|']
    for name, share, ncat in conc[:8]:
        L.append(f'| `{name}` | **{ncat}**/{len(cats)} | **{100 * share:.1f}%** |')
    L.append('')
    if locked:
        L += [f'⛔ **{len(locked)} class(es) live in ≤ 2 categories**: '
              + ', '.join(f'`{n}` ({k})' for n, k in locked) + '.',
              '',
              '> ⛔ **Category-disjoint folds are therefore NOT a decorrelation '
              'control here — they are a cross-category transfer test, and for these '
              'classes an impossible one**: hold the category out and the class has '
              'no calibration pixels at all. The fit would be asked to set a '
              'threshold it cannot see, and the failure would read as the method\'s.',
              '',
              '⭐ **Use stratified folds instead — split each category\'s 100 images '
              'across the 5 folds — so every fold sees every class.** Then run '
              'category-disjoint as a SECOND arm and report both, exactly as UAVid '
              'did: there the two protocols differed by +0.54 mIoU and two folds, '
              'and that difference was itself the finding.\n']
    else:
        L += ['✅ **Every class appears in at least 3 categories**, so '
              'category-disjoint folds are a decorrelation control rather than a '
              'transfer test. Use `--group-re` and quote the grouped number.\n']

    text = '\n'.join(L)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
