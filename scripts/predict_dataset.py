"""
Pre-register a prediction for a new dataset, from its LABELS ALONE.

WHY BOTHER. A third dataset that is simply measured adds a row. A third dataset
whose behaviour is PREDICTED before the pipeline runs, and then measured, tests
whether the mechanism is a rule or a coincidence. The difference is the whole
value of the exercise, and it costs one extra step: run this first, commit the
output, and only then touch the GPU.

THE MECHANISM (WEEK3_RESULTS.md 7). One variable orders everything measured so
far -- the share of ground truth annotated as a catch-all `background` class:

    dataset        bg share   discard @ tau=0.1   best detection AUC
    LoveDA           36.1%          10.88%              0.622
    OpenEarthMap      0.84%          3.78%              0.913

A catch-all class gives SAM 3 a plausible answer everywhere, so much of the scene
falls into it and a strong runner-up score carries no information. Where the
vocabulary covers the scene, the residual is small and the runner-up finds it.

WHAT THIS SCRIPT WILL AND WILL NOT SAY. Two points define a line, so no
functional form is claimed and none is fitted with confidence. What is predicted
is a REGIME and a bracketed range, which is falsifiable: if a dataset at 5%
background behaves like LoveDA, or one at 30% behaves like OpenEarthMap, the
mechanism is wrong and the paper's central claim needs rewriting.

No model, no GPU, no inference -- it reads ground-truth masks and counts.

    python scripts/predict_dataset.py --masks ~/data/potsdam/labels/val \
        --classes ~/SegEarth-OV-3/configs/cls_potsdam.txt --name Potsdam
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

# The two measured anchors. Update only from WEEK3_RESULTS.md.
ANCHORS = [
    dict(name='LoveDA',       bg=36.10, discard=10.88, auc=0.622, note='catch-all'),
    dict(name='OpenEarthMap', bg=0.84,  discard=3.78,  auc=0.913, note='covering'),
]


def read_masks(paths, limit, reduce_zero_label):
    from PIL import Image
    counts = Counter()
    shapes = set()
    for i, p in enumerate(paths[:limit] if limit else paths):
        a = np.array(Image.open(p))
        if a.ndim == 3:
            a = a[..., 0]
        shapes.add(a.shape)
        v, c = np.unique(a, return_counts=True)
        for vv, cc in zip(v.tolist(), c.tolist()):
            counts[vv] += cc
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(paths)}')
    return counts, shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', required=True, help='directory of GT masks')
    ap.add_argument('--classes', default=None,
                    help='prompt file; one class per LINE, commas separate synonyms')
    ap.add_argument('--name', default='new dataset')
    ap.add_argument('--bg-name', default='background',
                    help='which class name counts as the catch-all (e.g. clutter)')
    ap.add_argument('--reduce-zero-label', type=lambda v: v.lower() == 'true',
                    default=False,
                    help='True = raw 0 is no-data (LoveDA); False = raw 0 is a class (OEM)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.masks).expanduser().iterdir()
                   if p.suffix.lower() in ('.png', '.tif', '.tiff'))
    if not paths:
        raise SystemExit(f'no masks in {args.masks}')
    print(f'{len(paths)} masks | reduce_zero_label={args.reduce_zero_label}\n')
    counts, shapes = read_masks(paths, args.limit, args.reduce_zero_label)

    names = None
    if args.classes:
        names = [ln.split(',')[0].strip()
                 for ln in Path(args.classes).expanduser().read_text().splitlines()
                 if ln.strip()]

    # Map raw mask values to class positions using the same convention as labels.py:
    # reduce_zero_label True  -> raw 0 is no-data, class i is raw value i+1
    # reduce_zero_label False -> raw value i IS class i
    total = 0
    per_class = {}
    for raw, n in counts.items():
        if args.reduce_zero_label:
            if raw == 0:
                continue                      # no-data
            idx = raw - 1
        else:
            if raw == 255:
                continue                      # mmseg ignore
            idx = raw
        per_class[idx] = per_class.get(idx, 0) + n
        total += n

    def cname(i):
        return names[i] if names and i < len(names) else f'class {i}'

    bg_idx = None
    if names:
        low = [n.lower() for n in names]
        if args.bg_name.lower() in low:
            bg_idx = low.index(args.bg_name.lower())
    if bg_idx is None:
        bg_idx = 0
        print(f'  !! no class named "{args.bg_name}"; assuming index 0 '
              f'({cname(0)}). Override with --bg-name.')

    bg_share = 100 * per_class.get(bg_idx, 0) / max(total, 1)

    md = [f'# Pre-registered prediction — {args.name}\n',
          f'- masks: **{len(paths)}**  |  shapes: {sorted(shapes)[:3]}',
          f'- labelled pixels: **{total:,}**',
          f'- catch-all class: **`{cname(bg_idx)}`** (index {bg_idx})',
          f'- **`background` share of GT: {bg_share:.2f}%**\n',
          '## Class composition\n', '| class | pixels | share |', '|---|---|---|']
    for i in sorted(per_class, key=lambda k: -per_class[k]):
        md.append(f'| {cname(i)}{" ⬅ catch-all" if i == bg_idx else ""} | '
                  f'{per_class[i]:,} | {100 * per_class[i] / max(total, 1):.2f}% |')

    lo, hi = min(a['bg'] for a in ANCHORS), max(a['bg'] for a in ANCHORS)
    md += ['\n## The anchors\n',
           '| dataset | bg share | discard @ τ=0.1 | best detection AUC |',
           '|---|---|---|---|']
    for a in ANCHORS:
        md.append(f'| {a["name"]} | {a["bg"]:.2f}% | {a["discard"]:.2f}% | {a["auc"]:.3f} |')
    md.append(f'| **{args.name}** | **{bg_share:.2f}%** | **?** | **?** |\n')

    # Two anchors define a line; interpolate, but present it as a bracket and say
    # plainly that no functional form is being claimed.
    xs = np.array([a['bg'] for a in ANCHORS])
    order = np.argsort(xs)
    xs = xs[order]
    dd = np.array([a['discard'] for a in ANCHORS])[order]
    aa = np.array([a['auc'] for a in ANCHORS])[order]
    p_d = float(np.interp(bg_share, xs, dd))
    p_a = float(np.interp(bg_share, xs, aa))
    inside = lo <= bg_share <= hi

    md += ['## The prediction — written before any inference is run\n']
    if bg_share >= 20:
        regime = ('**catch-all regime**, like LoveDA. Expect a large residual and '
                  'detection near the ~0.53 floor.')
    elif bg_share <= 5:
        regime = ('**covering regime**, like OpenEarthMap. Expect a small residual and '
                  'detection well above the floor.')
    else:
        regime = ('**between the two anchors — the most informative place to land.** '
                  'It distinguishes a graded relationship from a step, which two points '
                  'cannot.')
    md += [f'`{args.name}` sits at **{bg_share:.2f}%**, the {regime}\n',
           '| quantity | point interpolation | honest bracket |',
           '|---|---|---|',
           f'| real-class pixels discarded @ τ=0.1 | ~{p_d:.1f}% | '
           f'{min(dd):.1f}–{max(dd):.1f}% |',
           f'| best detection AUC | ~{p_a:.3f} | {min(aa):.3f}–{max(aa):.3f} |']
    if not inside:
        md.append(f'\n⚠️ **{bg_share:.2f}% is outside the anchor range '
                  f'({lo:.2f}–{hi:.2f}%)**, so those figures are extrapolation. Predict the '
                  'ORDERING only: it should be more extreme than the nearer anchor.')
    # State the ORDERING against each anchor separately. An earlier version tried to
    # do this with one nested conditional and produced a self-contradictory bullet
    # ("below LoveDA's 10.88% and near or below OpenEarthMap's 3.78%") for a dataset
    # sitting between the two.
    ordering = []
    for a in ANCHORS:
        if abs(bg_share - a['bg']) < 0.5:
            rel_d, rel_a = 'about the same as', 'about the same as'
        elif bg_share > a['bg']:
            rel_d, rel_a = 'HIGHER than', 'LOWER than'
        else:
            rel_d, rel_a = 'LOWER than', 'HIGHER than'
        ordering.append(
            f'- vs **{a["name"]}** ({a["bg"]:.2f}% background): discard should be '
            f'**{rel_d}** {a["discard"]:.2f}%, and detection AUC **{rel_a}** '
            f'{a["auc"]:.3f}')

    md += ['\n**Two points define a line, so no functional form is claimed.** What is '
           'falsifiable is the ORDERING against each anchor — more catch-all means more '
           'discard and worse detection:\n'] + ordering + [
           f'- detection AUC should be **{"near the 0.53 floor" if bg_share > 20 else "well above 0.53"}**',
           '- `S_pres(background)` should stay far below the real classes either way — that '
           'part is a property of SAM 3, not of the dataset, and should NOT move\n',
           '> ⛔ **If a dataset near 5% background behaves like LoveDA, or one near 30% '
           'behaves like OpenEarthMap, the mechanism is wrong** and the paper\'s central '
           'claim needs rewriting. That is the point of running this before the pipeline '
           'rather than after.\n',
           '## Next\n',
           '1. Commit this file **before** running any inference.',
           '2. `measure_discard_rate.py` with this dataset\'s config and its own τ.',
           '3. `recoverability_signal.py` for the detection AUC.',
           '4. Compare against the bracket above and record whether the prediction held.']

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
