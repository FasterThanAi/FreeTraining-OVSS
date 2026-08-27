"""
Week 3 — are the ATOMS the problem?

WHERE THIS COMES FROM
---------------------
Five detection signals have now come back blind (recoverability_signal.py):
conf 0.582, fconf 0.559, gap 0.558, spres_arg 0.520, spres_max 0.434, fgap
0.447 -- against a 43.1% base rate. Meanwhile the oracle run shows a plain
neighbour vote is worth +3.47 mIoU IF it is applied to the right pixels.

Every one of those tests was PIXEL-level. The method is REGION-level, and the
atoms it has been using are connected components of "assigned to background" --
43% of a typical tile. A connected component of that is not an object; it is the
union of everything the model was unsure about, and it sprawls across the tile
touching real-class regions and genuine background alike.

If those atoms are internally mixed, then a per-region label is wrong for a large
share of its pixels BY CONSTRUCTION, region-mean confidence averages two
populations into mush, and the neighbour vote is being asked a question with no
right answer. Every negative result so far would be explained by bad atoms rather
than by absent signal.

ROADMAP Week 8 lists this as an open choice -- "SLIC atoms vs SAM 3 mask atoms"
-- and it was jumped over, because connected components were free from the cache.

WHAT THIS MEASURES
------------------
1. PURITY. For each atom, the share of its pixels belonging to its own majority
   GT class. Pixel-weighted, because large atoms carry the mass. An atom at 1.00
   is perfectly labelable; at 0.50 it is a coin flip no method can win.

2. CEILING. Assign every atom its own majority GT class -- a perfect oracle
   labeller. The resulting accuracy is the hard upper bound for ANY method built
   on these atoms. If it is low, the atoms are the bug, not the scoring function.

3. DETECTION, at region level. Aggregate each atom's confidence and ask whether
   that separates real-class atoms from background atoms, which pixel-level tests
   could not. Averaging over thousands of pixels kills independent noise, so a
   0.58 pixel AUC can become much higher per region -- or stay flat, which would
   say the signal is genuinely absent rather than merely noisy.

Compare `--atoms cc` (what we have been using) against `--atoms slic`, which
needs the images and scikit-image. If SLIC purity is much higher, atomisation is
the fix and SAM 3's own mask proposals -- better still -- are the next step.

    python scripts/atom_quality.py --cache ~/outputs/week3_fused/cache --tau 0.5
    python scripts/atom_quality.py --cache ~/outputs/week3_fused/cache --tau 0.5 \
        --atoms slic --img-dir ~/data/loveda/img_dir/val
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

NB = 512


def auc_from_hist(pos, neg):
    p = pos[::-1].astype(float); n = neg[::-1].astype(float)
    P, N = p.sum(), n.sum()
    if P == 0 or N == 0:
        return float('nan')
    tpr = np.concatenate([[0.0], np.cumsum(p) / P])
    fpr = np.concatenate([[0.0], np.cumsum(n) / N])
    f = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(f(tpr, fpr))


def atoms_cc(mask, *_):
    from scipy.ndimage import label
    lab, n = label(mask)
    return lab.astype(np.int32), n


def atoms_slic(mask, img, n_segments):
    """SLIC over the whole image, then intersected with the candidate mask.

    SLIC respects colour/texture edges, so an atom should not straddle a real
    boundary the way a connected component does. ANALYSIS 3.4: atomise first,
    agglomerate second -- this measures whether stage 1 is worth having.
    """
    from skimage.segmentation import slic
    seg = slic(img, n_segments=n_segments, compactness=10.0, start_label=1)
    seg = np.where(mask, seg, 0)
    u = np.unique(seg)
    u = u[u > 0]
    remap = np.zeros(seg.max() + 1, np.int32)
    remap[u] = np.arange(1, len(u) + 1)
    return remap[seg], len(u)


def _init_labels(cache):
    """Resolve class names from the cache. See labels.py -- background is located
    BY NAME, never assumed to sit at index 0, because pointing these scripts at a
    dataset with a different class order would otherwise compute nonsense against
    perfectly valid array indices."""
    global LOVEDA, NC, REAL, BG, CLASSES, LB
    LB = labels.from_cache(cache)
    CLASSES = LB.names
    LOVEDA = ['unknown'] + LB.names
    NC = LB.nc
    REAL = LB.real
    BG = LB.bg
    print(f'  classes: {LB}')
    return LB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--atoms', choices=['cc', 'slic'], default='cc')
    ap.add_argument('--img-dir', default=None, help='required for --atoms slic')
    ap.add_argument('--n-segments', type=int, default=600)
    ap.add_argument('--min-size', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()
    _init_labels(args.cache)
    warnings.filterwarnings('ignore')

    if args.atoms == 'slic' and not args.img_dir:
        raise SystemExit('--atoms slic needs --img-dir')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')

    pur_hist = np.zeros(101)               # pixel-weighted purity histogram
    n_atoms = 0
    tot_px = ceil_hit = 0
    pos = {k: np.zeros(NB, np.int64) for k in ('mean_conf', 'max_conf', 'size')}
    neg = {k: np.zeros(NB, np.int64) for k in ('mean_conf', 'max_conf', 'size')}
    npos = nneg = 0
    sizes = []

    print(f'{len(files)} tiles | τ={args.tau} | atoms={args.atoms}\n')
    for fi, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.uint8)
        conf = z['conf'].astype(np.float32)
        pred = z['pred'].astype(np.int16)
        cand = (gt > 0) & ((conf < args.tau) | (pred + 1 == BG))
        if not cand.any():
            continue

        if args.atoms == 'slic':
            from PIL import Image
            img = np.array(Image.open(Path(args.img_dir).expanduser() /
                                      f'{f.stem}.png').convert('RGB'))
            lab, n = atoms_slic(cand, img, args.n_segments)
        else:
            lab, n = atoms_cc(cand)
        if n == 0:
            continue

        size = np.bincount(lab.ravel(), minlength=n + 1)
        hist = np.bincount(lab.ravel().astype(np.int64) * NC + gt.ravel(),
                           minlength=(n + 1) * NC).reshape(n + 1, NC)
        hist[:, 0] = 0
        maj = hist[:, 1:].argmax(1) + 1
        majn = hist[:, 1:].max(1)

        csum = np.bincount(lab.ravel(), weights=conf.ravel(), minlength=n + 1)
        cmax = np.zeros(n + 1)
        np.maximum.at(cmax, lab.ravel(), conf.ravel())

        for c in range(1, n + 1):
            s = int(size[c])
            if s < args.min_size:
                continue
            n_atoms += 1
            sizes.append(s)
            p = majn[c] / max(s, 1)
            pur_hist[int(round(p * 100))] += s
            tot_px += s
            ceil_hit += int(majn[c])            # oracle labeller gets exactly these

            b_m = int(np.clip(csum[c] / s * NB, 0, NB - 1))
            b_x = int(np.clip(cmax[c] * NB, 0, NB - 1))
            b_s = int(np.clip(np.log10(max(s, 1)) / 6 * NB, 0, NB - 1))
            tgt = pos if maj[c] != BG else neg
            tgt['mean_conf'][b_m] += s
            tgt['max_conf'][b_x] += s
            tgt['size'][b_s] += s
            if maj[c] != BG:
                npos += s
            else:
                nneg += s

        if (fi + 1) % 250 == 0 or fi + 1 == len(files):
            print(f'  {fi + 1}/{len(files)}')

    w = pur_hist / max(pur_hist.sum(), 1)
    cum = np.cumsum(w)
    mean_pur = float((np.arange(101) / 100 * w).sum())
    sizes = np.array(sizes)

    md = [f'# Week 3 — atom quality (`{args.atoms}`)\n',
          f'- tiles: **{len(files)}**  |  τ: **{args.tau}**  |  '
          f'min atom: **{args.min_size}px**',
          f'- atoms: **{n_atoms:,}**, covering **{tot_px:,}** px',
          f'- median atom size: **{int(np.median(sizes)):,}** px, '
          f'p90 **{int(np.percentile(sizes, 90)):,}**, max **{int(sizes.max()):,}**\n',
          '## 1. Purity — can an atom be labelled at all?\n',
          'Share of an atom\'s pixels belonging to its own majority GT class, '
          'pixel-weighted. 1.00 = perfectly labelable, 0.50 = a coin flip no '
          'method can win.\n',
          '| purity | share of pixels at or below |', '|---|---|']
    for q in (50, 60, 70, 80, 90, 99):
        md.append(f'| ≤ {q / 100:.2f} | {100 * cum[q]:.1f}% |')
    md.append(f'\n**Mean purity (pixel-weighted): {mean_pur:.3f}**\n')

    md += ['## 2. Ceiling — a perfect labeller on these atoms\n',
           f'Give every atom its own majority GT class. That is the hard upper '
           f'bound for ANY method built on `{args.atoms}` atoms.\n',
           f'| | |', '|---|---|',
           f'| oracle-labeller accuracy | **{100 * ceil_hit / max(tot_px, 1):.1f}%** |',
           f'| pixels it would get right | {ceil_hit:,} of {tot_px:,} |\n']

    md += ['## 3. Detection at region level\n',
           'Pixel-level AUCs were all ≈0.5. Averaging over thousands of pixels '
           'kills independent noise, so a region aggregate can be far sharper — '
           'or stay flat, which would mean the signal is genuinely absent.\n',
           f'- atoms whose majority is a real class: **{npos:,}** px',
           f'- atoms whose majority is background: **{nneg:,}** px',
           f'- base rate: **{100 * npos / max(npos + nneg, 1):.1f}%**\n',
           '| region signal | AUC |', '|---|---|']
    best = 0.0
    for k in ('mean_conf', 'max_conf', 'size'):
        a = auc_from_hist(pos[k], neg[k])
        best = max(best, a if np.isfinite(a) else 0)
        md.append(f'| `{k}` | **{a:.3f}** |')

    md += ['\n## Verdict\n']
    if mean_pur < 0.75:
        md.append(f'⛔ **The atoms are the bug.** Mean purity {mean_pur:.3f} — a '
                  f'per-atom label is wrong for {100 * (1 - mean_pur):.0f}% of its '
                  'pixels before any scoring happens, and the oracle ceiling is '
                  f'{100 * ceil_hit / max(tot_px, 1):.1f}%. Every negative result so '
                  'far is consistent with this rather than with absent signal. '
                  '**Fix atomisation before concluding anything about the method**: '
                  'run `--atoms slic`, then SAM 3 mask proposals, and re-run the '
                  'ceiling and detection tests on those.')
    elif best >= 0.70:
        md.append(f'✅ **Region-level detection works** (AUC {best:.3f}) where '
                  'pixel-level did not. Aggregation was the missing step. Gate '
                  'recovery on this and re-run `selective_recovery_miou.py`.')
    else:
        md.append(f'⚠️ **Atoms are adequate (purity {mean_pur:.3f}) but detection '
                  f'still fails at region level (best AUC {best:.3f}).** That is the '
                  'cleaner negative: the signal is genuinely absent from SAM 3\'s '
                  'scalar outputs, not merely noisy. Appearance features are the only '
                  'remaining candidate.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
