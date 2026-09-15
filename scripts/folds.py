"""Fold assignment: group-disjoint, stratified, or plain.

TWO OPPOSITE CORRECTIONS, AND PICKING THE WRONG ONE IS ITS OWN BUG.

  --group-re     tiles sharing a key go in the SAME fold.
                 For CORRELATED tiles. UAVid val is 7 flights of 10 consecutive
                 frames; a frame-level split put near-duplicates of the
                 evaluation scene in the calibration set and reported
                 +1.05 ± 0.86 (5/5) where the truth was +0.51 ± 0.95 (3/5) --
                 the leak was worth +0.54 mIoU and two folds.

  --stratify-re  tiles sharing a key are SPREAD ACROSS folds, evenly.
                 For datasets whose groups carry DIFFERENT CLASSES.

⭐ WHY THE SECOND EXISTS, measured rather than assumed. DLRSD is 21 UC Merced
scene categories of 100 images, so `--group-re` looks like the obvious choice.
`scripts/dlrsd_class_map.py` shows it is the wrong one: `airplane`, `dock` and
`tanks` each occur in exactly ONE category, and `sea` and `mobile home` in two.
Hold a category out and those classes have no calibration pixels at all -- the
fit is asked to set a threshold it cannot see, and the resulting failure reads
as the method's rather than the protocol's. Group-disjoint folds there are not
a decorrelation control, they are a cross-category transfer test.

⚠️ SO THE CHOICE IS A PROPERTY OF THE DATA, NOT A HOUSE STYLE. Group when the
groups are near-duplicates of each other; stratify when they are different
subjects that each class does not span. Run `dlrsd_class_map.py`-style counts
before deciding, and report both arms where it is affordable.

⛔ NEITHER FLAG MAY PERTURB A RUN THAT DOES NOT USE IT. Everything here is
called only from the new branches; the plain and grouped paths in `tau_cv.py`
and `argmax_reorder.py` are untouched and consume randomness exactly as before,
so every recorded LoveDA / Potsdam / UAVid / ConInfer number stands.
Asserted by `scripts/test_stratified_folds.py`.
"""
import re

import numpy as np


def parse_keys(files, pattern, flag):
    """Regex -> integer stratum/group id per file. Draws NO randomness."""
    pat = re.compile(pattern)
    keys = []
    for f in files:
        m = pat.search(f.stem)
        if m is None:
            raise SystemExit(f'{flag} {pattern!r} matched nothing on '
                             f'{f.stem!r} -- every tile must have a key')
        keys.append(m.group(1) if m.groups() else m.group(0))
    uniq = sorted(set(keys))
    pos = {g: i for i, g in enumerate(uniq)}
    return np.array([pos[k] for k in keys]), uniq


def stratified(sid, k, rng):
    """Deal each stratum's tiles round-robin across k folds.

    Every fold then contains roughly the same number of tiles from EVERY
    stratum, so no class can be absent from a fold's calibration set because
    its category was held out.

    ⭐ The per-stratum random offset matters when a stratum does not divide
    evenly by k: without it every remainder lands on fold 0 and that fold grows
    systematically. DLRSD's 100 per category over 5 folds divides exactly, so
    it changes nothing there -- it is for the next dataset, not this one.
    """
    n = len(sid)
    owner = np.empty(n, dtype=int)
    for s in range(int(sid.max()) + 1):
        idx = np.where(sid == s)[0]
        if idx.size == 0:
            continue
        idx = idx[rng.permutation(idx.size)]
        off = int(rng.integers(k))
        for j, t in enumerate(idx):
            owner[t] = (j + off) % k
    folds = [np.where(owner == f)[0] for f in range(k)]
    if any(f.size == 0 for f in folds):
        raise SystemExit(f'a stratified fold came out empty: sizes '
                         f'{[f.size for f in folds]}. Use fewer folds.')
    return folds


def stratified_draw(sid, size, n, rng):
    """Draw `size` tiles for a learning curve, proportionally per stratum.

    ⚠️ A plain random draw of 25 tiles from 2100 would miss whole categories,
    and on DLRSD a missed category means a class with zero calibration pixels --
    the same failure stratified folds exist to prevent, one level down.
    """
    take = []
    for s in range(int(sid.max()) + 1):
        idx = np.where(sid == s)[0]
        if idx.size == 0:
            continue
        m = int(round(size * idx.size / n))
        m = min(idx.size, max(1, m))          # at least one, never more than it has
        take.append(idx[rng.permutation(idx.size)][:m])
    tr = np.concatenate(take) if take else np.array([], int)
    te = np.setdiff1d(np.arange(n), tr)
    return tr, te


def describe(uniq, sid, n, k):
    sizes = np.bincount(sid, minlength=len(uniq))
    return (f'  stratifying: {len(uniq)} strata from {n} tiles '
            f'(sizes {sizes.min()}-{sizes.max()}); every fold gets a share of '
            f'each, so no class is missing from a calibration set')
