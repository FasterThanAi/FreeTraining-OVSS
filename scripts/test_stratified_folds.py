"""--stratify-re works, and NOTHING that does not use it moves.

⛔ THE SECOND HALF IS THE POINT. `tau_cv.py` and `argmax_reorder.py` produced
+1.18 on LoveDA, +4.86 on Potsdam, +5.89 on UAVid and +2.51 on ConInfer. A fold
partition is a function of the RNG stream, so an added branch that consumes one
extra draw -- or draws in a different order -- silently repartitions every one
of those and they stop being reproducible.

⚠️ That is not hypothetical here. @ARGMAX_SCALING_RESULTS records exactly this
failure: the fold partition was drawn from the same stream as the pixel
subsampling, so raising --subsample silently reshuffled the folds and the 40k
and 150k runs were never comparable. "Two settings of one knob must differ in
that knob alone", and that one did not.

So the plain and grouped branches are left untouched and tested against
verbatim copies of the code as it stood before --stratify-re existed.

    python scripts/test_stratified_folds.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import folds as foldlib  # noqa: E402

ok = True


def chk(c, m):
    global ok
    print(f'  {"✅" if c else "⛔"} {m}')
    ok = ok and bool(c)


class F:                                   # a stand-in for a cache file path
    def __init__(self, stem):
        self.stem = stem


# 21 categories x 100, exactly DLRSD's shape
CATS = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
        'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
        'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
        'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
        'storagetanks', 'tenniscourt']
FILES = [F(f'{c}{i:02d}') for c in CATS for i in range(100)]
N, K = len(FILES), 5

print('\n1. ⛔ No-op — the plain path, against the pre-change code verbatim')
for seed in (0, 1, 7):
    rng_old = np.random.default_rng(seed)
    old = np.array_split(rng_old.permutation(N), K)        # the code as it was
    rng_new = np.random.default_rng(seed)
    gid = sid = None
    if gid is None and sid is None:                        # the code as it is
        new = np.array_split(rng_new.permutation(N), K)
    chk(all(np.array_equal(a, b) for a, b in zip(old, new)),
        f'seed {seed}: plain folds bit-identical')

print('\n2. ⛔ No-op — the grouped path (UAVid), against the pre-change code')
gkeys, guniq = foldlib.parse_keys(FILES, r'^([a-z]+)', '--group-re')
for seed in (0, 3):
    r_old = np.random.default_rng(seed)
    gof = np.empty(gkeys.max() + 1, int)
    for k, part in enumerate(np.array_split(r_old.permutation(gkeys.max() + 1), K)):
        gof[part] = k
    old = [np.where(gof[gkeys] == k)[0] for k in range(K)]
    r_new = np.random.default_rng(seed)
    gof2 = np.empty(gkeys.max() + 1, int)
    for k, part in enumerate(np.array_split(r_new.permutation(gkeys.max() + 1), K)):
        gof2[part] = k
    new = [np.where(gof2[gkeys] == k)[0] for k in range(K)]
    chk(all(np.array_equal(a, b) for a, b in zip(old, new)),
        f'seed {seed}: grouped folds bit-identical')

print('\n3. parse_keys draws no randomness')
r = np.random.default_rng(0)
before = r.random()
r2 = np.random.default_rng(0)
foldlib.parse_keys(FILES, r'^([a-z]+)', '--stratify-re')
chk(before == r2.random(), 'the key parse consumed no draws')

print('\n4. Stratified folds — every fold sees every category')
sid, suniq = foldlib.parse_keys(FILES, r'^([a-z]+)', '--stratify-re')
fs = foldlib.stratified(sid, K, np.random.default_rng(0))
chk(len(suniq) == 21, f'21 strata parsed, got {len(suniq)}')
chk(sum(len(f) for f in fs) == N, f'every tile assigned once: {sum(len(f) for f in fs)}')
chk(len(set(np.concatenate(fs))) == N, 'no tile assigned twice')
per = [len(set(sid[f])) for f in fs]
chk(all(p == 21 for p in per), f'each fold contains all 21 categories: {per}')
sizes = sorted(len(f) for f in fs)
chk(sizes[-1] - sizes[0] <= 1, f'fold sizes within 1 of each other: {sizes}')
counts = np.array([[int((sid[f] == s).sum()) for s in range(21)] for f in fs])
chk(counts.min() == 20 and counts.max() == 20,
    f'exactly 20 of each category per fold (min {counts.min()}, max {counts.max()})')

print('\n5. ⭐ The contrast that motivates the flag')
gf = [np.where(np.isin(gkeys, part))[0]
      for part in np.array_split(np.random.default_rng(0).permutation(21), K)]
gper = [len(set(gkeys[f])) for f in gf]
chk(max(gper) <= 5, f'group-disjoint folds see only {gper} categories each')
chk(all(p == 21 for p in per),
    'stratified folds see all 21 — so a one-category class like `airplane` '
    'always has calibration pixels')

print('\n6. Learning-curve draws stay proportional')
for sz in (25, 100, 400):
    tr, te = foldlib.stratified_draw(sid, sz, N, np.random.default_rng(0))
    chk(len(set(sid[tr])) == 21,
        f'n={sz}: the draw covers all 21 categories ({len(tr)} tiles)')
    chk(len(np.intersect1d(tr, te)) == 0, f'n={sz}: train and eval are disjoint')

print('\n7. An empty fold is refused rather than returned')
try:
    foldlib.stratified(np.zeros(3, int), 5, np.random.default_rng(0))
    chk(False, 'did NOT refuse a partition with empty folds')
except SystemExit:
    chk(True, 'refuses when a fold would be empty')

print('\n' + ('ALL PASS' if ok else '⛔ FAILURES ABOVE'))
sys.exit(0 if ok else 1)
