"""--group-re: are folds really group-disjoint, and is the flag a no-op when absent?

UAVid val is 7 flight sequences of 10 consecutive frames. A frame-level 5-fold puts
near-duplicate frames of the same flight in both calibration and evaluation, so the
fit can memorise the scene it is scored on and the gain is inflated. This checks the
grouping arithmetic that prevents it, and -- equally important -- that omitting the
flag leaves every recorded LoveDA/Potsdam/OEM number untouched.
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent / 'tau_cv.py'


def assign(stems, folds, pattern, seed=0):
    """The grouping path from tau_cv.py, lifted."""
    rng = np.random.default_rng(seed)
    pat = re.compile(pattern)
    keys = []
    for st in stems:
        m = pat.search(st)
        if m is None:
            raise ValueError(st)
        keys.append(m.group(1) if m.groups() else m.group(0))
    uniq = sorted(set(keys))
    pos = {g: i for i, g in enumerate(uniq)}
    gid = np.array([pos[k] for k in keys])
    gperm = rng.permutation(gid.max() + 1)
    gof = np.empty(gid.max() + 1, dtype=int)
    for k, part in enumerate(np.array_split(gperm, folds)):
        gof[part] = k
    owner = gof[gid]
    return gid, [np.where(owner == k)[0] for k in range(folds)], uniq


def main():
    fail = 0

    def check(ok, msg):
        nonlocal fail
        fail += not ok
        print(f'{"ok  " if ok else "FAIL"}  {msg}')

    # --- the no-op guarantee: absent flag must not touch the old code path ---
    src = SRC.read_text()
    check("ap.add_argument('--group-re', default=None," in src,
          '--group-re defaults to None')
    check('    if gid is None:\n        order = rng.permutation(n)\n'
          '        folds = np.array_split(order, args.folds)' in src,
          'ungrouped path is the original permutation split, verbatim')
    check(src.index('gid = None') < src.index('if args.group_re:'),
          'gid is initialised before the grouping branch')
    # the grouping block must consume no randomness when the flag is absent,
    # or every seeded result in the project would shift
    pre = src[src.index('gid = None'):src.index('# ---------------- k-fold')]
    check('rng.' not in pre, 'grouping block draws no randomness when unused')

    # --- UAVid's real shape: 7 sequences x 10 frames ---
    stems = [f'seq{s}_{f:06d}' for s in range(16, 23) for f in range(0, 1000, 100)]
    check(len(stems) == 70, f'70 synthetic UAVid stems ({len(stems)})')
    gid, folds, uniq = assign(stems, 5, r'^(seq[0-9]+)')
    check(len(uniq) == 7, f'7 groups recovered ({len(uniq)})')
    check(sum(len(f) for f in folds) == 70, 'every tile assigned exactly once')
    check(len({i for f in folds for i in f}) == 70, 'no tile in two folds')
    check(all(len(f) > 0 for f in folds), f'no empty fold {[len(f) for f in folds]}')

    # THE property: a group never straddles a fold
    straddle = 0
    for g in range(gid.max() + 1):
        owners = {k for k, f in enumerate(folds) if np.any(gid[f] == g)}
        straddle += len(owners) != 1
    check(straddle == 0, f'no group straddles a fold ({straddle} straddling)')

    # and the ungrouped split DOES straddle -- otherwise the test proves nothing
    rng = np.random.default_rng(0)
    ufolds = np.array_split(rng.permutation(70), 5)
    ustraddle = sum(len({k for k, f in enumerate(ufolds) if np.any(gid[f] == g)}) != 1
                    for g in range(7))
    check(ustraddle == 7, f'frame-level split straddles all 7 groups ({ustraddle}) '
                          f'-- this is the leak being fixed')

    # --- LoveDA-style stems must be rejected loudly, not silently one-group ---
    try:
        assign(['2522', '2523', '2524'], 2, r'^(seq[0-9]+)')
        check(False, 'a non-matching stem raises')
    except ValueError:
        check(True, 'a non-matching stem raises instead of collapsing to one group')

    # --- fewer groups than folds must be refused in the script ---
    check('is fewer than {args.folds} ' in src or 'fewer than' in src,
          'script refuses folds > groups')

    # --- the regex must cover EVERY real name form in both UAVid splits ---
    # val : 000000..000900 (seq16) + fileNN-k          -> 7 sequences
    # train: 000000..000900 (seq1) + fileNN-k + file-k -> 20 sequences
    RE = r'([a-z]+[0-9]*)(?=[-_][0-9]+$)'
    want = {'seq16_000000': 'seq16', 'seq16_file20-1': 'file20',
            'seq16_file29-10': 'file29', 'seq1_000900': 'seq1',
            'seq1_file5-4': 'file5', 'seq1_file18-10': 'file18',
            'seq1_file-1': 'file'}
    for stem, exp in want.items():
        m = re.compile(RE).search(stem)
        got = (m.group(1) if m else None)
        check(got == exp, f'{stem:<18} -> {got} (want {exp})')

    # the augmented copies must NOT silently become their own scenes
    for stem in ['seq1_flipped18', 'seq1_shifted144', 'seq1_shifted9']:
        m = re.compile(RE).search(stem)
        check(m is None, f'{stem:<18} does not match -- it must be EXCLUDED, '
                         f'not grouped')

    # --- exclude-re exists in all three readers and drops exactly the 400 ---
    import pathlib
    for f in ['tau_cv.py', 'tau_oracle.py', 'metric_report.py']:
        t = (pathlib.Path(__file__).resolve().parent / f).read_text()
        check("ap.add_argument('--exclude-re', default=None," in t,
              f'{f} accepts --exclude-re')
        check('removed every ' in t, f'{f} refuses an exclusion that drops everything')
    names = ([f'seq1_{i:06d}' for i in range(0, 1000, 100)] +
             [f'seq1_file{s}-{k}' for s in range(2, 20) for k in range(1, 11)] +
             [f'seq1_file-{k}' for k in range(1, 11)] +
             [f'seq1_flipped{i}' for i in range(200)] +
             [f'seq1_shifted{i}' for i in range(200)])
    check(len(names) == 600, f'600 synthetic train stems ({len(names)})')
    kept = [n for n in names if not re.search(r'_(flipped|shifted)', n)]
    check(len(kept) == 200, f'exclusion leaves 200 real frames ({len(kept)})')
    groups = {re.compile(RE).search(n).group(1) for n in kept}
    check(len(groups) == 20, f'200 real frames carry 20 sequences ({len(groups)})')

    # --- lever 2 (argmax_reorder) must have the SAME protections -----------
    ar = (pathlib.Path(__file__).resolve().parent / 'argmax_reorder.py').read_text()
    check("ap.add_argument('--group-re', default=None," in ar,
          'argmax_reorder accepts --group-re')
    check("ap.add_argument('--exclude-re', default=None," in ar,
          'argmax_reorder accepts --exclude-re')
    check('    if _gid is None:\n        folds = np.array_split(order, args.folds)' in ar,
          'argmax_reorder ungrouped path is the original split, verbatim')
    # the flag must not perturb the seeded partition of every recorded run
    pre = ar[ar.index('_frng = np.random.default_rng(args.seed)'):ar.index('rng = np.random.default_rng(args.seed + 1000)')]
    body = pre[pre.index('_gid = None'):]
    check('_frng.' not in body and 'rng.' not in body,
          'argmax_reorder grouping block draws no randomness when unused '
          '(recorded LoveDA/Potsdam/OEM partitions are unchanged)')
    check('a group-disjoint fold came out empty' in ar,
          'argmax_reorder refuses an empty fold')
    mdr = (pathlib.Path(__file__).resolve().parent / 'measure_discard_rate.py').read_text()
    check("ap.add_argument('--exclude-re', default=None," in mdr,
          'measure_discard_rate accepts --exclude-re (saves 3x disk AND 3x GPU)')
    check(mdr.index("names = sorted(p.stem") < mdr.index("if args.exclude_re:")
          < mdr.index('if args.limit:'),
          'measure_discard_rate excludes BEFORE --limit/--sample and the disk probe')

    print()
    print('ALL PASS' if not fail else f'{fail} FAILURE(S)')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
