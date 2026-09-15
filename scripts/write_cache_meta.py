"""Write the `_meta.json` sidecar for a cache that predates it.

⛔ THE BUG THIS REPAIRS, AND IT IS WORTH READING. `bg_idx` -- where the
segmentor sends sub-τ pixels -- is not recoverable from the cached arrays.
Every analysis script therefore located the catch-all BY NAME, which works on
every dataset that has one. DLRSD has none, `labels.py` guessed the first class,
and both the oracle sweep and the 5-fold ran to completion with `airplane` as
the discard target: every sub-threshold pixel in the dataset counted as a
predicted airplane. The tables were complete, plausible and void.

⭐ The tell was printed and scrolled past: `tau_oracle.py`'s cross-check row
read 34.63 where `measure_discard_rate.py` had reported 37.89, and that
section says in as many words that the two must agree.

`measure_discard_rate.py` now writes this file itself, so only caches built
before that need this script -- and it costs no GPU, because nothing about the
cached arrays was wrong.

    python scripts/write_cache_meta.py --cache ~/outputs/dlrsd_full/cache --bg-idx 17
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bg-idx', type=int, required=True,
                    help="the segmentor's `bg_idx`, 0-indexed, exactly as the "
                         "config sets it. ⚠️ Read it off the config that "
                         "PRODUCED this cache -- a different value here "
                         "silently re-points the discard target.")
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    cache = Path(args.cache).expanduser()
    files = sorted(cache.glob('*.npz'))
    if not files:
        raise SystemExit(f'no .npz under {cache}')
    z = np.load(files[0], allow_pickle=True)
    if 'classes' not in z.files:
        raise SystemExit(f'{files[0].name} has no `classes` key; re-run '
                         f'measure_discard_rate.py')
    classes = [str(x) for x in z['classes']]
    n = len(classes)
    print(f'  {len(files)} tiles, {n} classes: {", ".join(classes)}')

    if not 0 <= args.bg_idx <= n:
        raise SystemExit(f'⛔ bg_idx={args.bg_idx} is neither a class index '
                         f'(0..{n - 1}) nor the sink index ({n}).')

    named = [c for c in classes if c.lower() in labels.CATCH_ALL_ALIASES]
    if args.bg_idx == n:
        if named:
            raise SystemExit(
                f'⛔ bg_idx={args.bg_idx} says "unscored sink", but this cache '
                f'HAS a catch-all class ({named}). One of the two is wrong, and '
                f'guessing which would silently re-point every discard number.')
        print(f'  ⭐ bg_idx={args.bg_idx} == the class count, so the discard '
              f'target is an UNSCORED SINK outside the {n} classes. A discarded '
              f'pixel is a false negative for its true class and a false '
              f'positive for nothing.')
    else:
        print(f'  discard target is `{classes[args.bg_idx]}` (class index '
              f'{args.bg_idx})')
        if not named:
            print(f'  ⚠️ and no class name looks like a catch-all, so this is '
                  f'pointing sub-τ pixels at what appears to be a REAL class. '
                  f'Confirm that is what the config does.')

    out = cache / labels.CACHE_META
    if out.exists() and not args.force:
        print(f'\n  {out} already exists:\n    {out.read_text().strip()}')
        raise SystemExit('  pass --force to overwrite')
    meta = {'bg_idx': int(args.bg_idx), 'classes': classes,
            'sink': args.bg_idx >= n}
    if args.tau is not None:
        meta['tau'] = float(args.tau)
    out.write_text(json.dumps(meta, indent=2))
    print(f'\n✅ written: {out}')

    # prove it round-trips through the path the analysis scripts actually use
    LB = labels.from_cache(cache)
    print(f'  reads back as: {LB}')
    print(f'  n_pred = {LB.n_pred}  (confusion matrix is {LB.n}x{LB.n_pred})')


if __name__ == '__main__':
    main()
