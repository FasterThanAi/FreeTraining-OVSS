"""--cache-stride: is every cached array subsampled, and is the gate untouched?

UAVid frames are 3840x2160, so a full-resolution cache costs 43 MB/frame and 300
train frames do not fit on this disk. But `max_side` means SAM 3 never saw more than
1008x1008 -- the full-resolution cache is storing an UPSAMPLE of the model's own
output. A stride throws away that padding.

Two things must hold, and neither is obvious from reading the diff:
  1. EVERY per-pixel array in the .npz is strided, or arrays of different shapes end
     up in one file and every downstream script breaks on the first read.
  2. The confusion matrix, mIoU and every discard figure stay FULL resolution, so the
     validation gate still means what it meant.
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent / 'measure_discard_rate.py'


def main():
    src = SRC.read_text()
    fail = 0

    def check(ok, msg):
        nonlocal fail
        fail += not ok
        print(f'{"ok  " if ok else "FAIL"}  {msg}')

    # ---- 1. every per-pixel array in the savez call is wrapped ---------
    i = src.index('np.savez_compressed(')
    body = src[i:src.index('head_stacks', i) + 40]
    # named scalars/1-D are exempt: spres is (n_views, N), classes is a name list
    for key in ('conf', 'pred', 'conf2', 'pred2', 'gt'):
        m = re.search(rf'\n\s+{key}=(\S)', body)
        check(bool(m) and body[m.start(1):m.start(1) + 5] == '_sub(',
              f'{key}= is wrapped in _sub()')
    for blob, what in ((r'fused_arrays\.items\(\)', 'fused_arrays'),
                       (r'head_stacks\.items\(\)', 'head_stacks')):
        m = re.search(rf'\{{[^}}]*_sub\([^}}]*{blob}', body)
        check(bool(m), f'{what} values are wrapped in _sub()')
    check("'logits': _sub(" in body, "the --cache-full logits stack is wrapped")
    for key in ('spres=', 'classes='):
        m = re.search(rf'\n\s+{key}', body)
        check(bool(m) and '_sub' not in body[m.start():m.start() + 60],
              f'{key} correctly NOT strided (not a per-pixel array)')

    # ---- 2. the gate path must NOT be strided ---------------------------
    j = src.index('_cs = args.cache_stride')
    check('_sub(' not in src[:j],
          'no array is strided before the writer block -- the confusion matrix, '
          'mIoU and every discard figure stay full resolution')
    check("_sub = (lambda a: a[..., ::_cs, ::_cs]) if _cs > 1 else (lambda a: a)" in src,
          'stride 1 makes _sub the identity')
    check("ap.add_argument('--cache-stride', type=int, default=1," in src,
          '--cache-stride defaults to 1, so every existing cache command is unchanged')
    check('if args.cache_stride < 1:' in src, 'a stride below 1 is refused')

    # ---- 3. the disk estimate must use the STRIDED geometry -------------
    check('need_gb = _per * _cw * _ch * len(names)' in src,
          'disk estimate uses the strided width/height')
    check('_cw, _ch = -(-_w // _k), -(-_h // _k)' in src,
          'ceil division, matching numpy slice length')

    # ---- 4. ceil arithmetic agrees with numpy, on the real shapes -------
    for (w, h) in [(3840, 2160), (4096, 2160), (1024, 1024), (1000, 1000)]:
        for k in (1, 2, 3, 4, 8):
            a = np.zeros((h, w), np.uint8)
            got = a[::k, ::k].shape
            want = (-(-h // k), -(-w // k))
            ok = got == want
            fail += not ok
            if not ok:
                print(f'FAIL  {w}x{h} stride {k}: numpy {got} vs formula {want}')
    check(True, 'ceil formula matches numpy on 3840x2160, 4096x2160, 1024x1024, 1000x1000')

    # ---- 5. a stride is an UNBIASED subsample of the joint distribution --
    # this is the statistical claim the fit rests on: thresholds are chosen from
    # a (gt, conf) histogram, so the subsample must not shift the proportions.
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 7, size=(2160, 3840), dtype=np.uint8)
    conf = rng.random((2160, 3840)).astype(np.float32)
    for k in (2, 4, 8):
        full = np.bincount(gt.ravel(), minlength=7) / gt.size
        sub = gt[::k, ::k]
        part = np.bincount(sub.ravel(), minlength=7) / sub.size
        drift = float(np.abs(full - part).max())
        mean_drift = abs(conf.mean() - conf[::k, ::k].mean())
        ok = drift < 0.005 and mean_drift < 0.005
        fail += not ok
        print(f'{"ok  " if ok else "FAIL"}  stride {k}: class-share drift '
              f'{drift:.4f}, mean-conf drift {mean_drift:.5f}, '
              f'{sub.size:,} px kept ({100*sub.size/gt.size:.1f}%)')

    # ---- 6. and it is NOT unbiased against a periodic pattern -----------
    # stated as a known limit rather than discovered later: a stride aliases
    # anything with the same period. Real imagery has no 4-pixel periodicity in
    # its LABELS, but say so rather than claim a stride is free.
    stripe = (np.arange(2160)[:, None] % 4 == 0).astype(np.uint8)
    aliased = abs(stripe.mean() - stripe[::4, ::4].mean())
    check(aliased > 0.5, f'a 4-periodic pattern DOES alias under stride 4 '
                         f'(drift {aliased:.2f}) -- known limit, not a surprise')

    print()
    print('ALL PASS' if not fail else f'{fail} FAILURE(S)')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
