"""
Does the segmentor's per-class threshold rule match the histogram arithmetic?

WHY. §9b's +1.18 mIoU was never computed by running the pipeline. It was
computed by `confusion_at`, which sums a (gt, pred, conf-bin) histogram. Before a
25-minute GPU run is spent, the two must be shown to be the same rule -- if they
are not, the GPU run will disagree and it will not be obvious which side is wrong.

This is CPU-only and takes seconds. It checks three things:

  1. the patched segmentor line is present and has the expected shape
  2. a scalar threshold reproduces the ORIGINAL published line exactly, on random
     data -- the validation gate (47.37) is preserved by construction
  3. a per-class vector reproduces `confusion_at` exactly, on random data

Point 3 is the one that matters. Point 2 is what stops a refactor from quietly
moving the baseline the whole project is measured against.

    python scripts/verify_perclass_tau.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_oracle import confusion_at, NBINS          # noqa: E402

SRC = Path(__file__).resolve().parents[1] / 'reference' / 'segearthov3_segmentor.py'


def segmentor_rule(pred, conf, taus, bg):
    """The rule as `predict()` applies it, in numpy.

        thd = prob_thd_vec[seg_pred]
        seg_pred[max_vals < thd] = bg_idx

    A scalar `taus` is the published line, unchanged.
    """
    out = pred.copy()
    thd = taus[pred] if np.ndim(taus) else taus
    out[conf < thd] = bg
    return out


def confusion(gt, pred, nc):
    C = np.zeros((nc, nc), np.int64)
    np.add.at(C, (gt, pred), 1)
    return C


def main():
    ok = True

    # ---- 1. the patch is where it should be
    if not SRC.exists():
        print(f'⛔ {SRC} not found'); return 1
    src = SRC.read_text()
    for frag in ('def set_prob_thd(self, prob_thd):',
                 'self.prob_thd_vec[seg_pred]',
                 'seg_pred[max_vals < thd] = self.bg_idx'):
        hit = frag in src
        ok &= hit
        print(f'  {"✅" if hit else "⛔"} segmentor contains: {frag}')
    if 'self.num_cls' not in src.split('def set_prob_thd')[1][:900]:
        print('  ⛔ set_prob_thd does not check the vector length against num_cls')
        ok = False
    else:
        print('  ✅ set_prob_thd checks the vector length against num_cls')

    # ---- 2 & 3. behavioural equivalence on random data
    rng = np.random.default_rng(0)
    nc, bg, N = 7, 0, 400_000
    grid = np.arange(NBINS + 1) / NBINS

    for trial in range(20):
        gt = rng.integers(0, nc, N)
        pred = rng.integers(0, nc, N)
        # cluster confidences near the thresholds, where an off-by-one bin lives
        conf = np.clip(rng.normal(0.5, 0.25, N), 0, 1).astype(np.float32)
        b = np.clip((conf * NBINS).astype(np.int64), 0, NBINS - 1)
        H = np.zeros((nc, nc, NBINS), np.int64)
        np.add.at(H, (gt, pred, b), 1)

        taus = rng.choice(grid, nc) if trial else np.full(nc, 0.5)
        taus[bg] = 0.5                       # no effect either way; pin it

        # the histogram bins conf, so the direct rule must compare the BIN, not
        # the raw float -- otherwise this tests float16 rounding, not the rule
        got = confusion(gt, segmentor_rule(pred, b / NBINS, taus, bg), nc)
        want = confusion_at(H, taus, bg, NBINS)
        if not np.array_equal(got, want):
            print(f'  ⛔ trial {trial}: segmentor rule != confusion_at  '
                  f'(max diff {np.abs(got - want).max():,})')
            ok = False

    print(f'  {"✅" if ok else "⛔"} per-class rule == confusion_at over 20 random '
          f'threshold vectors, {N:,} px each')

    # scalar path is bit-identical to the published line
    for t in (0.5, 0.3, 0.1):
        gt = rng.integers(0, nc, N)
        pred = rng.integers(0, nc, N)
        conf = rng.random(N).astype(np.float32)
        published = pred.copy(); published[conf < t] = bg
        vec = segmentor_rule(pred, conf, np.full(nc, t), bg)
        same = np.array_equal(published, vec)
        ok &= same
        print(f'  {"✅" if same else "⛔"} τ={t}: per-class vector of a constant '
              f'== the published scalar line')

    print('\n' + ('✅ the histogram result and the pipeline rule are the same rule. '
                  'A GPU run should reproduce §9b to within float16 cache noise '
                  '(~0.01-0.02 mIoU).' if ok else
                  '⛔ MISMATCH — do not run the GPU eval until this passes; a '
                  'disagreement there would be ambiguous.'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
