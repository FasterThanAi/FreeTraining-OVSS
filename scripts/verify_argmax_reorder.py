"""
Does the segmentor's SCALED-argmax rule match the histogram arithmetic?

WHY. `argmax_reorder.py` reports +1.16 mIoU over per-class thresholds, and that
number was never produced by running the pipeline -- it comes from `hist_at`,
which applies the scale in numpy and bins the result. Before a 25-minute GPU run
is spent, the two must be shown to be the same rule. If they are not, the GPU run
will disagree and it will not be obvious which side is wrong.

This is the same discipline as verify_perclass_tau.py, and it exists because that
test found a real bug (the truncating bin edge) which nothing else would have
caught until a wrong threshold was deployed.

CPU-only, seconds. Four checks:

  1. the patch is present in reference/segearthov3_segmentor.py and has the
     expected shape
  2. ⭐ `class_scale=None` leaves the ORIGINAL argmax line untouched -- byte for
     byte, on random data. This is the 47.38 gate, preserved by construction.
  3. a scale vector reproduces `argmax_reorder.hist_at` exactly
  4. the whole chain -- scale, argmax, per-class threshold -- reproduces
     `confusion_at` on the histogram `hist_at` builds

⚠️ Check 2 is the one that protects the project. Everything measured in two
months rests on the baseline being 47.38, and a refactor that quietly moved the
argmax would invalidate all of it while still producing a plausible number.

    python scripts/verify_argmax_reorder.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_oracle import confusion_at, NBINS              # noqa: E402
from argmax_reorder import hist_at                      # noqa: E402

SRC = Path(__file__).resolve().parents[1] / 'reference' / 'segearthov3_segmentor.py'


def segmentor_rule(S, taus, w, bg):
    """`predict()` transcribed to numpy, including the branch.

        if class_scale is None:
            seg_pred = argmax(seg_logits);  max_vals = seg_logits.max(0)
        else:
            seg_pred = argmax(seg_logits * scale)
            max_vals = seg_logits.gather(0, seg_pred)
        thd = prob_thd_vec[seg_pred]
        seg_pred[max_vals < thd] = bg_idx

    S is (n_px, N) here rather than (N, H, W); the axis is the only difference.
    """
    if w is None:
        pred = np.argmax(S, axis=1)
        conf = S.max(axis=1)
    else:
        pred = np.argmax(S * w, axis=1)
        conf = S[np.arange(len(pred)), pred]
    out = pred.copy()
    thd = taus[pred] if np.ndim(taus) else taus
    out[conf < thd] = bg
    return out


def confusion(gt0, pred, nc):
    C = np.zeros((nc, nc), np.int64)
    np.add.at(C, (gt0, pred), 1)
    return C


def main():
    ok = True
    rng = np.random.default_rng(0)
    nc, bg, n = 7, 0, 200_000

    # ---------------- 1. the patch is where it should be
    if not SRC.exists():
        print(f'⛔ {SRC} not found')
        return 1
    src = SRC.read_text()
    need = ['def set_class_scale(self, class_scale):',
            'if self.class_scale is None:',
            'seg_logits * self.class_scale.view(-1, 1, 1)',
            'seg_logits.gather(0, seg_pred.unsqueeze(0)).squeeze(0)']
    for frag in need:
        hit = frag in src
        ok &= hit
        print(f'{"OK  " if hit else "FAIL"}  patch present: {frag[:56]}')

    # ⚠️ the unscaled branch must still contain the ORIGINAL two lines verbatim
    orig = ['seg_pred = torch.argmax(seg_logits, dim=0)',
            'max_vals = seg_logits.max(0)[0]']
    for frag in orig:
        hit = frag in src
        ok &= hit
        print(f'{"OK  " if hit else "FAIL"}  original line intact: {frag[:56]}')
    print()

    # ---------------- 2. class_scale=None must equal the published behaviour
    S = rng.random((n, nc)).astype(np.float32)
    gt0 = rng.integers(0, nc, n)
    for tau in (0.5, 0.1, 0.8):
        a = segmentor_rule(S, tau, None, bg)                 # the branch taken
        b = np.argmax(S, axis=1)                             # published, by hand
        b[S.max(axis=1) < tau] = bg
        same = np.array_equal(a, b)
        ok &= same
        print(f'{"OK  " if same else "FAIL"}  class_scale=None == published rule '
              f'at tau={tau}')
    # and a scale of all ones must agree with it too, up to argmax ties
    ones = np.ones(nc, np.float32)
    a = segmentor_rule(S, 0.5, None, bg)
    b = segmentor_rule(S, 0.5, ones, bg)
    same = np.array_equal(a, b)
    ok &= same
    print(f'{"OK  " if same else "FAIL"}  class_scale=1 == class_scale=None')
    print()

    # ---------------- 3 & 4. the scaled rule vs the histogram arithmetic
    for trial in range(5):
        w = np.exp(rng.uniform(np.log(0.4), np.log(2.5), nc)).astype(np.float32)
        taus = np.round(rng.uniform(0.05, 0.95, nc) * NBINS) / NBINS
        taus[bg] = 0.0                                  # tau[bg] is never applied
        gt_mask = gt0 + 1                               # cache convention: 1-indexed

        H = hist_at(S, gt_mask, w, nc, NBINS)
        C_hist = confusion_at(H, taus, bg, NBINS)

        pred = segmentor_rule(S, taus, w, bg)
        C_seg = confusion(gt0, pred, nc)

        same = np.array_equal(C_hist, C_seg)
        ok &= same
        if same:
            print(f'OK    trial {trial + 1}: histogram == segmentor '
                  f'({C_seg.sum():,} px, {int((pred == bg).sum()):,} to bg)')
        else:
            d = np.abs(C_hist - C_seg)
            print(f'FAIL  trial {trial + 1}: {int(d.sum()):,} px differ, '
                  f'largest cell {int(d.max()):,}')
            print(f'      w    = {np.round(w, 3).tolist()}')
            print(f'      taus = {np.round(taus, 3).tolist()}')

    # ---------------- a scale that must CHANGE something, or the test is vacuous
    w = np.ones(nc, np.float32)
    w[3] = 2.5
    moved = int((segmentor_rule(S, 0.5, None, bg)
                 != segmentor_rule(S, 0.5, w, bg)).sum())
    nontrivial = moved > n // 100
    ok &= nontrivial
    print(f'{"OK  " if nontrivial else "FAIL"}  a 2.5x scale moves {moved:,}/{n:,} '
          f'pixels — the scaled path is actually exercised')

    print('\n' + ('=' * 62))
    if ok:
        print('ALL PASS — the histogram arithmetic and the segmentor apply the')
        print('same rule, and class_scale=None leaves the published path alone.')
        print('The GPU run is now a verification, not a discovery.')
    else:
        print('FAILURES ABOVE — do not run the GPU job until these are resolved.')
    print('=' * 62)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
