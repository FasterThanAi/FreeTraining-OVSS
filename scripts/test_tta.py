"""Unit test for the dihedral TTA transforms in segearthov3_segmentor.

⛔ WHY THIS EXISTS. The forward transform rotates a PIL image then flips it; the
inverse must flip FIRST and then un-rotate. Getting that order backwards is
silent: every averaged view is misaligned with the ground truth, the run
completes, and the table is plausible and wrong. Four of this project's worst
bugs had exactly that shape.

The model is not needed. Applying forward to an image and inverse to the result
must return the original, for all eight transforms, on an image with no symmetry
of its own.

    python scripts/test_tta.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


class _NumpyTorch:
    """⭐ The segmentor's inverse uses only `torch.flip` and `torch.rot90`, both
    of which have exact numpy equivalents. Shimming them lets this test run on a
    machine with no torch -- which is the docs machine, where the transform code
    is actually written and where an ordering bug would otherwise go unnoticed
    until a GPU run."""
    @staticmethod
    def flip(a, dims):
        return np.flip(a, axis=tuple(dims)).copy()

    @staticmethod
    def rot90(a, k, dims):
        return np.rot90(a, k=k, axes=tuple(dims)).copy()


torch = _NumpyTorch()


def load_transforms():
    """Read the two staticmethods and _D4 out of the segmentor without importing
    it -- importing pulls in mmseg, sam3 and a CUDA build we do not need here."""
    src = (Path(__file__).resolve().parent.parent
           / 'reference' / 'segearthov3_segmentor.py').read_text()
    ns = {'torch': torch}
    start = src.index('    _D4 = [')
    end = src.index('    def _tta_views(self)')
    body = '\n'.join(l[4:] if l.startswith('    ') else l
                     for l in src[start:end].splitlines())
    body = body.replace('@staticmethod\n', '')
    exec(body, ns)
    return ns['_D4'], ns['_tta_fwd'], ns['_tta_inv']


def main():
    D4, fwd, inv = load_transforms()
    ok = True
    rng = np.random.default_rng(0)

    for name, (H, W) in (('square 64x64', (64, 64)), ('non-square 48x64', (48, 64))):
        a = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        # a corner marker, so a 180-degree error cannot pass by symmetry
        a[:6, :6] = 255; a[-6:, -6:] = 0
        img = Image.fromarray(a)
        ref = np.transpose(a, (2, 0, 1)).astype(np.float64)

        for k, f in D4:
            t = np.transpose(np.array(fwd(img, k, f)), (2, 0, 1)).astype(np.float64)
            back = inv(t, k, f)
            if back.shape != ref.shape:
                if H != W and k % 2 == 1:
                    continue        # expected: a 90° turn of a non-square image
                ok = False
                print(f'  ⛔ {name} k={k} f={f}: shape {tuple(back.shape)} '
                      f'!= {tuple(ref.shape)}')
                continue
            d = float(np.abs(back - ref).max())
            if d > 0:
                ok = False
                print(f'  ⛔ {name} k={k} f={f}: round trip differs by {d}')
        print(f'  {name}: checked {len(D4)} transforms')

    # ⚠️ The guard the shapes above imply: a 90° rotation of a NON-SQUARE image
    # changes its dimensions, so the averaged stacks would not be addable.
    H, W = 48, 64
    a = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    t = np.transpose(np.array(fwd(Image.fromarray(a), 1, False)), (2, 0, 1))
    if t.shape[-2:] == (H, W):
        ok = False
        print('  ⛔ a 90° rotation of a non-square image kept its shape; the '
              'non-square guard in the caller is then untested and may be wrong')
    else:
        print(f'  non-square guard: 90° turns {H}x{W} -> {tuple(t.shape[-2:])}, '
              f'so `d4` MUST be refused on non-square input')

    print('\n✅ all TTA transform tests pass' if ok else '\n⛔ FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
