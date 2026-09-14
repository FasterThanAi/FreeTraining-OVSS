"""max_side: does the downscale fire only where it should, and preserve shape?

The claim this guards is narrow and load-bearing: `max_side` exists so UAVid's
3840x2160 frames stop OOMing a 16 GB card, and it MUST be a strict no-op on every
dataset already measured. `reference/segearthov3_segmentor.py` is the code behind
LoveDA 47.38 and Potsdam 57.83; a resize that fired there would silently move both.

Pure arithmetic, no GPU, no model. The expression under test is lifted verbatim
from the segmentor and compared against it textually, so the two cannot drift.
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'reference' / 'segearthov3_segmentor.py'


def scale(size, max_side):
    """Verbatim copy of the segmentor's branch. Returns the size SAM 3 will see."""
    if max_side and max(size) > max_side:
        _sc = max_side / float(max(size))
        return (max(1, int(round(size[0] * _sc))),
                max(1, int(round(size[1] * _sc))))
    return size


def main():
    src = SRC.read_text()
    fail = 0

    # -- the copy above must still match the source, or this file tests nothing --
    for frag in ("if self.max_side and max(image.size) > self.max_side:",
                 "_sc = self.max_side / float(max(image.size))",
                 "image = image.resize(_ns, Image.BILINEAR)"):
        if frag not in src:
            print(f'FAIL  segmentor no longer contains: {frag}')
            fail += 1
    if 'max_side=0,' not in src:
        print('FAIL  max_side no longer defaults to 0 -- every existing run changes')
        fail += 1
    # the downscale must sit AFTER the file is re-opened; mmseg's tensor is discarded
    i_open = src.find("image = Image.open(image_path).convert('RGB')")
    i_cut = src.find('if self.max_side and max(image.size) > self.max_side:')
    if not (0 < i_open < i_cut):
        print('FAIL  the downscale is not between Image.open and the forward pass')
        fail += 1

    cases = [
        # (size,            max_side, expected,      why)
        ((1024, 1024),      0,        (1024, 1024),  'LoveDA, feature off -- the default'),
        ((1024, 1024),      1024,     (1024, 1024),  'equality must NOT resize'),
        ((1024, 1024),      1008,     (1008, 1008),  'below the tile size it WOULD fire'),
        ((6000, 6000),      0,        (6000, 6000),  'Potsdam, feature off'),
        ((3840, 2160),      1008,     (1008, 567),   'UAVid: 16:9 preserved'),
        ((3840, 2160),      0,        (3840, 2160),  'UAVid untouched when off'),
        ((900, 500),        1008,     (900, 500),    'already small -- untouched'),
        ((2160, 3840),      1008,     (567, 1008),   'portrait: long side governs'),
    ]
    for size, ms, want, why in cases:
        got = scale(size, ms)
        ok = got == want
        fail += not ok
        print(f'{"ok  " if ok else "FAIL"}  {str(size):>14} max_side={ms:<5} -> '
              f'{str(got):>14}  ({why})')

    # aspect ratio must survive to within a rounding pixel
    for size in [(3840, 2160), (4096, 2160), (1920, 1080), (2160, 3840)]:
        w, h = size
        nw, nh = scale(size, 1008)
        before, after = w / h, nw / nh
        ok = abs(before - after) / before < 0.002
        fail += not ok
        print(f'{"ok  " if ok else "FAIL"}  aspect {str(size):>14} '
              f'{before:.4f} -> {after:.4f}')

    # the guard rejects a negative
    if 'if max_side < 0:' not in src:
        print('FAIL  no guard against a negative max_side')
        fail += 1

    print()
    print('ALL PASS' if not fail else f'{fail} FAILURE(S)')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
