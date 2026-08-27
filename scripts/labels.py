"""Dataset-agnostic class handling.

Every Week 2/3 script hardcoded LoveDA's seven classes and assumed `background`
sits at index 0. Pointed at OpenEarthMap -- 8 classes, different order -- they
would not crash; they would silently compute nonsense, because the label indices
would still be valid array positions. That is the worst failure mode available,
so the names come from one place now.

THE CONVENTION, everywhere in this repo:

    ground-truth masks   0 = no-data / ignore, i+1 = classes[i]
    cached `pred`        0-indexed into classes, so pred + 1 lands in mask space
    cached `classes`     the authority, written by measure_discard_rate.py

`background` is located BY NAME rather than assumed to be first, because the
whole project is about pixels assigned to it and an off-by-one there would
invalidate every number silently.
"""
from pathlib import Path

import numpy as np

LOVEDA_FALLBACK = ['background', 'building', 'road', 'water',
                   'barren', 'forest', 'agricultural']


class Labels:
    """names[i] is the class at mask value i+1. Mask value 0 is always no-data."""

    def __init__(self, names):
        self.names = [str(n) for n in names]
        self.n = len(self.names)
        self.nc = self.n + 1                      # mask-space width incl. no-data
        low = [n.lower() for n in self.names]
        if 'background' in low:
            self.bg = low.index('background') + 1
        else:
            self.bg = 1
            print(f'  !! no class literally named "background" in {self.names}; '
                  f'assuming mask value 1 ({self.names[0]}). Check this.')
        self.real = [c for c in range(1, self.nc) if c != self.bg]

    def name(self, mask_value):
        return self.names[mask_value - 1]

    def __repr__(self):
        return (f'Labels({self.n} classes, background=mask value {self.bg}, '
                f'{", ".join(self.names)})')


def from_cache(cache_dir):
    """Read the class list written into the .npz cache."""
    files = sorted(Path(cache_dir).expanduser().glob('*.npz'))
    if not files:
        raise SystemExit(f'no .npz under {cache_dir}')
    z = np.load(files[0], allow_pickle=True)
    if 'classes' in z.files:
        return Labels([str(x) for x in z['classes']])
    print('  !! cache has no `classes` key (written before this was recorded); '
          'falling back to LoveDA. Re-run measure_discard_rate.py to fix.')
    return Labels(LOVEDA_FALLBACK)


def from_model(model, cfg=None):
    """Class list for an mmseg/SegEarth-OV3 model, for the inference-time script.

    Prefers mmseg's own metadata; falls back to the prompt file, where each LINE
    is one class and commas separate synonyms of it -- `building,house` is one
    class, not two. The first synonym is taken as the display name.
    """
    meta = getattr(model, 'dataset_meta', None) or {}
    if meta.get('classes'):
        return Labels(list(meta['classes']))

    path = None
    for src in (cfg, getattr(model, 'cfg', None)):
        if src is not None:
            path = (getattr(src, 'classname_path', None)
                    or (src.get('classname_path') if hasattr(src, 'get') else None))
            if path:
                break
    path = path or getattr(model, 'classname_path', None)
    if path and Path(path).exists():
        names = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                names.append(line.split(',')[0].strip())
        if names:
            return Labels(names)

    print('  !! could not resolve class names from model or config; '
          'falling back to LoveDA. Verify before trusting any output.')
    return Labels(LOVEDA_FALLBACK)
