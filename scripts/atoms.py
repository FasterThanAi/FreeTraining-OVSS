"""Shared atomisation for the Week 3 region experiments.

atom_quality.py settled which atoms to use, and the gap is not marginal:

    atoms   count     median px   mean purity   ORACLE CEILING
    cc      41,952    179         0.728         72.8%
    slic    506,064   1,551       0.928         92.8%

Connected components of "assigned to background" sprawl -- the largest was
1,048,576 px, an entire tile as one atom -- so a per-atom label was wrong for 27%
of its pixels before any scoring happened. SLIC respects colour/texture edges and
is size-bounded (max 60,740 px), which lifts the ceiling on ANY region-level
method by 20 points.

Every experiment that assigns one label per region must therefore run on SLIC
atoms, not connected components. This module exists so the three scripts that do
so cannot drift apart in how they define an atom.
"""
import numpy as np


def atoms_cc(mask, img=None, n_segments=600):
    """4-connected components of the candidate mask. Kept for the ablation row
    only -- see the ceiling gap above."""
    from scipy.ndimage import label
    lab, n = label(mask)
    return lab.astype(np.int32), n


def atoms_slic(mask, img, n_segments=600):
    """SLIC over the whole image, intersected with the candidate mask.

    Segmenting the FULL image rather than the mask matters: superpixel boundaries
    should follow the image's own edges, not the arbitrary outline of wherever
    the model happened to be unconfident.
    """
    from skimage.segmentation import slic
    seg = slic(img, n_segments=n_segments, compactness=10.0, start_label=1)
    seg = np.where(mask, seg, 0)
    u = np.unique(seg)
    u = u[u > 0]
    if len(u) == 0:
        return np.zeros_like(seg, np.int32), 0
    remap = np.zeros(int(seg.max()) + 1, np.int32)
    remap[u] = np.arange(1, len(u) + 1)
    return remap[seg].astype(np.int32), len(u)


def get_atomiser(kind):
    return {'cc': atoms_cc, 'slic': atoms_slic}[kind]


def load_image(img_dir, stem):
    from PIL import Image
    from pathlib import Path
    p = Path(img_dir).expanduser() / f'{stem}.png'
    if not p.exists():
        for ext in ('.jpg', '.tif', '.tiff'):
            q = p.with_suffix(ext)
            if q.exists():
                p = q
                break
    return np.array(Image.open(p).convert('RGB'))
