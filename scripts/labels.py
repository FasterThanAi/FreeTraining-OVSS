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


# Datasets name the catch-all differently. LoveDA and OpenEarthMap call it
# `background`; ISPRS Potsdam and Vaihingen call it `clutter`. Checked in this
# order, so an explicit `background` always wins.
CATCH_ALL_ALIASES = ['background', 'clutter', 'unlabeled', 'unlabelled',
                     'void', 'ignore', 'other', 'misc']


class Labels:
    """names[i] is the class at mask value i+1. Mask value 0 is always no-data."""

    def __init__(self, names, bg_name=None, discard_idx=None):
        """`discard_idx` is the segmentor's ACTUAL `bg_idx` (0-indexed), where it
        is known. Pass it and the catch-all/discard-target check becomes a real
        comparison instead of an assumption -- see the note below."""
        self.names = [str(n) for n in names]
        self.n = len(self.names)
        # True when the discard target sits outside the class list, so it is
        # never scored. Consumers size their confusion matrix with `n_pred`.
        self.sink = False
        self.nc = self.n + 1                      # mask-space width incl. no-data
        low = [n.lower() for n in self.names]

        self.bg = None
        for cand in ([bg_name.lower()] if bg_name else CATCH_ALL_ALIASES):
            if cand in low:
                self.bg = low.index(cand) + 1
                break
        # ⭐ A dataset can legitimately have NO catch-all. DLRSD is the first:
        # 17 classes, all of them real things, every one of its 137,625,600
        # pixels labelled, index 0 absent from all 2100 files. There the
        # segmentor still needs somewhere to send sub-tau pixels, and
        # cfg_dlrsd.py points `bg_idx` at an index OUTSIDE the class list -- an
        # unscored sink. A discarded pixel then becomes a false negative for
        # its true class and a false positive for nothing.
        #
        # ⛔ Without this branch the fallback below fires and silently nominates
        # the FIRST class as the catch-all (`airplane` on DLRSD), which is the
        # `g > BACKGROUND` mistake from Potsdam wearing a different hat.
        if self.bg is None and discard_idx is not None and discard_idx >= self.n:
            self.bg = discard_idx + 1              # mask value, outside 1..n
            self.sink = True
            print(f'  ⭐ no catch-all class, and the segmentor sends sub-τ pixels '
                  f'to bg_idx={discard_idx}, outside the {self.n} classes. '
                  f'Treating it as an UNSCORED SINK: a discarded pixel is a '
                  f'false negative for its true class and a false positive for '
                  f'nothing. Full mIoU == catch-all-excluded mIoU here.')
        elif self.bg is None:
            # ⛔ THIS USED TO BE A WARNING AND IT COST A WHOLE ANALYSIS PASS.
            # On DLRSD it printed "assuming mask value 1 (airplane)" at the top
            # of a two-minute run, scrolled away, and every sub-τ pixel in the
            # dataset was then counted as a predicted `airplane`. Both the
            # oracle and the 5-fold produced complete, plausible, entirely void
            # tables. A warning at the start of a long run is not a safeguard.
            raise SystemExit(
                f'⛔ NO CATCH-ALL CLASS, AND NO DISCARD TARGET GIVEN.\n'
                f'   classes: {self.names}\n'
                f'   none matches {CATCH_ALL_ALIASES}, and '
                + (f'the segmentor\'s bg_idx={discard_idx} is INSIDE the class '
                   f'range, so it is a real class rather than an unscored sink.\n'
                   if discard_idx is not None else
                   'no `bg_idx` was supplied, so where sub-τ pixels go is '
                   'unknown.\n')
                + f'   Guessing would silently make `{self.names[0]}` the discard '
                  f'target and pile every thresholded pixel onto it.\n\n'
                  f'   Fix: pass bg_name=, or write the cache sidecar so this is '
                  f'read rather than guessed:\n'
                  f'     python scripts/write_cache_meta.py --cache <dir> '
                  f'--bg-idx <n>')
        elif self.bg != 1:
            # The segmentor's `bg_idx` decides where sub-tau pixels go. Its
            # DEFAULT is 0, so on a dataset whose catch-all is not first the
            # discard target and the catch-all can differ -- and conflating them
            # invalidates every number silently.
            #
            # ⚠️ But the config can override it, and Potsdam's does (`bg_idx=5`,
            # `clutter`). An earlier version warned unconditionally whenever
            # `bg != 1`, so it cried wolf on every Potsdam run while the two
            # actually agreed. A check that fires when nothing is wrong stops
            # being read, which is worse than no check. Compare the real value.
            if discard_idx is None:
                print(f'  note: catch-all `{self.names[self.bg - 1]}` is at mask '
                      f'value {self.bg}, not 1. Could not read the segmentor\'s '
                      f'`bg_idx` here, so the discard target is unverified -- '
                      f'confirm the config sets bg_idx={self.bg - 1}.')
            elif discard_idx + 1 != self.bg:
                print(f'  !! MISMATCH: the catch-all is `{self.names[self.bg - 1]}` '
                      f'(mask value {self.bg}) but the segmentor sends sub-τ pixels '
                      f'to bg_idx={discard_idx} '
                      f'(`{self.names[discard_idx]}`). The DISCARD TARGET and the '
                      f'CATCH-ALL CLASS are different, and every discard number '
                      f'below is measuring the wrong class.')
        self.real = [c for c in range(1, self.nc) if c != self.bg]

    @property
    def catch_all(self):
        """Display name of the catch-all, or None where none exists.

        Four report scripts did `LB.names[LB.bg - 1]` directly, which is an
        IndexError the moment a dataset has no catch-all. Ask for this instead
        and handle None.
        """
        return None if self.sink else self.names[self.bg - 1]

    @property
    def n_pred(self):
        """Width of a confusion matrix's PREDICTED axis.

        Equal to `n` normally. With an unscored sink the segmentor can emit one
        index past the class list, so the matrix needs an extra column -- and
        that column is deliberately NOT a class: nothing iterates over it, so
        the sink contributes to no IoU. ⛔ Sizing the matrix `n x n` instead
        raises `IndexError: index 17 is out of bounds` at best, and at worst a
        `np.clip` upstream folds every discarded pixel onto the last class.
        """
        return max(self.n, self.bg)

    def name(self, mask_value):
        return self.names[mask_value - 1]

    def __repr__(self):
        where = ('unscored sink' if self.sink
                 else f'background=mask value {self.bg}')
        return f'Labels({self.n} classes, {where}, {", ".join(self.names)})'


CACHE_META = '_meta.json'


def from_cache(cache_dir, bg_name=None, discard_idx=None):
    """Read the class list written into the .npz cache.

    ⭐ `discard_idx` -- where the segmentor sends sub-τ pixels -- is read from a
    `_meta.json` sidecar written beside the cache, because it is NOT derivable
    from the arrays. Where the catch-all is findable by name this only turns an
    assumption into a check; where a dataset has no catch-all at all (DLRSD) it
    is the difference between a correct run and a void one, and the void one
    does not look void.
    """
    cache_dir = Path(cache_dir).expanduser()
    if discard_idx is None:
        meta = cache_dir / CACHE_META
        if meta.is_file():
            import json
            try:
                discard_idx = json.loads(meta.read_text()).get('bg_idx')
                if discard_idx is not None:
                    print(f'  cache meta: bg_idx={discard_idx}')
            except Exception as e:                        # noqa: BLE001
                print(f'  !! could not read {meta}: {e}')
    files = sorted(cache_dir.glob('*.npz'))
    if not files:
        raise SystemExit(f'no .npz under {cache_dir}')
    z = np.load(files[0], allow_pickle=True)
    if 'classes' in z.files:
        return Labels([str(x) for x in z['classes']], bg_name, discard_idx)
    print('  !! cache has no `classes` key (written before this was recorded); '
          'falling back to LoveDA. Re-run measure_discard_rate.py to fix.')
    return Labels(LOVEDA_FALLBACK, bg_name, discard_idx)


def from_model(model, cfg=None):
    """Class list for an mmseg/SegEarth-OV3 model, for the inference-time script.

    Prefers mmseg's own metadata; falls back to the prompt file, where each LINE
    is one class and commas separate synonyms of it -- `building,house` is one
    class, not two. The first synonym is taken as the display name.
    """
    # The segmentor carries the value that actually decides where sub-tau pixels
    # go, so read it rather than assuming the class default.
    bgi = getattr(model, 'bg_idx', None)
    meta = getattr(model, 'dataset_meta', None) or {}
    if meta.get('classes'):
        return Labels(list(meta['classes']), None, bgi)

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
            return Labels(names, None, bgi)

    print('  !! could not resolve class names from model or config; '
          'falling back to LoveDA. Verify before trusting any output.')
    return Labels(LOVEDA_FALLBACK, None, bgi)
