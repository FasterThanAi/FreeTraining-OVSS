"""
Share or confusability? Breaking the n=2 confound without a third dataset.

THE PROBLEM. WEEK3 §7 attributes everything to the catch-all's SHARE of ground
truth (LoveDA 36.1%, OpenEarthMap 0.84%). But across those two points share moves
together with a second property -- whether the catch-all LOOKS like the real
classes -- and the second is the more plausible cause. `conf2` may fail on LoveDA
not because background is COMMON but because it genuinely resembles `road`,
`barren` and `building`, so a strong runner-up cannot tell a suppressed real class
from a background region that merely looks like one. Two observational points
cannot separate confounded variables.

⛔ A THIRD OBSERVATIONAL POINT DOES NOT FIX THIS unless share and confusability
DISSOCIATE in it. iSAID does not: its catch-all is 97.11% of ground truth AND
maximally confusable, since it is everything outside 15 small object classes. It
would confirm the ordering and discriminate nothing. Check any candidate dataset
against this file before spending GPU time on it.

WHAT THIS SCRIPT DOES INSTEAD, on caches that already exist, with no GPU:

  1. MEASURES confusability instead of asserting it. The catch-all's confusability
     with the real classes is observable in the confusion matrix -- the fraction of
     true catch-all pixels the model gives a real class, and the reverse. Both are
     already computed for every dataset here. That alone upgrades §7's second
     column from an assertion to a number.

  2. SPLITS one dataset into strata where the two DISSOCIATE. LoveDA's urban and
     rural domains differ in both, and in OPPOSITE directions: rural has the
     larger catch-all share (~43% vs ~26% of area) while urban's catch-all is
     pavement and built structure, which is far more confusable with `road` and
     `building`. So:

         detection better in RURAL (higher share, lower confusability)
             -> CONFUSABILITY drives it; §7's mechanism must be restated
         detection better in URBAN (lower share, higher confusability)
             -> SHARE drives it; §7 stands as written

     Either way one explanation is eliminated, which is what n=2 cannot do.

⚠️ This is a stratification, not a randomised intervention. Urban and rural differ
in more than these two variables, so it constrains the explanation rather than
proving one. Say so in the paper.

    # build the domain map once from the original Kaggle layout
    python scripts/confound_split.py --make-map ~/data/loveda_raw/Val \\
        --map-out ~/splits/loveda_domain.txt

    python scripts/confound_split.py --cache ~/outputs/week3_fused/cache \\
        --tau 0.5 --map ~/splits/loveda_domain.txt --md ~/outputs/week3/confound.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                            # noqa: E402


def make_map(root, out):
    """LoveDA ships Val/Rural/images_png and Val/Urban/images_png. Once the two
    are merged into one directory for evaluation the domain is lost, so recover
    it from the original layout rather than guessing at ID ranges."""
    root = Path(root).expanduser()
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = list(d.rglob('*.png')) + list(d.rglob('*.tif'))
        stems = sorted({p.stem for p in imgs if 'mask' not in p.parent.name.lower()})
        if not stems:
            continue
        print(f'  {d.name}: {len(stems)} tiles')
        rows += [f'{s}\t{d.name.lower()}' for s in stems]
    if not rows:
        raise SystemExit(f'no image directories under {root}')
    p = Path(out).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(rows) + '\n')
    print(f'\nwritten: {p}  ({len(rows)} tiles)')


def auc(score, label):
    """Rank AUC. `label` is 1 for a recoverable pixel (real class assigned to the
    catch-all), 0 for genuine catch-all."""
    ok = np.isfinite(score)
    s, y = score[ok], label[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    r = np.empty(len(s))
    order = np.argsort(s, kind='stable')
    r[order] = np.arange(1, len(s) + 1)
    # average ranks over ties, or a coarse float16 score inflates the AUC
    su = np.sort(s)
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def scan(files, LB, tau):
    nc, BG = LB.nc, LB.bg
    C = np.zeros((nc, nc), np.int64)
    sc = {k: [] for k in ('conf', 'conf2', 'gap', 'fconf')}
    lab = []
    for f in files:
        z = np.load(f)
        gt = z['gt'].astype(np.int32)
        conf = z['conf'].astype(np.float32)
        pred = (z['pred'].astype(np.int32) + 1)
        conf2 = z['conf2'].astype(np.float32) if 'conf2' in z.files else conf
        fconf = z['fconf'].astype(np.float32) if 'fconf' in z.files else conf * np.nan
        pred = np.where(conf < tau, BG, pred)
        m = gt > 0
        np.add.at(C, (gt[m], pred[m]), 1)
        # the detection problem exactly as §9 poses it: among pixels the pipeline
        # assigned to the catch-all, which ones are really a land-cover class?
        d = m & (pred == BG)
        if d.any():
            lab.append((gt[d] != BG).astype(np.int8))
            sc['conf'].append(conf[d]); sc['conf2'].append(conf2[d])
            sc['gap'].append(conf[d] - conf2[d]); sc['fconf'].append(fconf[d])
    return C, {k: np.concatenate(v) for k, v in sc.items() if v}, np.concatenate(lab)


def stats(C, S, y, LB):
    nc, BG = LB.nc, LB.bg
    real = [c for c in range(1, nc) if c != BG]
    tot = C[1:, 1:].sum()
    bg_gt = C[BG].sum()
    # confusability, MEASURED: how much mass crosses between the catch-all and the
    # real classes in each direction, as a share of the catch-all's own totals
    bg_to_real = C[BG, real].sum() / max(bg_gt, 1)
    real_to_bg = C[np.ix_(real, [BG])].sum() / max(C[real, :].sum(), 1)
    return dict(
        share=100 * bg_gt / max(tot, 1),
        conf_out=100 * bg_to_real,          # true catch-all given a real class
        conf_in=100 * real_to_bg,           # real class given the catch-all
        discard=100 * C[np.ix_(real, [BG])].sum() / max(C[real, :].sum(), 1),
        base=100 * y.mean(),
        **{f'auc_{k}': auc(v, y) for k, v in S.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make-map', default=None, help='original split root, e.g. .../Val')
    ap.add_argument('--map-out', default=None)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--map', default=None, help='TSV: tile-stem <TAB> domain')
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    if args.make_map:
        return make_map(args.make_map, args.map_out or 'domain_map.txt')
    if not (args.cache and args.tau is not None):
        raise SystemExit('need --cache and --tau (or --make-map)')

    LB = labels.from_cache(args.cache)
    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    groups = {'all': files}
    if args.map:
        dom = {}
        for ln in Path(args.map).expanduser().read_text().splitlines():
            if ln.strip():
                a, _, b = ln.partition('\t')
                dom[a.strip()] = b.strip() or 'unknown'
        miss = [f for f in files if f.stem not in dom]
        if miss:
            print(f'  !! {len(miss)} tiles absent from the map, e.g. {miss[0].stem} '
                  f'-- grouped as "unmapped"')
        for f in files:
            groups.setdefault(dom.get(f.stem, 'unmapped'), []).append(f)
    print(f'  classes: {LB}\n  groups: '
          + ', '.join(f'{k} ({len(v)})' for k, v in groups.items()) + '\n')

    R = {}
    for name, fs in groups.items():
        print(f'  scanning {name} ({len(fs)} tiles)…')
        R[name] = stats(*scan(fs, LB, args.tau), LB)

    md = [f'# Share or confusability? — {Path(args.cache).name}\n',
          f'- cache: `{args.cache}`  |  τ = **{args.tau}**  |  catch-all: '
          f'**`{LB.names[LB.bg - 1]}`**\n',
          '`share` is the catch-all\'s share of ground truth. **`conf_out` is '
          'confusability, measured**: the percentage of true catch-all pixels the '
          'model gives a real class — high means the catch-all looks like real land '
          'cover. `base` is the detection base rate.\n',
          '| group | tiles | share % | **conf_out %** | conf_in % | discard % | base % | '
          'AUC `conf` | AUC `conf2` | AUC `gap` |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for k, v in R.items():
        md.append(f'| **{k}** | {len(groups[k])} | {v["share"]:.1f} | '
                  f'**{v["conf_out"]:.1f}** | {v["conf_in"]:.1f} | {v["discard"]:.1f} | '
                  f'{v["base"]:.1f} | {v.get("auc_conf", float("nan")):.3f} | '
                  f'{v.get("auc_conf2", float("nan")):.3f} | '
                  f'{v.get("auc_gap", float("nan")):.3f} |')
    md.append('')

    strata = [k for k in R if k != 'all']
    if len(strata) == 2:
        a, b = sorted(strata, key=lambda k: -R[k]['share'])   # a = higher share
        ra, rb = R[a], R[b]
        best = lambda v: max(v.get('auc_conf2', 0), v.get('auc_conf', 0))
        diss = (ra['conf_out'] < rb['conf_out'])              # do they dissociate?
        md += ['## Verdict\n',
               f'`{a}` has the **higher catch-all share** ({ra["share"]:.1f}% vs '
               f'{rb["share"]:.1f}%) and `{b}` the **higher measured confusability** '
               f'({rb["conf_out"]:.1f}% vs {ra["conf_out"]:.1f}%).\n'
               if diss else
               f'⛔ **The two strata do not dissociate.** `{a}` has both the higher '
               f'share ({ra["share"]:.1f}%) and the higher confusability '
               f'({ra["conf_out"]:.1f}%), so this split reproduces the same confound '
               f'as the two datasets and cannot break it. Find a stratification where '
               f'they move in opposite directions, or say plainly in the paper that '
               f'the mechanism is stated at the level of label design and not '
               f'attributed to either variable alone.\n']
        if diss:
            if best(rb) > best(ra) + 0.02:
                md.append(f'⭐ **Detection is better in `{b}` — the higher-confusability, '
                          f'lower-share stratum ({best(rb):.3f} vs {best(ra):.3f}).** That '
                          'is the opposite of what the share explanation predicts, so '
                          '**share is not sufficient** and §7 must be restated in terms '
                          'of confusability.')
            elif best(ra) > best(rb) + 0.02:
                md.append(f'✅ **Detection is better in `{a}` — the lower-confusability, '
                          f'higher-share stratum ({best(ra):.3f} vs {best(rb):.3f}).** '
                          'Share and detectability move together even where confusability '
                          'opposes them, which supports §7 as written.')
            else:
                md.append(f'⚠️ **Neither: the two strata detect equally well** '
                          f'({best(ra):.3f} vs {best(rb):.3f}, difference below 0.02). '
                          'Within one dataset the variation in both variables is too '
                          'small to separate them. Report the confound as open and use '
                          'a dataset whose catch-all is common but visually distinct.')
        md.append('\n⚠️ This is a stratification, not a randomised intervention — the '
                  'strata differ in more than these two variables — so it constrains the '
                  'explanation rather than proving one. State that beside the result.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text); print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
