"""
Catch-all share, INTERVENED ON rather than stratified. ROADMAP §6.1.

THE PROBLEM THIS FIXES. WEEK3 §7 claims the catch-all's SHARE of ground truth
governs the residual and its detectability. §7a broke the share/confusability
confound by stratifying LoveDA into urban and rural -- but it says so plainly:
"this is a stratification, not a randomised intervention." Urban and rural differ
in class mix, object scale and scene density too. That sentence is the weakest
joint in the project's strongest claim, and a reviewer will find it.

THE FIX. Catch-all share is not a property we have to go looking for. It is SET
BY THE VOCABULARY we hand SAM 3. So intervene on it.

⭐ WHY THIS IS CPU-ONLY, AND WHY THAT MAKES IT BETTER, NOT JUST CHEAPER.
In segearthov3_segmentor.py every class is an INDEPENDENT forward pass with its
own text prompt (lines 141-189): `seg_logits[query_idx]` depends on `query_word`
and nothing else. There is no softmax, no normalisation, no interaction across
classes anywhere -- the ONLY cross-class operation in the whole pipeline is the
argmax in predict(). Therefore:

    dropping a class from the vocabulary  ==  dropping its channel

exactly, not approximately. One cached score stack answers every vocabulary
question. And because all arms then read the SAME model outputs, they differ in
the vocabulary and in literally nothing else -- no run-to-run variation, no
resampling noise. Re-running the model five times would be strictly worse.

THE DESIGN -- a dose-response curve WITH a class-count control.

    A_0   published vocabulary                        catch-all at its own share
    A_k   merge k real classes INTO the catch-all     share rises: ~10%, ~25%, ~40%
    C_k   merge THE SAME k classes into EACH OTHER    share UNCHANGED

⚠️ ARM C IS THE EXPERIMENT. Merging classes also reduces the class count, which
moves the detection base rate and mIoU on its own. C changes the class count by
exactly the same amount while leaving the catch-all's share untouched, so it
isolates share from arity. Without C this is a dose-response curve confounded
with "fewer classes is easier", which is no better than the stratification it
replaces.

Both arm families are implemented identically -- a group of channels reduced by
max -- which is precisely how the segmentor already handles synonym groups
(`(seg_logits * cls_index).max(1)[0]`, line 304). The two differ ONLY in whether
the group contains the catch-all.

THE REVERSE DIRECTION. `--drop-catchall` removes the catch-all from the
vocabulary altogether, so the vocabulary covers the scene. The mechanism predicts
detectability should RISE toward OpenEarthMap's regime. A mechanism that survives
being pushed in both directions is very hard to argue with.

⚠️ Predictions are pre-registered in PREREGISTRATION.md and committed BEFORE this
is run. `git log` is the timestamp. Do not edit them afterwards.

    # one GPU pass, ~25 min, writes ~10x the usual cache
    python scripts/measure_discard_rate.py --config configs/cfg_openearthmap.py \
        --tau 0.1 --cache-full --out ~/outputs/oem_full

    # every arm, CPU
    python scripts/vocab_intervention.py --cache ~/outputs/oem_full/cache \
        --tau 0.1 --targets 10 25 40 --md ~/outputs/week3/vocab_intervention.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                              # noqa: E402
from confound_split import auc_from_hist, NBINS            # noqa: E402


class Arm:
    """A vocabulary. `groups[i]` is the set of ORIGINAL 0-indexed classes reduced
    by max into output class i. `bg` is the output class the catch-all landed in,
    or None when the catch-all was dropped from the vocabulary entirely -- in
    which case sub-tau pixels go to a synthetic sink at index len(groups)."""

    def __init__(self, name, groups, bg, note=''):
        self.name, self.groups, self.bg, self.note = name, groups, bg, note
        self.sink = bg if bg is not None else len(groups)
        self.n_out = len(groups) + (0 if bg is not None else 1)

    def remap(self):
        """original class index -> output class index."""
        m = {}
        for i, g in enumerate(self.groups):
            for c in g:
                m[c] = i
        return m


def build_arms(LB, share, targets, drop_catchall):
    """Dose arms and their arity-matched controls, chosen from the data.

    The merge set S is picked greedily from the SMALLEST real classes upward, so
    the catch-all reaches each target share while disturbing as little of the
    benchmark as possible.

    ⚠️ THE CONTROL MUST BE EXACTLY ARITY-MATCHED, and getting this wrong is easy.
    A first version merged S into the catch-all for the dose arm and S into one
    real class for the control -- which leaves the control with ONE MORE output
    class, so it no longer controls for class count, which is its entire job. A
    smoke test caught it. The correct pairing at each dose is:

        dose     merge  S ∪ {catch-all}  into one class
        control  merge  S ∪ {c0}         into one class,  c0 a REAL class

    with `c0` chosen as the real class whose share is closest to the catch-all's.
    Identical output-class count, identical group cardinality, identical S, a
    receiver of comparable size -- the ONLY difference is whether the receiving
    class is the catch-all.
    """
    bg = LB.bg - 1
    real = sorted((c for c in range(LB.n) if c != bg), key=lambda c: share[c])
    if len(real) < 3:
        raise SystemExit(f'{LB.n} classes is too few to merge and still control.')
    # the real class most comparable to the catch-all, and never itself merged
    c0 = min(real, key=lambda c: abs(share[c] - share[bg]))
    pool = [c for c in real if c != c0]
    arms = [Arm('A0 published', [[c] for c in range(LB.n)], bg,
                'the unmodified vocabulary')]
    acc, S = share[bg], []
    for t in targets:
        while acc * 100 < t and len(S) < len(pool) - 1:
            c = pool[len(S)]
            S.append(c)
            acc += share[c]
        if not S:
            continue
        nm = '+'.join(LB.names[c] for c in S)
        rest_d = [c for c in real if c not in S]
        arms.append(Arm(f'A{t:g} dose', [[bg] + list(S)] + [[c] for c in rest_d], 0,
                        f'`{LB.names[bg]}` absorbs {nm} — catch-all share rises'))
        rest_c = [c for c in real if c not in S and c != c0]
        arms.append(Arm(f'C{t:g} control', [[bg]] + [[c0] + list(S)]
                        + [[c] for c in rest_c], 0,
                        f'the same {nm} merged into `{LB.names[c0]}` instead — '
                        f'identical class count, catch-all share unchanged'))
    if drop_catchall:
        arms.append(Arm('R reverse', [[c] for c in range(LB.n) if c != bg], None,
                        'the catch-all is removed from the vocabulary, which now '
                        'covers the scene; sub-τ pixels have no catch-all prompt '
                        'to fall into'))
    return arms


def scan(files, LB, arms, tau):
    """One pass over the cache, every arm scored on the same tiles."""
    nb = NBINS
    st = [dict(gt=np.zeros(a.n_out + 1, np.int64),
               sink_gt=np.zeros(a.n_out + 1, np.int64),
               pos=np.zeros(nb, np.int64), neg=np.zeros(nb, np.int64),
               pos2=np.zeros(nb, np.int64), neg2=np.zeros(nb, np.int64),
               C=np.zeros((a.n_out, a.n_out), np.int64)) for a in arms]
    remaps = [a.remap() for a in arms]
    for i, f in enumerate(files):
        z = np.load(f)
        if 'logits' not in z.files:
            raise SystemExit(
                f'{f.name} has no `logits` key. This experiment needs the full '
                f'per-class score stack, which only --cache-full writes:\n\n'
                f'    python scripts/measure_discard_rate.py --cache-full ...')
        L = z['logits'].astype(np.float32)                 # (N, H, W)
        gt = z['gt'].astype(np.int32)
        valid = gt > 0
        g0 = gt[valid] - 1                                  # original class index
        for a, s, rm in zip(arms, st, remaps):
            G = np.stack([L[g].max(0) for g in a.groups])[:, valid]
            k = min(2, G.shape[0])
            order = np.argpartition(-G, k - 1, axis=0)[:k]
            top = np.take_along_axis(G, order, 0)
            o = np.argsort(-top, axis=0)
            srt = np.take_along_axis(top, o, 0)
            conf, conf2 = srt[0], (srt[1] if k > 1 else srt[0])
            pred = np.take_along_axis(order, o, 0)[0]
            pred = np.where(conf < tau, a.sink, pred)
            gg = np.array([rm.get(c, a.sink) for c in range(LB.n)])[g0]
            np.add.at(s['gt'], gg, 1)
            m = pred == a.sink
            if m.any():
                np.add.at(s['sink_gt'], gg[m], 1)
                b = np.clip((conf[m] * nb).astype(np.int32), 0, nb - 1)
                b2 = np.clip((conf2[m] * nb).astype(np.int32), 0, nb - 1)
                real_px = gg[m] != a.sink
                np.add.at(s['pos'], b[real_px], 1)
                np.add.at(s['neg'], b[~real_px], 1)
                np.add.at(s['pos2'], b2[real_px], 1)
                np.add.at(s['neg2'], b2[~real_px], 1)
            if a.bg is not None:
                np.add.at(s['C'], (np.minimum(gg, a.n_out - 1), pred), 1)
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}')
    return st


def miou(C):
    inter = np.diag(C).astype(float)
    union = C.sum(0) + C.sum(1) - np.diag(C)
    with np.errstate(invalid='ignore', divide='ignore'):
        v = np.where(union > 0, inter / union, np.nan)
    return float(np.nanmean(v)) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--targets', type=float, nargs='+', default=[10, 25, 40],
                    help='catch-all share (%%) each dose arm should reach')
    ap.add_argument('--drop-catchall', action='store_true',
                    help='add the reverse arm: remove the catch-all from the '
                         'vocabulary entirely (use on LoveDA)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')
    print(f'  classes: {LB}\n  {len(files)} tiles | τ = {args.tau}')

    # class shares, from ground truth, to choose the merge sets
    cnt = np.zeros(LB.n, np.int64)
    for f in files:
        g = np.load(f)['gt'].astype(np.int32)
        np.add.at(cnt, g[g > 0] - 1, 1)
    share = cnt / cnt.sum()
    print('  GT share: ' + ', '.join(f'{LB.names[c]} {share[c] * 100:.2f}%'
                                     for c in np.argsort(-share)))

    arms = build_arms(LB, share, args.targets, args.drop_catchall)
    print(f'  {len(arms)} arms\n')
    st = scan(files, LB, arms, args.tau)

    rows = []
    for a, s in zip(arms, st):
        tot = s['gt'].sum()
        sh = s['gt'][a.sink] / tot * 100
        real_tot = tot - s['gt'][a.sink]
        disc = s['sink_gt'][:a.sink].sum() + s['sink_gt'][a.sink + 1:].sum()
        base = disc / max(s['sink_gt'].sum(), 1) * 100
        rows.append(dict(
            arm=a, share=sh, classes=len(a.groups),
            discard=disc / max(real_tot, 1) * 100, base=base,
            auc=auc_from_hist(s['pos'], s['neg']),
            auc2=auc_from_hist(s['pos2'], s['neg2']),
            miou=miou(s['C']) if a.bg is not None else float('nan')))

    md = [f'# Catch-all share — intervened on, not stratified\n',
          f'- cache: `{args.cache}` | tiles: **{len(files)}** | τ = **{args.tau}**',
          f'- catch-all: **`{LB.names[LB.bg - 1]}`**\n',
          'Every class is an independent forward pass with its own text prompt and the '
          'only cross-class operation in the pipeline is the `argmax`, so **dropping a '
          'class from the vocabulary is exactly equivalent to dropping its channel**. '
          'All arms therefore read one cached score stack and differ in the vocabulary '
          'and in nothing else — no run-to-run variation.\n',
          '⚠️ **The `C` arms are the experiment.** Merging classes also reduces the class '
          'count, which moves the base rate and mIoU on its own. Each `C` arm merges the '
          '*same* classes into one another instead of into the catch-all, so the arity '
          'change is identical and only the share differs.\n',
          '| arm | vocabulary size | catch-all share of GT | discard % | base % | **AUC `conf`** | '
          'AUC `conf2` | mIoU |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append(f'| **{r["arm"].name}** | {r["classes"]} | **{r["share"]:.2f}%** | '
                  f'{r["discard"]:.2f} | {r["base"]:.1f} | **{r["auc"]:.3f}** | '
                  f'{r["auc2"]:.3f} | '
                  + (f'{r["miou"]:.2f}' if np.isfinite(r['miou']) else '—') + ' |')
    md.append('')
    for r in rows:
        md.append(f'- **{r["arm"].name}** — {r["arm"].note}.')

    dose = [r for r in rows if r['arm'].name.startswith('A') and r['arm'].name != 'A0 published']
    ctrl = [r for r in rows if r['arm'].name.startswith('C')]
    base0 = next(r for r in rows if r['arm'].name == 'A0 published')

    md += ['\n## Verdict\n']
    if dose and ctrl:
        d_drop = base0['auc'] - dose[-1]['auc']
        c_drop = base0['auc'] - ctrl[-1]['auc']
        md.append('| | share at the largest dose | AUC change from `A0` |')
        md.append('|---|---|---|')
        md.append(f'| **dose** (`{dose[-1]["arm"].name}`) | '
                  f'{base0["share"]:.2f}% → **{dose[-1]["share"]:.2f}%** | '
                  f'**{-d_drop:+.3f}** |')
        md.append(f'| **control** (`{ctrl[-1]["arm"].name}`) | '
                  f'{base0["share"]:.2f}% → {ctrl[-1]["share"]:.2f}% (unchanged) | '
                  f'{-c_drop:+.3f} |')
        mono = all(dose[i]['auc'] >= dose[i + 1]['auc'] for i in range(len(dose) - 1))
        if d_drop > 0 and d_drop > 2 * abs(c_drop):
            md.append(f'\n⭐ **Catch-all share is CAUSAL.** Raising it from '
                      f'{base0["share"]:.2f}% to {dose[-1]["share"]:.2f}% costs '
                      f'**{d_drop:.3f} AUC**, while the control — the same classes '
                      f'merged, the same class count, the share left alone — moves only '
                      f'{c_drop:+.3f}. The effect follows the share, not the arity.'
                      + (' Detection is monotone across the doses.' if mono else
                         ' ⚠️ It is **not** monotone across the doses, so report the '
                         'endpoints and say so.')
                      + '\n\n**§7 upgrades from a stratification to an intervention**, and '
                        'the concession in §7a ("not a randomised intervention") can be '
                        'replaced by this table.')
        elif d_drop > 0:
            md.append(f'\n⚠️ **AUC falls with the dose ({d_drop:.3f}) but the control '
                      f'moves comparably ({c_drop:+.3f}).** The effect is at least partly '
                      f'the class count, not the share. §7a\'s stratification stands as '
                      f'the stronger evidence and this must be reported as inconclusive — '
                      f'it is exactly the confound arm C exists to expose.')
        else:
            md.append(f'\n⛔ **The prediction FAILS.** Raising the catch-all\'s share does '
                      f'not reduce detectability ({-d_drop:+.3f} AUC). §7\'s mechanism is '
                      f'not causal in the direction claimed, and the paper must say so. '
                      f'The stratified result (§7a) then needs a different explanation — '
                      f'most likely something urban and rural differ in besides these two '
                      f'variables.')
    rev = [r for r in rows if r['arm'].name.startswith('R')]
    if rev:
        d = rev[0]['auc'] - base0['auc']
        md.append(f'\n**Reverse arm.** Removing the catch-all from the vocabulary moves '
                  f'AUC {d:+.3f} ({base0["auc"]:.3f} → {rev[0]["auc"]:.3f}). '
                  + ('✅ Detectability rises when the vocabulary covers the scene, which '
                     'is the mechanism pushed in the opposite direction and holding.'
                     if d > 0.02 else
                     '⛔ It does not rise, so the mechanism does not survive the reverse '
                     'intervention. Report this beside the forward result.')
                  + ' ⚠️ mIoU is not comparable for this arm — it has no catch-all class '
                    'to predict — so only the detection columns are meaningful.')
    md.append('\n⚠️ Compare against the predictions in `PREREGISTRATION.md`, which was '
              'committed before this was run. Do not edit them now.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
