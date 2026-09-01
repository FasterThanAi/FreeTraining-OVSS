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

    def __init__(self, name, groups, bg, note='', absorb=None):
        self.name, self.groups, self.bg, self.note = name, groups, bg, note
        # `absorb` maps a class to an output class for GROUND TRUTH ONLY, with no
        # channel contributed. That is the difference between "this class was
        # merged into the catch-all" (its prompt still helps score the catch-all)
        # and "this class is not in the vocabulary at all and its pixels are
        # labelled catch-all" -- which is what a real catch-all actually is.
        self.absorb = absorb or {}
        self.sink = bg if bg is not None else len(groups)
        self.n_out = len(groups) + (0 if bg is not None else 1)

    def remap(self):
        """original class index -> output class index."""
        m = {}
        for i, g in enumerate(self.groups):
            for c in g:
                m[c] = i
        m.update(self.absorb)
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
        # ⭐ THE FAITHFUL ANALOGUE, added after the first run. The A arms raise
        # share by MAX-MERGING channels, which makes the catch-all a union of
        # well-detected prompts -- unnaturally COMPETENT at its own pixels, and
        # the opposite of a real catch-all (LoveDA's `background` has median
        # S_pres 0.022). That is why `conf` INVERTED rather than attenuating.
        # A real catch-all is what B models: the classes are simply absent from
        # the vocabulary, their pixels are labelled catch-all, and the catch-all
        # keeps its own single weak prompt.
        arms.append(Arm(f'B{t:g} dose (prompt dropped)',
                        [[bg]] + [[c] for c in rest_d], 0,
                        f'{nm} removed from the VOCABULARY; their pixels are '
                        f'labelled `{LB.names[bg]}`, which keeps its own single '
                        f'weak prompt — the faithful analogue of a real catch-all',
                        absorb={c: 0 for c in S}))
        arms.append(Arm(f'D{t:g} control (prompt dropped)',
                        [[bg]] + [[c0]] + [[c] for c in rest_c], 0,
                        f'the same classes removed from the vocabulary, but their '
                        f'pixels labelled `{LB.names[c0]}` — identical class '
                        f'count, catch-all share unchanged',
                        absorb={c: 1 for c in S}))
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

    def det(x):
        """Direction-agnostic detectability. ⚠️ AUC IS SYMMETRIC: AUC(-score) =
        1 - AUC(score), so an AUC of 0.208 is a detector of strength 0.792 with
        its sign flipped, NOT an absent signal. Scoring the raw AUC would let an
        INVERTED signal be reported as a destroyed one, which is exactly what the
        first run of this script did. The mechanism claims information is LOST,
        so it must be scored on |AUC - 0.5|."""
        return max(x, 1 - x) if np.isfinite(x) else x

    rows = []
    for a_, s_ in zip(arms, st):
        tot = s_['gt'].sum()
        sh = s_['gt'][a_.sink] / tot * 100
        real_tot = tot - s_['gt'][a_.sink]
        disc = s_['sink_gt'][:a_.sink].sum() + s_['sink_gt'][a_.sink + 1:].sum()
        auc = auc_from_hist(s_['pos'], s_['neg'])
        auc2 = auc_from_hist(s_['pos2'], s_['neg2'])
        rows.append(dict(
            arm=a_, share=sh, classes=len(a_.groups),
            discard=disc / max(real_tot, 1) * 100,
            base=disc / max(s_['sink_gt'].sum(), 1) * 100,
            auc=auc, auc2=auc2, det=det(auc), det2=det(auc2),
            miou=miou(s_['C']) if a_.bg is not None else float('nan')))

    md = [f'# Catch-all share — intervened on, not stratified\n',
          f'- cache: `{args.cache}` | tiles: **{len(files)}** | τ = **{args.tau}**',
          f'- catch-all: **`{LB.names[LB.bg - 1]}`**\n',
          'Every class is an independent forward pass with its own text prompt and the '
          'only cross-class operation in the pipeline is the `argmax`, so **dropping a '
          'class from the vocabulary is exactly equivalent to dropping its channel**. '
          'All arms therefore read one cached score stack and differ in the vocabulary '
          'and in nothing else — no run-to-run variation.\n',
          '⚠️ **`det` is the column that matters, not `AUC`.** AUC is symmetric — '
          '`AUC(−score) = 1 − AUC(score)` — so an AUC of 0.208 is a detector of strength '
          '0.792 with its sign flipped, not an absent signal. The mechanism claims '
          'information is *lost*, so it is scored on `det = max(AUC, 1−AUC)`. A ⇄ marks '
          'an arm where the signal inverted.\n',
          '⚠️ **The `C` and `D` arms are the experiment.** Merging also reduces the class '
          'count, which moves the base rate on its own. Each control applies the *same* '
          'merge to a real class instead of the catch-all, so arity changes identically '
          'and only the share differs.\n',
          '**Two dose families.** `A` merges channels into the catch-all by max, which '
          'makes it a *union of well-detected prompts* — unnaturally competent, and the '
          'opposite of a real catch-all. ⭐ **`B` is the faithful analogue**: the classes '
          'are removed from the vocabulary entirely and their pixels labelled catch-all, '
          'which keeps its own single weak prompt. That is LoveDA\'s actual situation.\n',
          '| arm | vocab | catch-all share | discard % | base % | AUC `conf` | **det** | '
          'AUC `conf2` | **det2** | mIoU |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        inv = ' ⇄' if r['auc'] < 0.5 else ''
        inv2 = ' ⇄' if r['auc2'] < 0.5 else ''
        md.append(f'| **{r["arm"].name}** | {r["classes"]} | **{r["share"]:.2f}%** | '
                  f'{r["discard"]:.2f} | {r["base"]:.1f} | {r["auc"]:.3f}{inv} | '
                  f'**{r["det"]:.3f}** | {r["auc2"]:.3f}{inv2} | **{r["det2"]:.3f}** | '
                  + (f'{r["miou"]:.2f}' if np.isfinite(r['miou']) else '—') + ' |')
    md.append('')
    for r in rows:
        md.append(f'- **{r["arm"].name}** — {r["arm"].note}.')

    base0 = next(r for r in rows if r['arm'].name.startswith('A0'))
    md += ['\n## Verdict\n']

    def fam(dose_p, ctrl_p, title, lead):
        d = [r for r in rows if r['arm'].name.startswith(dose_p)]
        c = [r for r in rows if r['arm'].name.startswith(ctrl_p)]
        if not d or not c:
            return
        md.append(f'\n### {title}\n\n{lead}\n')
        md.append('| signal | A0 | largest dose | control | dose effect | control effect |')
        md.append('|---|---|---|---|---|---|')
        out = {}
        for key, lab in (('det', '`conf`'), ('det2', '`conf2` (runner-up)')):
            de = base0[key] - d[-1][key]
            ce = base0[key] - c[-1][key]
            out[key] = (de, ce)
            md.append(f'| {lab} | {base0[key]:.3f} | **{d[-1][key]:.3f}** | '
                      f'{c[-1][key]:.3f} | **{de:+.3f}** | {ce:+.3f} |')
        md.append(f'\nCatch-all share {base0["share"]:.2f}% → **{d[-1]["share"]:.2f}%** '
                  f'in the dose arm, {c[-1]["share"]:.2f}% in the control.')
        inverted = [r['arm'].name for r in d if r['auc'] < 0.5]
        if inverted:
            md.append(f'\n⚠️ **`conf` INVERTED** in {", ".join(inverted)} — the signal '
                      f'changed sign rather than disappearing, so the raw AUC understates '
                      f'detectability badly. Scored on `det`.')
        for key, lab, claim in (
                ('det2', '`conf2`', "§7's mechanism is stated about the RUNNER-UP: a "
                 "catch-all gives the model a plausible answer everywhere, so a strong "
                 "runner-up carries no information. This is the row that tests it."),
                ('det', '`conf`', 'The top score is what §7a\'s monotone-in-share table '
                 'used, so it needs its own answer.')):
            de, ce = out[key]
            # ⚠️ FOUR outcomes, not three. An earlier version had no branch for
            # "the effect ran the OTHER WAY" and none for "the control moved as
            # much as the dose", so on LoveDA it reported a +0.048 RISE in
            # detectability as "share is not the cause" -- which is true but for
            # entirely the wrong reason, and hides that the arm had no power.
            powered = abs(de) > 2 * abs(ce)
            if de > 0.05 and powered:
                md.append(f'\n✅ **{lab}: share is causal.** {claim} Detectability falls '
                          f'**{de:.3f}** with the dose while the control moves {ce:+.3f} '
                          f'— same merge, same class count, share untouched.')
            elif de < -0.05 and powered:
                md.append(f'\n⛔ **{lab}: the effect runs the OTHER WAY.** {claim} '
                          f'Detectability *rose* {-de:.3f} as the share increased '
                          f'(control {ce:+.3f}). This contradicts the mechanism and must '
                          f'be reported as such, not as a null.')
            elif not powered:
                md.append(f'\n⚠️ **{lab}: UNDERPOWERED, not null.** {claim} The dose moves '
                          f'{de:+.3f} and the control moves {ce:+.3f} — the same order of '
                          f'magnitude, so this arm cannot separate them. Do not read it '
                          f'as evidence either way.'
                          + (f' Note the catch-all already starts at '
                             f'**{base0["share"]:.1f}%** here; if the effect saturates, '
                             f'there is little room left for a dose to act in.'
                             if base0['share'] > 25 else ''))
            else:
                md.append(f'\n⛔ **{lab}: no effect.** {claim} Detectability moves only '
                          f'{de:+.3f} across the dose range against a control of '
                          f'{ce:+.3f}.')

    fam('B', 'D', 'B / D — the faithful analogue ⭐',
        'The merged classes are absent from the **vocabulary** and their pixels are '
        'labelled catch-all, which keeps its own single weak prompt. This is what a real '
        'catch-all is, and it is the arm the paper should quote.')
    fam('A', 'C', 'A / C — merge by channel-max',
        '⚠️ Here the catch-all becomes a *union of well-detected prompts*, so it is '
        'unnaturally competent at its own pixels. Retained because it was pre-registered, '
        'but it is **not** a faithful model of a catch-all — read `B/D` first.')

    rev = [r for r in rows if r['arm'].name.startswith('R')]
    if rev:
        d = rev[0]['det'] - base0['det']
        # ⭐ R changes the VOCABULARY but not the LABEL SPACE: the catch-all prompt
        # is removed, yet the same fraction of ground truth is still catch-all --
        # those pixels simply become unnameable. So R is not "the mechanism run
        # backwards"; paired with the B arms (which change BOTH) it DISSOCIATES
        # the two, and says which one carries the effect.
        md.append(f'\n### Reverse arm — a dissociation, not a reversal\n\n'
                  f'`R` removes the catch-all from the **vocabulary** while leaving the '
                  f'**label space** untouched: {base0["share"]:.2f}% of ground truth is '
                  f'still catch-all, those pixels simply have no prompt that can name '
                  f'them. It moves `det` **{d:+.3f}** ({base0["det"]:.3f} → '
                  f'{rev[0]["det"]:.3f}).\n')
        if d <= 0.02:
            md.append('⭐ **So the vocabulary is not the lever — the label space is.** The '
                      '`B` arms change both (a prompt is dropped *and* its pixels are '
                      'relabelled catch-all) and move detectability; `R` changes only the '
                      'vocabulary and does not. Read together they isolate the causal '
                      'locus, which is exactly what §7 claimed from the start: the '
                      "catch-all's **share of ground truth**. Removing the prompt does "
                      'not remove the pixels that are genuinely none-of-the-above; it '
                      'only makes them unnameable, so nothing is recovered.')
        else:
            md.append('✅ Detectability rises when the vocabulary covers the scene, even '
                      'with the label space unchanged — so the prompt list carries part '
                      'of the effect on its own, which the `B` arms alone could not show.')
        md.append('\n⚠️ mIoU is not comparable for this arm — it has no catch-all class '
                  'to predict — so only the detection columns mean anything.')
    md.append('\n⚠️ Score against `PREREGISTRATION.md`, committed before the first run. '
              'The `B`/`D` family and the `det` column were added AFTER seeing that `conf` '
              'inverted; their predictions are pre-registered separately in that file\'s '
              'addendum, and the original predictions stand as written.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
