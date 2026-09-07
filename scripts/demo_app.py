"""
A demo you can click: baseline vs calibrated, on any image, with any vocabulary.

WHY. The contribution is two vectors of numbers, which is invisible. This shows
the same image segmented three ways -- the published rule, our per-class
thresholds, and the thresholds plus the argmax scale -- with the recovered pixels
highlighted. On Potsdam the baseline finds 39% of the trees and ours finds 67%,
and that is a picture rather than a table.

⭐ It also demonstrates the open-vocabulary property, which is the part people do
not believe until they see it: type `solar panel, cricket ground` and the model
segments them, with no retraining and no new labels.

⛔ IT INSTALLS NOTHING. `CLAUDE.md` is emphatic that `segov3` must not be
disturbed -- it is the only working combination of torch/mmcv/mmseg found after
five failures, and Gradio would pull fastapi, pydantic, starlette and more into
it. So:

    gradio importable  ->  launch the interactive app
    gradio absent      ->  render a self-contained HTML page instead

The HTML mode needs only numpy and matplotlib, both already present. It is also
the better artefact for a viva: it survives the GPU being busy, opens on a phone,
and can be sent to an examiner in advance.

⭐ CALIBRATION VECTORS ARE READ FROM THE GENERATED CONFIGS, never hardcoded.
`reorder_deploy.py` writes `class_scale=[...]` and `prob_thd=[...]` into a config;
this parses them back out. A number typed twice is a number that will disagree
with itself eventually, and every table in this project is built to avoid that.

⚠️ A preset only applies to the vocabulary it was fitted for. Type a different
class list and the presets are switched off with a message rather than silently
misapplied -- a threshold vector indexed by the wrong class would still produce a
plausible-looking mask. That is also the paper's own finding on display: the
parameters are dataset-specific and do not transfer.

    cd ~/SegEarth-OV-3            # the vendored sam3/ must win over ~/sam3
    python ~/FreeTraining-OVSS/scripts/demo_app.py \\
        --config configs/cfg_potsdam.py \\
        --preset "Potsdam=configs/cfg_potsdam_reorder.py" \\
        --preset "LoveDA=configs/cfg_loveda_reorder.py" \\
        --images ~/demo_tiles/*.png \\
        --out ~/demo_out
"""
import argparse
import base64
import glob
import re
import sys
from pathlib import Path

import numpy as np

# Palette chosen to stay distinguishable in greyscale print as well as on screen.
PALETTE = np.array([
    [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
], dtype=np.uint8)


# --------------------------------------------------------------------------- #
def parse_vocab(text):
    """One class per line; commas separate SYNONYMS of one class, not classes.

    `building,house` is one class with two prompts -- the convention the
    segmentor's own class files use, and getting it wrong silently doubles the
    class count.
    """
    words, idx, names = [], [], []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        syns = [w.strip() for w in line.split(',') if w.strip()]
        if not syns:
            continue
        names.append(syns[0])
        for w in syns:
            words.append(w)
            idx.append(len(names) - 1)
    return words, idx, names


def read_preset(path):
    """Pull `class_scale` and `prob_thd` back out of a generated config."""
    t = Path(path).expanduser().read_text()
    out = {}
    for key in ('class_scale', 'prob_thd'):
        m = re.search(key + r'\s*=\s*\[([^\]]*)\]', t)
        if m:
            out[key] = [float(x) for x in m.group(1).replace('\n', ' ').split(',')
                        if x.strip()]
    return out


def apply_rule(logits, tau, bg, scale=None):
    """The segmentor's rule, in numpy. Mirrors predict() including the branch:
    the argmax reads SCALED scores, the threshold reads the RAW winning one."""
    if scale is None:
        pred = np.argmax(logits, axis=0)
        conf = logits.max(axis=0)
    else:
        pred = np.argmax(logits * np.asarray(scale)[:, None, None], axis=0)
        conf = np.take_along_axis(logits, pred[None], axis=0)[0]
    thd = np.asarray(tau)[pred] if np.ndim(tau) else float(tau)
    out = pred.copy()
    out[conf < thd] = bg
    return out


def colourise(pred, n):
    return PALETTE[np.clip(pred, 0, min(n, len(PALETTE)) - 1) % len(PALETTE)]


def blend(img, mask_rgb, a=0.55):
    img = np.asarray(img)[..., :3].astype(np.float32)
    if img.shape[:2] != mask_rgb.shape[:2]:
        return mask_rgb
    return (img * (1 - a) + mask_rgb.astype(np.float32) * a).astype(np.uint8)


# --------------------------------------------------------------------------- #
class Engine:
    """Loads SAM 3 once. Changing the vocabulary costs nothing -- the text prompt
    is set per query INSIDE inference, so only `query_words`/`query_idx` change."""

    def __init__(self, config, device='cuda'):
        sys.path.insert(0, str(Path.cwd()))
        import segearthov3_segmentor                      # noqa: F401
        import custom_datasets                            # noqa: F401
        from mmseg.apis import init_model, inference_model
        self._infer = inference_model
        print(f'  loading {config} ...')
        from mmengine.config import Config
        cfg = Config.fromfile(str(config))
        # ⚠️ the segmentor reads classname_path in __init__ and does not keep it,
        # so it must come off the CONFIG. Reading it off the model returns None and
        # used to fall through to a placeholder vocabulary -- a wrong-but-plausible
        # run, which is the failure mode WEEK3_RESULTS §11 documents repeatedly.
        self.classname_path = cfg.get('model', {}).get('classname_path')
        # where the masks live, and how they are encoded. Read from the config --
        # WEEK3_RESULTS SS11 records four dataset-specific assumptions that were
        # hardcoded and survived until a dataset broke them.
        dl = cfg.get('test_dataloader', {}).get('dataset', {})
        self.data_root = dl.get('data_root')
        pre = dl.get('data_prefix', {})
        self.img_prefix, self.ann_prefix = pre.get('img_path'), pre.get('seg_map_path')
        self.reduce_zero = dl.get('reduce_zero_label')   # often absent; see _encoding()
        self._enc = None
        self.model = init_model(str(config), device=device)
        self.bg = int(getattr(self.model, 'bg_idx', 0))
        self.base_tau = float(getattr(self.model, 'prob_thd', 0.5) or 0.5)
        print(f'  ready. bg_idx={self.bg}, published tau={self.base_tau}, '
              f'reduce_zero_label={self.reduce_zero}')

    def gt(self, image_path, n_classes=None):
        """The mask for this tile, as 0-indexed class ids with -1 for ignore.

        Returns None rather than guessing if the mask cannot be read -- a wrong
        mask prints a confident IoU for the wrong answer, which is worse than
        no IoU at all."""
        self.n_gt_max = n_classes or self.model.num_cls
        p = Path(image_path)
        cands = []
        if self.data_root and self.img_prefix and self.ann_prefix:
            root = Path(self.data_root).expanduser()
            rel = p.relative_to(root / self.img_prefix) if str(root / self.img_prefix) in str(p) else Path(p.name)
            for ext in (p.suffix, '.png', '.tif'):
                cands.append(root / self.ann_prefix / rel.with_suffix(ext))
        for ext in (p.suffix, '.png', '.tif'):
            cands.append(Path(str(p.parent).replace('img_dir', 'ann_dir')) / (p.stem + ext))
        for c in cands:
            if not c.exists():
                continue
            # ⛔ NOT matplotlib.imread. Segmentation masks are palette-mode PNGs
            # whose pixel values ARE the class ids; imread expands the palette to
            # RGB, so the array holds display colours and every IoU comes out ~0
            # while still looking like a real number. PIL without .convert() keeps
            # the raw ids.
            from PIL import Image
            im = Image.open(str(c))
            if im.mode not in ('P', 'L', 'I', 'I;16'):
                self._gt_why = (f'{c.name} is mode {im.mode}, not an index mask — '
                                'an RGB-coded mask needs its colour map to decode')
                return None
            g = np.array(im).astype(np.int32)
            if self._enc is None and not self._encoding(c.parent, self.n_gt_max):
                return None
            if self._enc == 'one':
                g = np.where(g == 0, -1, g - 1)
            lo, hi = int(g.min()), int(g.max())
            if hi >= self.n_gt_max or lo < -1:
                self._gt_why = (f'{c.name} holds ids {lo}..{hi} after decoding, '
                                f'outside [-1, {self.n_gt_max - 1}]')
                return None
            return g
        self._gt_why = 'no mask file found beside the image'
        return None

    def _encoding(self, ann_dir, n):
        """0-indexed (0..n-1) or 1-indexed with 0=ignore (0..n)? Decided from the
        MASKS, once, not from the config.

        ⚠️ reduce_zero_label is declared on the dataset class in mmseg, not under
        test_dataloader, so reading it off the config silently returned False for
        Potsdam and shifted every class id by one -- `low vegetation` was scored
        as `tree`. The data answers this unambiguously and cannot go stale."""
        from PIL import Image
        files = sorted(p for p in ann_dir.iterdir()
                       if p.suffix.lower() in ('.png', '.tif', '.tiff'))[:40]
        hi = max((int(np.array(Image.open(str(f))).max()) for f in files),
                 default=-1)
        if hi == n:
            self._enc = 'one'
        elif hi == n - 1:
            self._enc = 'zero'
        else:
            self._gt_why = (f'masks in {ann_dir.name} reach id {hi}; a {n}-class '
                            f'set should reach {n - 1} (0-indexed) or {n} '
                            f'(1-indexed with 0=ignore). Cannot decode.')
            return False
        cfgsays = self.reduce_zero
        print(f'  ground truth: {self._enc}-indexed (max id {hi} over '
              f'{len(files)} masks)'
              + ('' if cfgsays is None else
                 f'; config said reduce_zero_label={cfgsays}'
                 + ('' if bool(cfgsays) == (self._enc == 'one')
                    else '  ⚠️ DISAGREES — trusting the data')))
        return True

    def set_vocab(self, text):
        import torch
        words, idx, names = parse_vocab(text)
        if not names:
            raise ValueError('empty vocabulary')
        self.model.query_words = words
        self.model.query_idx = torch.tensor(idx, dtype=torch.int64,
                                            device=self.model.query_idx.device)
        self.model.num_cls = len(names)
        self.model.num_queries = len(words)
        return names

    def logits(self, image_path):
        r = self._infer(self.model, str(image_path))
        return r.seg_logits.data.float().cpu().numpy()


def iu(gt, pred, n):
    """Per-class intersection and union over valid pixels, as raw counts.

    Counts rather than ratios, so tiles can be POOLED. A mean of per-tile IoUs
    is not the dataset IoU and the two can differ by tens of points."""
    v = gt >= 0
    g, p_ = gt[v], pred[v]
    inter = np.zeros(n, np.int64)
    union = np.zeros(n, np.int64)
    for c in range(n):
        gi, pi = g == c, p_ == c
        inter[c] = int((gi & pi).sum())
        union[c] = int((gi | pi).sum())
    return inter, union


def ratio(inter, union):
    out = np.full(len(inter), np.nan)
    nz = union > 0
    out[nz] = 100.0 * inter[nz] / union[nz]
    return out


def miou(ib, if_, present, keep=None):
    """Mean IoU for two rungs over the SAME class set.

    ⚠️ Averaging each rung over whatever it happens to define is a trap. If the
    calibration removes a false-positive class that is absent from the ground
    truth, that class's IoU goes from 0 to undefined -- and a nanmean then drops
    it from the denominator, handing the method a gain it did not earn. On one
    Potsdam tile that alone was worth +10 mIoU. The class set is the classes
    PRESENT IN GROUND TRUTH, which is what mIoU means, and it is fixed across
    both rungs."""
    idx = [c for c in range(len(ib)) if present[c] and (keep is None or keep(c))]
    if not idx:
        return np.nan, np.nan, idx
    f = lambda v: float(np.mean([0.0 if np.isnan(v[c]) else v[c] for c in idx]))
    return f(ib), f(if_), idx


def iou(gt, pred, n):
    return ratio(*iu(gt, pred, n))


def panels(eng, image_path, vocab_text, preset, tau_override=None):
    """Returns (names, dict of panels, summary markdown)."""
    import matplotlib.image as mpimg
    names = eng.set_vocab(vocab_text)
    stats = {}
    lg = eng.logits(image_path)
    n = len(names)
    img = mpimg.imread(str(image_path))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    tau0 = eng.base_tau if tau_override is None else float(tau_override)
    base = apply_rule(lg, tau0, eng.bg)
    out = {'input': img, 'baseline': blend(img, colourise(base, n))}
    note = []

    eng._gt_why = ''
    g = eng.gt(image_path, n)
    if g is not None and g.shape != base.shape:
        eng._gt_why = f'mask is {g.shape}, prediction is {base.shape}'
        g = None
    if g is not None:
        out['truth'] = blend(img, colourise(np.where(g < 0, eng.bg, g), n))
    else:
        note.append(f'_no usable ground truth ({eng._gt_why}) — panels are '
                    'qualitative only, no IoU is reported._')

    ok = preset and len(preset.get('prob_thd', [])) == n
    if preset and not ok:
        note.append(f'⚠️ preset ignored: it was fitted for '
                    f'{len(preset.get("prob_thd", []))} classes, this vocabulary has '
                    f'{n}. A threshold vector indexed by the wrong class would still '
                    f'produce a plausible mask, so it is switched off rather than '
                    f'misapplied.')
    if ok:
        tau = preset['prob_thd']
        sc = preset.get('class_scale')
        fit = apply_rule(lg, tau, eng.bg, sc)
        out['calibrated'] = blend(img, colourise(fit, n))
        diff = fit != base
        hi = np.zeros_like(out['input'])
        hi[..., 0] = 255
        out['changed'] = np.where(diff[..., None], hi, blend(img, np.zeros_like(hi), 0.0))
        moved = int(diff.sum())
        note.append(f'**{moved:,} pixels ({100 * moved / diff.size:.1f}%) change label.**')
        if g is not None:
            (i0, u0), (i1, u1) = iu(g, base, n), iu(g, fit, n)
            present = np.array([int((g == c).sum()) > 0 for c in range(n)])
            stats.update(i0=i0, u0=u0, i1=i1, u1=u1, present=present.astype(np.int64))
            ib, if_ = ratio(i0, u0), ratio(i1, u1)
            m0, m1, idx = miou(ib, if_, present)
            r0, r1, ridx = miou(ib, if_, present, keep=lambda c: c != eng.bg)
            # ⭐ the direct answer to "is it hallucinating?": of the pixels the
            # calibration MOVED, how many landed on the right class and how many
            # on the wrong one. IoU going up is consistent with adding some wrong
            # pixels too; this is not.
            ch = diff & (g >= 0)
            nch = int(ch.sum())
            if nch:
                won = int(((base[ch] != g[ch]) & (fit[ch] == g[ch])).sum())
                lost = int(((base[ch] == g[ch]) & (fit[ch] != g[ch])).sum())
                neither = nch - won - lost
                note.append(f'Of the {nch:,} changed pixels with a label: '
                            f'**{100 * won / nch:.1f}% became correct**, '
                            f'{100 * lost / nch:.1f}% became wrong, '
                            f'{100 * neither / nch:.1f}% were wrong before and after. '
                            f'Net **{won - lost:+,}** pixels.')
            note.append(f'**tile mIoU {m0:.2f} → {m1:.2f} ({m1 - m0:+.2f})** over '
                        f'the {len(idx)} classes present in the ground truth'
                        + ('' if not ridx else
                           f', excluding `{names[eng.bg]}` {r0:.2f} → {r1:.2f} '
                           f'({r1 - r0:+.2f})'))
            gshare = np.array([100.0 * int((g == c).sum()) / max(int((g >= 0).sum()), 1)
                               for c in range(n)])
            rows = [(names[c], ib[c], if_[c], present[c], gshare[c]) for c in range(n)
                    if not (np.isnan(ib[c]) and np.isnan(if_[c]))]
            rows.sort(key=lambda r: -(-1e9 if np.isnan(r[2] - r[1]) else r[2] - r[1]))
            note.append('| class | % of truth | baseline IoU | calibrated | Δ |')
            note.append('|---|---|---|---|---|')
            for nm, a, b, pr, sh in rows:
                fmt = lambda v: '—' if np.isnan(v) else f'{v:.1f}'
                d = '—' if (np.isnan(a) or np.isnan(b)) else f'{b - a:+.1f}'
                note.append(f'| {nm} | {sh:.1f}% | {fmt(a)} | {fmt(b)} | {d} |')
            if any(not r[3] for r in rows):
                note.append('_&mdash; = in neither the truth nor that prediction, so '
                            'IoU is undefined. Classes at 0.0% of the truth are '
                            'excluded from both means, so removing a false positive '
                            'cannot inflate the gain._')
        for c in range(n):
            d = int((fit == c).sum()) - int((base == c).sum())
            if abs(d) > diff.size * 0.002:
                note.append(f'- `{names[c]}`: {d:+,} px')
    else:
        note.append('_Baseline only — no calibration fitted for this vocabulary._')
    return names, out, '\n'.join(note), stats


# --------------------------------------------------------------------------- #
def render(note):
    """The tiny slice of markdown the notes actually use: **bold**, `code`, and
    pipe tables. Written out rather than pulling in a dependency."""
    html, tbl = [], []

    def flush():
        if not tbl:
            return
        head, body = tbl[0], [r for r in tbl[1:] if set(r.replace('|', '').strip()) - set('-: ')]
        cells = lambda r, t: '<tr>' + ''.join(
            f'<{t}>{c.strip()}</{t}>' for c in r.strip().strip('|').split('|')) + '</tr>'
        html.append('<table>' + cells(head, 'th')
                    + ''.join(cells(r, 'td') for r in body) + '</table>')
        tbl.clear()

    for line in note.split('\n'):
        if line.strip().startswith('|'):
            tbl.append(line)
            continue
        flush()
        html.append(line + '<br>')
    flush()
    out = ''.join(html)
    out = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', out)
    out = re.sub(r'`(.+?)`', r'<code>\1</code>', out)
    out = re.sub(r'_(.+?)_', r'<i>\1</i>', out)
    return out


def write_html(results, names, out_dir, summary=''):
    """Self-contained page. No server, no dependencies, opens on a phone."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib.image as mpimg

    cards = []
    for i, (stem, panels_d, note) in enumerate(results):
        imgs = []
        for key in ('input', 'truth', 'baseline', 'calibrated', 'changed'):
            if key not in panels_d:
                continue
            f = out / f'{stem}_{key}.png'
            mpimg.imsave(f, panels_d[key])
            b64 = base64.b64encode(f.read_bytes()).decode()
            imgs.append(f'<figure><img src="data:image/png;base64,{b64}">'
                        f'<figcaption>{key}</figcaption></figure>')
            f.unlink()
        cards.append(f'<section><h2>{stem}</h2><div class=row>{"".join(imgs)}</div>'
                     f'<div class=note>{render(note)}</div></section>')

    legend = ''.join(
        f'<span class=key><i style="background:rgb({",".join(map(str, PALETTE[c % len(PALETTE)]))})"></i>{n}</span>'
        for c, n in enumerate(names))

    (out / 'index.html').write_text(f"""<!doctype html><meta charset=utf-8>
<title>Calibrating the decision — demo</title><style>
body{{font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px;background:#fafafa;color:#111}}
h1{{margin:0 0 4px}} .sub{{color:#666;margin:0 0 24px}}
section{{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:16px;margin:0 0 20px}}
h2{{margin:0 0 12px;font-size:15px;font-family:ui-monospace,monospace}}
.row{{display:flex;gap:12px;flex-wrap:wrap}}
figure{{margin:0;flex:1 1 220px;min-width:200px}}
img{{width:100%;border-radius:4px;display:block;border:1px solid #ddd}}
figcaption{{font-size:12px;color:#666;margin-top:4px;text-transform:uppercase;letter-spacing:.05em}}
.note{{margin-top:12px;font-size:13px;color:#333;border-top:1px solid #eee;padding-top:10px}}
table{{border-collapse:collapse;margin:8px 0;font-size:12px}}
th,td{{border:1px solid #ddd;padding:3px 9px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{background:#f4f4f4}}
code{{background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:12px}}
.summary{{background:#fff;border:1px solid #e3e3e3;border-radius:8px;
padding:12px 16px;margin:0 0 20px;font-size:14px}}
.legend{{margin:0 0 24px}} .key{{display:inline-flex;align-items:center;gap:5px;margin:0 12px 6px 0;font-size:13px}}
.key i{{width:13px;height:13px;border-radius:3px;display:inline-block;border:1px solid #0002}}
</style>
<h1>Calibrating the decision, not the model</h1>
<p class=sub>baseline vs per-class thresholds and argmax scaling &mdash; same model,
same weights, no retraining</p>
{f'<p class=summary>{summary}</p>' if summary else ''}
<div class=legend>{legend}</div>
{''.join(cards)}""")
    print(f'\n  wrote {out / "index.html"}  —  open it in a browser')


def launch_gradio(eng, presets, default_vocab):
    import gradio as gr

    def run(image, vocab, preset_name, tau):
        pset = presets.get(preset_name)
        _, p, note = panels(eng, image, vocab, pset, tau)
        return (p['baseline'], p.get('calibrated'), p.get('changed'), note)

    with gr.Blocks(title='Calibrating the decision') as ui:
        gr.Markdown('## Calibrating the decision, not the model\n'
                    'Same model, same weights, no retraining. Type any class names.')
        with gr.Row():
            with gr.Column(scale=1):
                im = gr.Image(type='filepath', label='image')
                vb = gr.Textbox(default_vocab, lines=8, label='vocabulary '
                                '(one class per line; commas = synonyms)')
                ps = gr.Dropdown(['none'] + list(presets), value='none',
                                 label='calibration preset')
                tv = gr.Slider(0.0, 1.0, eng.base_tau, step=0.005,
                               label='baseline threshold')
                go = gr.Button('segment', variant='primary')
            with gr.Column(scale=2):
                o1 = gr.Image(label='baseline')
                o2 = gr.Image(label='calibrated (ours)')
                o3 = gr.Image(label='pixels that changed')
                nt = gr.Markdown()
        go.click(run, [im, vb, ps, tv], [o1, o2, o3, nt])
    ui.launch(share=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--preset', action='append', default=[],
                    help='NAME=path/to/generated_config.py, repeatable')
    ap.add_argument('--images', nargs='*', default=[])
    ap.add_argument('--out', default='~/demo_out')
    ap.add_argument('--vocab', help='file with the class list; defaults to the '
                                    "config's own classname_path")
    ap.add_argument('--force-html', action='store_true')
    args = ap.parse_args()

    presets = {}
    for p in args.preset:
        name, _, path = p.partition('=')
        presets[name] = read_preset(path)
        print(f'  preset {name}: '
              + ', '.join(f'{k} x{len(v)}' for k, v in presets[name].items()))

    eng = Engine(args.config)
    vp = args.vocab or eng.classname_path
    if not (vp and Path(vp).expanduser().exists()):
        raise SystemExit(
            f'⛔ cannot find the class list (looked for {vp!r}).\n'
            '   Pass --vocab <file>, one class per line. Refusing to guess: a\n'
            '   substituted vocabulary produces a plausible mask for the wrong\n'
            '   classes and switches every fitted preset off silently.')
    default_vocab = Path(vp).expanduser().read_text()
    print(f'  vocabulary: {vp} '
          f'({len(parse_vocab(default_vocab)[2])} classes: '
          + ', '.join(parse_vocab(default_vocab)[2]) + ')')

    try:
        import gradio  # noqa: F401
        has_gr = True
    except ImportError:
        has_gr = False

    if has_gr and not args.force_html and not args.images:
        print('  gradio found — launching the interactive app')
        return launch_gradio(eng, presets, default_vocab)

    if not has_gr:
        print('  gradio is not installed (and this script will NOT install it into\n'
              '  segov3 — see CLAUDE.md). Rendering the static page instead.')
    files = [f for pat in args.images for f in sorted(glob.glob(str(Path(pat).expanduser())))]
    if not files:
        raise SystemExit('no --images given, and no gradio to run interactively')
    pset = next(iter(presets.values()), None)
    results, names = [], []
    pool = {}
    for f in files:
        print(f'  {Path(f).name}')
        names, p, note, st = panels(eng, f, default_vocab, pset)
        results.append((Path(f).stem, p, note))
        for k, v in st.items():
            pool[k] = pool.get(k, 0) + v
    summary = ''
    if 'u0' in pool:
        b, c = ratio(pool['i0'], pool['u0']), ratio(pool['i1'], pool['u1'])
        pres = pool['present'] > 0
        m0, m1, idx = miou(b, c, pres)
        r0, r1, _ = miou(b, c, pres, keep=lambda i: i != eng.bg)
        summary = (f'Pooled over {len(files)} tiles, {len(idx)} classes present: '
                   f'<b>mIoU {m0:.2f} &rarr; {m1:.2f} ({m1 - m0:+.2f})</b>, excluding '
                   f'<code>{names[eng.bg]}</code> {r0:.2f} &rarr; {r1:.2f} '
                   f'({r1 - r0:+.2f}). '
                   f'<i>Pooled over the tiles shown, not a dataset result &mdash; '
                   f'too few tiles for a stable mIoU. The measured Potsdam figure is '
                   f'57.60 &rarr; 63.27 over 1816 held-out tiles.</i>')
        print('\n  ' + re.sub('<[^>]+>', '', summary))
    write_html(results, names, args.out, summary)


if __name__ == '__main__':
    main()
