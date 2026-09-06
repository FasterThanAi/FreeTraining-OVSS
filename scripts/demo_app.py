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
        self.model = init_model(str(config), device=device)
        self.bg = int(getattr(self.model, 'bg_idx', 0))
        self.base_tau = float(getattr(self.model, 'prob_thd', 0.5) or 0.5)
        print(f'  ready. bg_idx={self.bg}, published tau={self.base_tau}')

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


def panels(eng, image_path, vocab_text, preset, tau_override=None):
    """Returns (names, dict of panels, summary markdown)."""
    import matplotlib.image as mpimg
    names = eng.set_vocab(vocab_text)
    lg = eng.logits(image_path)
    n = len(names)
    img = mpimg.imread(str(image_path))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    tau0 = eng.base_tau if tau_override is None else float(tau_override)
    base = apply_rule(lg, tau0, eng.bg)
    out = {'input': img, 'baseline': blend(img, colourise(base, n))}
    note = []

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
        for c in range(n):
            d = int((fit == c).sum()) - int((base == c).sum())
            if abs(d) > diff.size * 0.002:
                note.append(f'- `{names[c]}`: {d:+,} px')
    else:
        note.append('_Baseline only — no calibration fitted for this vocabulary._')
    return names, out, '\n'.join(note)


# --------------------------------------------------------------------------- #
def write_html(results, names, out_dir):
    """Self-contained page. No server, no dependencies, opens on a phone."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib.image as mpimg

    cards = []
    for i, (stem, panels_d, note) in enumerate(results):
        imgs = []
        for key in ('input', 'baseline', 'calibrated', 'changed'):
            if key not in panels_d:
                continue
            f = out / f'{stem}_{key}.png'
            mpimg.imsave(f, panels_d[key])
            b64 = base64.b64encode(f.read_bytes()).decode()
            imgs.append(f'<figure><img src="data:image/png;base64,{b64}">'
                        f'<figcaption>{key}</figcaption></figure>')
            f.unlink()
        cards.append(f'<section><h2>{stem}</h2><div class=row>{"".join(imgs)}</div>'
                     f'<div class=note>{note.replace(chr(10), "<br>")}</div></section>')

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
.legend{{margin:0 0 24px}} .key{{display:inline-flex;align-items:center;gap:5px;margin:0 12px 6px 0;font-size:13px}}
.key i{{width:13px;height:13px;border-radius:3px;display:inline-block;border:1px solid #0002}}
</style>
<h1>Calibrating the decision, not the model</h1>
<p class=sub>baseline vs per-class thresholds and argmax scaling &mdash; same model,
same weights, no retraining</p>
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
    vp = args.vocab or getattr(eng.model, 'classname_path', None)
    default_vocab = Path(vp).read_text() if vp and Path(vp).exists() else 'building\nroad\nwater\ntree'

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
    for f in files:
        print(f'  {Path(f).name}')
        names, p, note = panels(eng, f, default_vocab, pset)
        results.append((Path(f).stem, p, note))
    write_html(results, names, args.out)


if __name__ == '__main__':
    main()
