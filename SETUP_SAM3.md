# Setting up SAM 3 on a Linux lab PC

---

## Part 1 — Is SAM 3 actually needed?

**Yes. It is not optional for this project.** Three reasons, in order of importance:

### 1. Your method structurally depends on SAM 3's decoupled heads

Your Step 1 splits the image into *identified* and *unidentified* regions using per-class confidence. Your Step 4 needs per-class probability maps to score candidate labels. SAM 3 gives you both directly:

| Head | Output | Which of your steps needs it |
|---|---|---|
| Presence head | `S_pres` — scalar, "does this concept exist in the image?" | Step 1 — kills vocabulary hallucination before you threshold |
| Semantic head | `P_sem` — dense H×W map | Steps 1, 5 — coverage of amorphous "stuff" (road, bareland) |
| Transformer decoder | `{(P_inst, s_conf)}` — instance queries | Steps 1, 3 — sharp boundaries on "things"; also your patch proposals |

**SAM 1 + CLIP — what your repo currently implements — has none of these.** SAM 1 is class-agnostic: it segments, then you crop each mask and ask CLIP what it is. That is slower, and much weaker, because CLIP sees a cropped box stripped of the surrounding context that makes aerial imagery interpretable in the first place.

### 2. Your baseline is SAM 3, so you must run SAM 3

SegEarth-OV3 *is* a SAM 3 pipeline. If you build on SAM 1 + CLIP and report a lower number, you have learned nothing about your own contribution — the gap is your backbone, not your method. **Same backbone, one variable changed.** That's the whole experiment.

### 3. It is the least risky part of the project

Setup is a day. Everything downstream depends on it. Do it in week 1.

### What you don't need it for (yet)

- **Building intuition** — the free web demo at [segment-anything.com](https://segment-anything.com/) needs zero setup. Upload an aerial tile, type "road", watch it fragment. Do this today while checkpoint access is pending.
- **The co-occurrence module itself** — that logic is backbone-agnostic and operates on cached patch records. If setup gets blocked on an admin ticket, keep moving on Steps 3–4 against dummy data.

---

## Part 2 — Run the readiness check first

```bash
cd ~/FreeTraining-OVSS   # or wherever this repo lives
bash check_env.sh
```

It's read-only and installs nothing. It reports GPU, driver CUDA version, Python, conda, disk, sudo, network and build tools. **Paste the output back to me and I'll tell you exactly which path to take.**

### The one thing that can genuinely block you

SAM 3 requires **CUDA 12.6 or higher**. The ceiling is set by the *NVIDIA driver*, and on a shared lab PC you cannot upgrade the driver without root.

```bash
nvidia-smi        # look at the "CUDA Version:" field, top-right
```

| Reported | Verdict |
|---|---|
| **12.6+** | Fine. Proceed. |
| **12.0 – 12.5** | Probably fine — CUDA minor-version compatibility means cu128 wheels usually run. Below spec, so test early. |
| **< 12.0** | **Blocker.** Email your lab admin today asking for a driver upgrade. This has a multi-day lead time — start it now, work on the web demo and Phase 1 theory meanwhile. |
| **no `nvidia-smi`** | No NVIDIA GPU or no driver. Find out which before planning anything. |

VRAM: the model is 848M params. Image inference is comfortable at 16 GB, workable at 8–12 GB with FP16 and tiling. Below 8 GB, tile aggressively and expect pain.

---

## Part 3 — Installation

Everything below runs **entirely in your home directory. No root required at any point.**

### Step 1 — Request checkpoint access (do this first, it has latency)

Checkpoints are gated. Approval is manual and not instant.

1. Create a HuggingFace account
2. Visit [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and accept the license
3. Also request [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1) — the improved SAM 3.1 checkpoints (released 27 Mar 2026). Use these if starting fresh.
4. Generate an access token: Settings → Access Tokens → New token (read scope)

Mirror if HF is blocked on your network: [ModelScope](https://modelscope.cn/models/facebook/sam3).

### Step 2 — Miniconda, if `check_env.sh` says you don't have it

```bash
cd ~ && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec bash            # reload shell
```

`-b` accepts the licence and installs unattended to `$HOME`. Nothing touches system paths.

> **If your home directory is small or quota'd** (very common on lab machines), install to scratch instead: `-p /scratch/$USER/miniconda3`.

### Step 3 — Environment

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
```

Python **3.12+** is a hard requirement — this is why conda is worth it even if the system has Python 3.10.

### Step 4 — PyTorch

```bash
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify before going further — do not skip this:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True` and your GPU name. If it prints `False`, stop: it's a driver/CUDA mismatch, and nothing downstream will work.

### Step 5 — SAM 3

```bash
cd ~ && git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[notebooks]"
```

`-e` is an editable install — you'll be reading and modifying this source constantly, so you want it.

### Step 6 — Authenticate and pull weights

```bash
hf auth login          # paste your token
```

**Redirect the HF cache if home is tight** — checkpoints are several GB and the default is `~/.cache/huggingface`:

```bash
echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc && source ~/.bashrc
```

### Step 7 — Smoke test

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()          # downloads weights on first call
processor = Sam3Processor(model)

image = Image.open("aerial_tile.jpg")
state = processor.set_image(image)
out = processor.set_text_prompt(state=state, prompt="building")

masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
print(masks.shape, scores)
```

If that prints mask shapes, you're done. Then run the official notebook:

```bash
jupyter notebook examples/sam3_image_predictor_example.ipynb
```

---

## Part 4 — Lab-PC gotchas

**Shared GPU.** Check `nvidia-smi` before launching anything heavy — someone's training run and your inference will OOM each other. Pin yourself to one device:

```bash
export CUDA_VISIBLE_DEVICES=0
```

**Home directory quotas.** The single most common lab failure. Conda env (~15 GB) + checkpoints (~5 GB) + datasets (20 GB+) will blow a 20 GB quota. Put conda, `HF_HOME`, and datasets on scratch from the start — migrating later is miserable.

**Your session dies when SSH drops.** Use `tmux` for anything long:

```bash
tmux new -s sam3     # detach: Ctrl-b then d      reattach: tmux attach -t sam3
```

**University proxies.** If `check_env.sh` shows blocked domains:

```bash
export https_proxy=http://proxy.your-uni.edu:8080
export http_proxy=$https_proxy
```

Add to `~/.bashrc`, and note pip may also need `--proxy`.

**Scratch disks get purged.** Many are wiped after 30–90 days of inactivity. Keep code and results in home or git; keep only regenerable artefacts (caches, checkpoints, extracted datasets) on scratch.

**Missing compilers.** If `pip install -e .` fails on a build step and you have no root:

```bash
conda install -c conda-forge gcc_linux-64 gxx_linux-64 make cmake -y
```

**Don't reuse the `venv/` in this repo.** It's a macOS venv built for the old SAM 1 + CLIP pipeline. It's already gitignored — leave it alone and build fresh.

---

## Troubleshooting

### `nvidia-smi has failed because it couldn't communicate with the NVIDIA driver`

If `diagnose_gpu.sh` shows the DKMS module **is** built for your running kernel and the `.ko` files **are** present in `/lib/modules/$(uname -r)/updates/dkms/`, but the module is **not loaded** — and Secure Boot is **enabled** — then Secure Boot is rejecting the module. The driver is fine; the kernel is refusing to load an unsigned module.

Confirm it:

```bash
sudo modprobe nvidia
# "Key was rejected by service"  ->  Secure Boot confirmed
```

Fix by enrolling Ubuntu's Machine Owner Key (keeps Secure Boot on — the correct fix):

```bash
mokutil --list-enrolled | grep -ci 'Subject'          # is a key already enrolled?
ls -l /var/lib/shim-signed/mok/                       # does the key exist?
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der
# set a simple one-time password - you'll type it at boot on a US layout
sudo reboot
```

On reboot a blue **MOK Manager** screen appears: `Enroll MOK` → `Continue` → `Yes` → password → `Reboot`.

> The MOK screen times out in ~10 seconds and only appears once. Don't walk away from the machine. If you miss it, re-run `mokutil --import` and reboot again.

Then verify with `nvidia-smi`.

**Booting an older kernel does not help here** — Secure Boot blocks the module on every kernel. Only enrolment (or disabling Secure Boot in BIOS) fixes it.

If `/var/lib/shim-signed/mok/MOK.der` doesn't exist, regenerate it by reinstalling the DKMS package:

```bash
sudo dpkg-reconfigure nvidia-dkms-580
```

### Multiple driver versions installed

`dpkg -l | grep nvidia-driver` showing several versions (e.g. 535, 550 and 580 together) is an unstable state and a common source of `Driver/library version mismatch`. Once the GPU is working, clean up — **one change at a time, and not before it works**:

```bash
sudo apt remove --purge nvidia-driver-535 nvidia-driver-550 nvidia-utils-550
sudo apt autoremove
```

### Other issues

| Symptom | Cause / fix |
|---|---|
| `torch.cuda.is_available()` is `False` | Driver/CUDA mismatch. Compare `nvidia-smi` CUDA version against your wheel's cu128. Reinstall torch for a lower CUDA if needed. |
| `401` / `403` downloading checkpoints | Access not yet granted, or not logged in. Check the HF repo page, re-run `hf auth login`. |
| `CUDA out of memory` | Reduce tile size; run FP16 (`torch.autocast`); confirm nobody else is on the GPU. |
| `No space left on device` | Quota. Move conda + `HF_HOME` to scratch (Part 4). |
| `ModuleNotFoundError: sam3` | Environment not activated, or `pip install -e .` was run from the wrong directory. |
| Install hangs at "Solving environment" | Use `mamba`, or `pip` inside the conda env rather than `conda install`. |

---

## Sources

- [SAM 3 official repo + README](https://github.com/facebookresearch/sam3) — install requirements and API quoted above
- [SAM 3 checkpoints (gated)](https://huggingface.co/facebook/sam3) · [SAM 3.1 checkpoints](https://huggingface.co/facebook/sam3.1)
- [SAM 3 paper — arXiv:2511.16719](https://arxiv.org/abs/2511.16719)
- [Web demo — segment-anything.com](https://segment-anything.com/)
