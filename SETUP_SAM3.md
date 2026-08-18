# Environment Setup — SAM 3 + MMSegmentation

> **This file previously described SAM 1 + CLIP** (`pip install segment-anything`,
> `sam_vit_h_4b8939.pth`). That was from the original project sketch and is **wrong** for
> this project. SAM 3 replaces the SAM + CLIP two-stage design entirely: it takes text
> prompts natively and exposes a presence head and a semantic head that the method depends on.

Verified working on: **HP Z4 G5 · NVIDIA RTX 2000 Ada (16 GB) · Ubuntu · CUDA driver 13.3**

---

## 1. Why the versions are pinned

Three constraints intersect at exactly one workable point. This is not preference — every
other combination fails.

| Component | Constraint | Evidence |
|---|---|---|
| SAM 3 | **torch ≥ 2.3** | `vl_combiner.py` imports `torch.nn.attention`, added in torch 2.3 |
| mmcv prebuilt wheels | torch 2.1–2.4 only | no `.whl` published for torch 2.5 |
| mmsegmentation 1.2.2 | `mmcv >= 2.0.0rc4, < 2.2.0` | assert in `mmseg/__init__.py` |

**torch 2.4.1** is the only version that satisfies SAM 3 *and* has a prebuilt mmcv wheel.
That wheel is mmcv **2.2.0**, excluded by mmseg's upper bound by one patch version — hence
the `MMCV_MAX` patch below. The ops mmseg actually calls are unchanged between 2.1 and 2.2.

### Approaches that do not work

| Attempt | Failure |
|---|---|
| Python 3.13 | No mmcv wheels exist for 3.13 |
| `mmcv-lite` | Ships no compiled ops. Every mmseg version imports `mmcv.ops` at package import (`focal_loss`, `mask_classification`) → `ModuleNotFoundError: mmcv._ext`. Installing mmcv *and* mmcv-lite together also breaks |
| mmseg 1.0.0 + mmcv-lite | Same — fails at `losses/focal_loss.py` instead |
| Building mmcv from source | System nvcc is CUDA **13.3**, torch is built against **12.1**; torch refuses to compile extensions across that gap. `conda install cuda-toolkit` does **not** provide `nvcc` (that is `cuda-nvcc`) |
| torch 2.1.2 + prebuilt mmcv | mmcv installs fine, but SAM 3 fails: `No module named 'torch.nn.attention'` |

---

## 2. Install

```bash
conda create -n segov3 python=3.11 -y
conda activate segov3

# torch first — everything else compiles/links against it
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"          # mmcv 2.x predates the numpy 2.0 ABI break

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.4.1+cu121 True

# mmcv — PREBUILT WHEEL, do not build from source
pip install mmcv==2.2.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install "mmsegmentation==1.2.2"

# raise mmseg's mmcv upper bound
sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/" \
  $CONDA_PREFIX/lib/python3.11/site-packages/mmseg/__init__.py
grep MMCV_MAX $CONDA_PREFIX/lib/python3.11/site-packages/mmseg/__init__.py

# SAM 3 runtime deps (incl. the video stack — see note below)
pip install einops psutil pycocotools hydra-core iopath timm huggingface_hub omegaconf

# verify the whole stack
python -c "import torch, mmcv, mmseg; from mmcv.ops import point_sample; \
  print('ok', torch.__version__, mmcv.__version__, mmseg.__version__)"
```

⚠️ **The `MMCV_MAX` patch edits site-packages.** It will not survive an environment rebuild
and is not captured by `pip freeze`. It is included in `scripts/setup_env.sh` for that reason.

---

## 3. SAM 3 checkpoint

```bash
huggingface-cli download facebook/sam3
# or download sam3.pt manually from https://huggingface.co/facebook/sam3

mkdir -p weights/sam3
ln -s "$(ls ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt)" \
  weights/sam3/sam3.pt
ls -lL weights/sam3/sam3.pt      # ~3,450,062,241 bytes (3.45 GB)
```

Use a symlink, not a copy — the file is 3.45 GB. The reference implementation hardcodes
`weights/sam3/sam3.pt` relative to the repo root.

Add to `.gitignore`: `*.pt`, `*.pth`, `weights/`.

---

## 4. Reference implementation

```bash
git clone https://github.com/earth-insights/SegEarth-OV-3.git
cd SegEarth-OV-3
python demo.py                    # writes seg_pred.png
```

**Note:** SegEarth-OV-3 ships a **vendored `sam3/`** directory that shadows any separate
SAM 3 install when running from inside the repo. Edits to an external `~/sam3` clone have
no effect there. This is fine — the vendored copy is the version the authors tested — but
be aware of which copy is live.

---

## 5. Known quirks

**SAM 3 pulls in the video stack.** `sam3/model_builder.py` imports
`sam3_video_predictor` / `sam3_video_inference` at module level, so `psutil` and
`pycocotools` are required even for image-only work.

**mmcv 2.1.0 source builds need extra flags.** Its `setup.py` imports `pkg_resources`,
removed from modern setuptools. If you ever must build from source:
```bash
pip install "setuptools<70" wheel ninja
pip install mmcv==2.1.0 --no-binary mmcv --no-build-isolation
```
This still requires nvcc to match torch's CUDA version.

**Order matters.** Install torch first. mmcv and SAM 3 both link against the installed
torch; installing them first, or letting a later `pip install` silently upgrade torch, breaks
the environment. After any install that touches torch, re-check:
```bash
python -c "import torch; print(torch.__version__)"   # must stay 2.4.1+cu121
```

---

## 6. Verified working stack

| Package | Version |
|---|---|
| Python | 3.11.15 (conda-forge) |
| torch | 2.4.1+cu121 |
| torchvision | 0.19.1+cu121 |
| mmcv | 2.2.0 (prebuilt, torch2.4/cu121) |
| mmsegmentation | 1.2.2 (`MMCV_MAX` → 2.3.0) |
| mmengine | 0.10.7 |
| numpy | 1.26.4 |
| opencv-python | 4.10.0.84 |
| SAM 3 checkpoint | `sam3.pt`, 3.45 GB |

Measured on LoveDA val (1669 images): **mIoU 47.38**, 0.85 s/image, **peak memory 6115 MB**.
