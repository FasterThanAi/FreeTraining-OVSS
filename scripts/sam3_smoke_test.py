"""
SAM 3 smoke test + output introspection.

Two jobs:
  1. Prove SAM 3 runs end-to-end on this machine.
  2. Print the exact structure of what it returns, so you know the tensors
     your pipeline will operate on. This is Phase 0.3 of ROADMAP.md and it
     is the single most useful hour of week one.

Usage:
    python scripts/sam3_smoke_test.py --image path/to/tile.jpg --prompt road
    python scripts/sam3_smoke_test.py --image tile.jpg --prompt road --save-vis out.png
"""

import argparse
import sys


def describe(obj, name="output", indent=0):
    """Recursively print structure/shape/dtype/range of whatever SAM 3 returns."""
    pad = "  " * indent
    try:
        import torch
        import numpy as np
    except ImportError:
        print("torch/numpy missing"); return

    if isinstance(obj, dict):
        print(f"{pad}{name}: dict with {len(obj)} key(s)")
        for k, v in obj.items():
            describe(v, str(k), indent + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__}, len={len(obj)}")
        if obj:
            describe(obj[0], f"{name}[0]", indent + 1)
    elif hasattr(obj, "shape"):
        t = obj.detach().cpu() if hasattr(obj, "detach") else obj
        arr = t.numpy() if hasattr(t, "numpy") else t
        try:
            lo, hi = float(np.min(arr)), float(np.max(arr))
            rng = f"  range=[{lo:.4f}, {hi:.4f}]"
        except Exception:
            rng = ""
        print(f"{pad}{name}: shape={tuple(obj.shape)}  dtype={obj.dtype}{rng}")
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        print(f"{pad}{name}: {type(obj).__name__} = {obj}")
    else:
        attrs = [a for a in dir(obj) if not a.startswith("_")][:20]
        print(f"{pad}{name}: {type(obj).__name__}   attrs: {attrs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="building")
    ap.add_argument("--save-vis", default=None, help="write a mask overlay PNG here")
    args = ap.parse_args()

    # ---- 1. environment ------------------------------------------------
    import torch
    print("=" * 60)
    print("torch      :", torch.__version__)
    print("torch cuda :", torch.version.cuda)
    print("cuda avail :", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("\nGPU not visible to torch. Fix this before continuing.")
        sys.exit(1)
    print("device     :", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"vram       : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    print("=" * 60)

    # ---- 2. load model -------------------------------------------------
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("\nloading model (first run downloads weights)...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("model loaded.")

    # ---- 3. run --------------------------------------------------------
    image = Image.open(args.image).convert("RGB")
    print(f"image      : {args.image}  size={image.size}")
    state = processor.set_image(image)
    print(f"prompt     : {args.prompt!r}")
    output = processor.set_text_prompt(state=state, prompt=args.prompt)

    # ---- 4. introspect -------------------------------------------------
    print("\n" + "=" * 60)
    print("OUTPUT STRUCTURE")
    print("=" * 60)
    describe(output, "output")

    print("\n" + "=" * 60)
    print("INFERENCE STATE  (where P_sem / presence score may live)")
    print("=" * 60)
    describe(state, "state")

    n = len(output["masks"]) if "masks" in output else 0
    print(f"\n>>> {n} instance mask(s) returned for {args.prompt!r}")
    if "scores" in output and n:
        s = output["scores"]
        print(f">>> score range: {float(s.min()):.4f} .. {float(s.max()):.4f}")

    # ---- 5. optional overlay -------------------------------------------
    if args.save_vis and n:
        import numpy as np
        masks = output["masks"]
        m = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        m = (m > 0.5) if m.dtype != bool else m
        union = np.any(m.reshape(m.shape[0], *m.shape[-2:]), axis=0)
        base = np.array(image).astype(float)
        base[union] = 0.5 * base[union] + 0.5 * np.array([255, 60, 60])
        Image.fromarray(base.clip(0, 255).astype("uint8")).save(args.save_vis)
        print(f">>> overlay written to {args.save_vis}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
