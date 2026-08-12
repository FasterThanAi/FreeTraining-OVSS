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
    import numpy as np

    if isinstance(obj, dict):
        print(f"{pad}{name}: dict with {len(obj)} key(s)")
        for k, v in obj.items():
            describe(v, str(k), indent + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__}, len={len(obj)}")
        if obj:
            describe(obj[0], f"{name}[0]", indent + 1)
    elif hasattr(obj, "shape"):
        rng = ""
        try:
            if hasattr(obj, "detach"):
                # numpy has no bfloat16, so go through float32
                t = obj.detach().float().cpu()
                lo, hi = float(t.min()), float(t.max())
            else:
                lo, hi = float(np.min(obj)), float(np.max(obj))
            rng = f"  range=[{lo:.4f}, {hi:.4f}]"
        except Exception:
            pass
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
    ap.add_argument("--raw", action="store_true",
                    help="also dump raw forward_grounding outputs (presence head etc.)")
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
    print(f"prompt     : {args.prompt!r}")

    # NOTE: autocast(bfloat16) is REQUIRED, not an optimisation.
    # sam3/perflib/fused.py::addmm_act hardcodes .to(torch.bfloat16) on the
    # fused fc1 path, so the MLP emits bf16 activations. fc2 is a plain
    # nn.Linear holding fp32 weights, so outside autocast you get
    #   "mat1 and mat2 must have the same dtype, but got BFloat16 and Float"
    # Inside autocast, fc2 casts its weight to bf16 and the matmul matches.
    # Upstream issue: facebookresearch/sam3 #507
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.prompt)

    # ---- 4. introspect -------------------------------------------------
    # set_text_prompt returns the SAME dict it was given - output IS state.
    print("\n" + "=" * 60)
    print("STATE AFTER INFERENCE")
    print("=" * 60)
    describe(output, "state")

    if args.raw:
        # The public API discards the richer head outputs. forward_grounding
        # is where presence_logit_dec / pred_logits / pred_masks actually live -
        # these are what your pipeline needs, not just the filtered masks.
        print("\n" + "=" * 60)
        print("RAW forward_grounding OUTPUTS  (presence head, per-query logits)")
        print("=" * 60)
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            raw = model.forward_grounding(
                backbone_out=state["backbone_out"],
                find_input=processor.find_stage,
                geometric_prompt=state["geometric_prompt"],
                find_target=None,
            )
        describe(raw, "raw")
        if "presence_logit_dec" in raw:
            p = raw["presence_logit_dec"].sigmoid()
            print(f"\n>>> presence score for {args.prompt!r}: {float(p.flatten()[0]):.4f}")

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
