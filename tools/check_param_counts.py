#!/usr/bin/env python3
"""
Count parameter totals for OmniVideo checkpoints without loading full tensors.

Usage:
  python tools/check_param_counts.py --ckpt-root ../Omni-Video/omni_ckpts
"""

import argparse
import os
import torch
from safetensors import safe_open


def count_safetensors(path: str) -> int:
    total = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            shape = f.get_slice(key).get_shape()
            n = 1
            for d in shape:
                n *= d
            total += n
    return total


def count_torch(path: str) -> int | None:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "module", "model_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if isinstance(obj, dict):
        total = 0
        for v in obj.values():
            if torch.is_tensor(v):
                total += v.numel()
        return total
    if torch.is_tensor(obj):
        return obj.numel()
    return None


def fmt(n: int | None) -> str:
    if n is None:
        return "n/a"
    return f"{n:,} ({n/1e9:.3f}B)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="../Omni-Video/omni_ckpts")
    args = ap.parse_args()

    root = args.ckpt_root
    paths = {
        "wan_dit": os.path.join(root, "wan", "wanxiang1_3b", "diffusion_pytorch_model.safetensors"),
        "wan_t5": os.path.join(root, "wan", "wanxiang1_3b", "models_t5_umt5-xxl-enc-bf16.pth"),
        "wan_vae": os.path.join(root, "wan", "wanxiang1_3b", "Wan2.1_VAE.pth"),
        "adapter": os.path.join(root, "adapter", "model.pt"),
        "vision_head": os.path.join(root, "vision_head", "pytorch_model.bin"),
        "transformer": os.path.join(root, "transformer", "model.pt"),
        "ar_llm_shard1": os.path.join(root, "ar_model", "checkpoint", "llm", "model-00001-of-00004.safetensors"),
        "ar_llm_shard2": os.path.join(root, "ar_model", "checkpoint", "llm", "model-00002-of-00004.safetensors"),
        "ar_llm_shard3": os.path.join(root, "ar_model", "checkpoint", "llm", "model-00003-of-00004.safetensors"),
        "ar_llm_shard4": os.path.join(root, "ar_model", "checkpoint", "llm", "model-00004-of-00004.safetensors"),
        "ar_mm_projector": os.path.join(root, "ar_model", "checkpoint", "mm_projector", "model.safetensors"),
        "ar_vision_tower": os.path.join(root, "ar_model", "checkpoint", "vision_tower", "model.safetensors"),
        "ar_vision_head": os.path.join(root, "ar_model", "checkpoint", "vision_head", "pytorch_model.bin"),
    }

    results: dict[str, int | None] = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            results[name] = None
            continue
        if path.endswith(".safetensors"):
            results[name] = count_safetensors(path)
        else:
            results[name] = count_torch(path)

    print("Per-file counts:")
    for k, v in results.items():
        print(f"  {k:16s}: {fmt(v)}")

    wan_total = sum(v for k, v in results.items() if v is not None and k.startswith("wan_"))
    ar_total = sum(v for k, v in results.items() if v is not None and k.startswith("ar_"))
    # NOTE: "transformer" is a full state dict for OmniVideoMixedConditionModel, so don't sum with wan/adapter.
    omnivideo_mixed = results.get("transformer")

    print("\nAggregates:")
    print(f"  wan_total          : {fmt(wan_total)}")
    print(f"  ar_total           : {fmt(ar_total)}")
    print(f"  omnivideo_mixed    : {fmt(omnivideo_mixed)} (use this instead of wan+adapter+vision_head)")

    if omnivideo_mixed is not None:
        generator_total = omnivideo_mixed + (results.get("wan_t5") or 0) + (results.get("wan_vae") or 0)
        print(f"  generator_total    : {fmt(generator_total)} (mixed + T5 + VAE)")

    if omnivideo_mixed is not None:
        full_total = (omnivideo_mixed + (results.get("wan_t5") or 0) + (results.get("wan_vae") or 0) +
                      (results.get("ar_llm_shard1") or 0) + (results.get("ar_llm_shard2") or 0) +
                      (results.get("ar_llm_shard3") or 0) + (results.get("ar_llm_shard4") or 0) +
                      (results.get("ar_mm_projector") or 0) + (results.get("ar_vision_tower") or 0) +
                      (results.get("ar_vision_head") or 0))
        print(f"  full_pipeline_total: {fmt(full_total)} (generator + AR)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
