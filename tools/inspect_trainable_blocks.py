#!/usr/bin/env python3
"""
Inspect trainable blocks and parameter shapes for Stage-1 teacher-free training.
"""
import argparse
import os
from collections import defaultdict

import torch

from tools.train_stage1_teacher_free import (
    AttrDict,
    load_sana_config,
    load_sana_diffusion_model,
    build_student,
)


def human_count(n: int) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if n < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", default="omni_ckpts/sana_video_2b_480p")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--dump-tree", action="store_true")
    ap.add_argument("--skip-sana", action="store_true", help="Skip loading SANA DiT (faster param count for student)")
    args = ap.parse_args()

    def _to_attr(d):
        if isinstance(d, dict):
            return AttrDict({k: _to_attr(v) for k, v in d.items()})
        if isinstance(d, list):
            return [_to_attr(x) for x in d]
        return d

    import yaml
    with open(args.config, "r") as f:
        cfg = _to_attr(yaml.safe_load(f))

    device = torch.device(args.device)
    sana_cfg = load_sana_config(cfg.sana.config)
    diffusion_model = None
    if not args.skip_sana:
        diffusion_model = load_sana_diffusion_model(sana_cfg, args.checkpoint_dir, device, torch.float32)

    student = build_student(cfg, device, torch.float32)

    # Collect trainable params
    trainable = []
    if diffusion_model is not None:
        for name, p in diffusion_model.named_parameters():
            if p.requires_grad:
                trainable.append((f"dit.{name}", p.numel(), tuple(p.shape)))
    for name, p in student.named_parameters():
        if p.requires_grad:
            trainable.append((f"student.{name}", p.numel(), tuple(p.shape)))

    total = sum(n for _, n, _ in trainable)
    print(f"Trainable params total: {total} ({human_count(total)})")

    # Group by top-level module
    group = defaultdict(int)
    for name, n, _ in trainable:
        prefix = name.split(".")[1] if "." in name else name
        group[prefix] += n
    print("\nTop-level trainable groups:")
    for k, v in sorted(group.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:24s} {human_count(v)}")

    print(f"\nTop {args.topk} trainable tensors:")
    for name, n, shape in sorted(trainable, key=lambda x: x[1], reverse=True)[: args.topk]:
        print(f"  {name:60s} {human_count(n):>8s} {shape}")

    if args.dump_tree:
        print("\nTrainable module tree:")
        if diffusion_model is not None:
            for name, module in diffusion_model.named_modules():
                if any(p.requires_grad for p in module.parameters(recurse=False)):
                    print(f"  dit.{name} -> {module.__class__.__name__}")
        for name, module in student.named_modules():
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                print(f"  student.{name} -> {module.__class__.__name__}")


if __name__ == "__main__":
    main()
