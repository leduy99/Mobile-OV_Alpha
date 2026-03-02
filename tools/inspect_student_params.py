#!/usr/bin/env python3
"""Inspect trainable params for student (bridge) without loading SANA DiT."""
import argparse
from collections import defaultdict
import yaml
import torch

from nets.omni.modules.sana_prompt_bridge import SanaPromptBridge


def to_attr(d):
    if isinstance(d, dict):
        return {k: to_attr(v) for k, v in d.items()}
    if isinstance(d, list):
        return [to_attr(x) for x in d]
    return d


def human(n):
    for unit in ["", "K", "M", "B", "T"]:
        if n < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = to_attr(yaml.safe_load(f))

    student_cfg = cfg["model"]["student"]
    bridge_cfg = student_cfg["conditioner_bridge"]

    device = torch.device(args.device)
    student = SanaPromptBridge(
        smolvlm2_ckpt_path=student_cfg["text_encoder"]["ckpt_path"],
        adapter_ckpt_dir=student_cfg.get("adapter_ckpt_dir"),
        adapter_in_channels=student_cfg.get("adapter_in_channels", 1024),
        adapter_out_channels=student_cfg.get("adapter_out_channels", 2304),
        adapter_query_length=student_cfg.get("adapter_query_length", 64),
        adapter_num_encoder_layers=student_cfg.get("adapter_num_encoder_layers", 2),
        adapter_num_decoder_layers=student_cfg.get("adapter_num_decoder_layers", 2),
        adapter_ff_mult=student_cfg.get("adapter_ff_mult", 2),
        smol_vh_num_queries=student_cfg.get("smol_vh_num_queries", 1),
        num_prompt_queries=bridge_cfg["out_seq_len"],
        caption_channels=bridge_cfg["out_dim"],
        precision_dtype=torch.float32,
        device=device,
        tokenizer_model_id=student_cfg["text_encoder"].get("tokenizer_model_id", "HuggingFaceTB/SmolVLM-Instruct"),
        force_adapter_query_length=student_cfg.get("force_adapter_query_length"),
        max_length=student_cfg["text_encoder"]["max_length"],
        use_vision_head=student_cfg.get("use_vision_head", False),
        resampler_num_heads=student_cfg.get("resampler_num_heads", 8),
        resampler_mlp_mult=student_cfg.get("resampler_mlp_mult", 2),
        lora_enable=student_cfg.get("lora", {}).get("enable", False),
        lora_r=student_cfg.get("lora", {}).get("r", 8),
        lora_alpha=student_cfg.get("lora", {}).get("alpha", 16),
        lora_dropout=student_cfg.get("lora", {}).get("dropout", 0.05),
        lora_target_modules=student_cfg.get("lora", {}).get("target_modules"),
    )

    trainable = [(n, p.numel(), tuple(p.shape)) for n, p in student.named_parameters() if p.requires_grad]
    total = sum(n for _, n, _ in trainable)
    print(f"Student trainable params: {total} ({human(total)})")

    group = defaultdict(int)
    for name, n, _ in trainable:
        key = name.split(".")[0]
        group[key] += n
    print("\nTop-level groups:")
    for k, v in sorted(group.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:24s} {human(v)}")

    print(f"\nTop {args.topk} tensors:")
    for name, n, shape in sorted(trainable, key=lambda x: x[1], reverse=True)[: args.topk]:
        print(f"  {name:60s} {human(n):>8s} {shape}")


if __name__ == "__main__":
    main()
