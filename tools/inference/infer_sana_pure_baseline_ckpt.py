#!/usr/bin/env python3
"""
Infer SANA video from pure-baseline checkpoint that stores only trainable LoRA weights.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import yaml

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.train_stage1_teacher_free import apply_lora_to_module
from tools.inference import sana_video_inference_fixed as svi


def _to_dtype(name: str) -> torch.dtype:
    name = str(name).lower().strip()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_trainable_state_to_model(model: torch.nn.Module, trainable_state: Dict[str, torch.Tensor]):
    named_params = dict(model.named_parameters())
    loaded = 0
    missing = []
    mismatched = []
    for key, tensor in trainable_state.items():
        p = named_params.get(key)
        if p is None:
            missing.append(key)
            continue
        if tuple(p.shape) != tuple(tensor.shape):
            mismatched.append((key, tuple(p.shape), tuple(tensor.shape)))
            continue
        p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))
        loaded += 1
    return loaded, missing, mismatched


def parse_args():
    parser = argparse.ArgumentParser(description="Infer from pure SANA baseline LoRA checkpoint")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint_stepXXX.pt produced by train_sana_msrvtt_pure_baseline.py",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/sana_msrvtt_pure_baseline.yaml",
        help="Pure baseline train config (for LoRA config)",
    )
    parser.add_argument(
        "--sana-config",
        type=str,
        default="configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="omni_ckpts/sana_video_2b_480p")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sampling-algo", type=str, default="flow_euler")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="output/infer_sana_pure_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_cfg = _load_yaml(args.train_config)
    sana_cfg = svi.load_config_file(args.sana_config)

    model_dtype = _to_dtype(getattr(getattr(sana_cfg, "model", {}), "mixed_precision", "bf16"))
    vae_dtype = svi.get_weight_dtype(sana_cfg.vae.weight_dtype)
    latent_size = int(args.height) // int(getattr(sana_cfg.vae, "vae_downsample_rate", 8))
    svi.set_env(args.seed, latent_size)

    models = svi.load_sana_models(
        config=sana_cfg,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        model_dtype=model_dtype,
        vae_dtype=vae_dtype,
        latent_size=latent_size,
    )

    # Inject LoRA wrappers to match training graph.
    lora_cfg = (
        train_cfg.get("model", {})
        .get("dit_lora", {})
    )
    if bool(lora_cfg.get("enable", False)):
        replaced = apply_lora_to_module(
            models["diffusion_model"],
            target_modules=lora_cfg.get("target_modules", ["q_linear", "kv_linear", "proj"]),
            r=int(lora_cfg.get("r", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            include_patterns=lora_cfg.get("include_patterns", ["cross_attn"]),
            exclude_patterns=lora_cfg.get("exclude_patterns", []),
        )
        print(f"[INFO] Applied LoRA wrappers to DiT: replaced_linear_layers={replaced}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    trainable_state = ckpt.get("trainable_state", {})
    loaded, missing, mismatched = _load_trainable_state_to_model(models["diffusion_model"], trainable_state)
    print(f"[INFO] Loaded trainable params: {loaded}")
    if missing:
        print(f"[WARN] Missing params in model: {len(missing)}")
    if mismatched:
        print(f"[WARN] Mismatched shapes: {len(mismatched)}")

    video = svi.generate_video(
        models=models,
        prompt=args.prompt,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        device=args.device,
        dtype=model_dtype,
        sampling_algo=args.sampling_algo,
        negative_prompt=args.negative_prompt,
        motion_score=10,
        high_motion=False,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = args.prompt[:60].replace(" ", "_").replace("/", "_")
    out_video = os.path.join(args.output_dir, f"sana_pure_ckpt_{Path(args.ckpt).stem}_{ts}_{slug}.mp4")
    svi.save_video(video, out_video, fps=16)

    out_meta = os.path.join(args.output_dir, f"sana_pure_ckpt_{Path(args.ckpt).stem}_{ts}.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt": args.prompt,
                "ckpt": args.ckpt,
                "loaded_params": loaded,
                "missing_params": len(missing),
                "mismatched_params": len(mismatched),
                "video_path": out_video,
                "seed": args.seed,
                "steps": args.steps,
                "cfg": args.cfg,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Saved video: {out_video}")
    print(f"[INFO] Saved meta: {out_meta}")


if __name__ == "__main__":
    main()

