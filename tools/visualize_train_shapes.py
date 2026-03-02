#!/usr/bin/env python3
"""
Visualize key training-time shapes (latents, prompt embeds) for Stage-1 teacher-free.
This script does NOT run diffusion; it only inspects data + student conditioner.
"""
import argparse
import glob
import os
import pickle
import random

import torch
import yaml

from tools.train_stage1_teacher_free import AttrDict, load_sana_config, build_student


def _to_attr(d):
    if isinstance(d, dict):
        return AttrDict({k: _to_attr(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_attr(x) for x in d]
    return d


def _load_cfg(path: str) -> AttrDict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return _to_attr(data)


def _pick_preprocessed(pre_dir: str) -> str | None:
    if not pre_dir or not os.path.isdir(pre_dir):
        return None
    paths = sorted(glob.glob(os.path.join(pre_dir, "*_features.pkl")))
    return paths[0] if paths else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample-pkl", default=None, help="Override a specific preprocessed pkl file")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--show-student", action="store_true", help="Run student prompt embeds for a sample prompt")
    ap.add_argument("--prompt", default="a cat playing with a wool beside a fireside")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    sana_cfg = load_sana_config(cfg.sana.config)

    print("=== Config summary ===")
    print(f"Train prompt max tokens: {cfg.data.preprocessing.get('max_prompt_tokens')}")
    print(f"Use chi prompt: {cfg.data.preprocessing.get('use_chi_prompt')}")
    print(f"Prompt templates: {len(cfg.data.get('prompt_templates', []))}")
    print(f"Motion score: {cfg.data.get('motion_score')}")
    print(f"Latent window (frames): {cfg.train.get('latent_window', {}).get('frames')}")
    print(f"VAE stride: {getattr(sana_cfg.vae, 'vae_stride', None)}")
    print(f"VAE downsample: {sana_cfg.vae.vae_downsample_rate}")
    print(f"VAE latent dim: {sana_cfg.vae.vae_latent_dim}")

    sample_pkl = args.sample_pkl or _pick_preprocessed(cfg.data.openvid.preprocessed_dir)
    if sample_pkl:
        with open(sample_pkl, "rb") as f:
            data = pickle.load(f)
        lat = data.get("latent_feature")
        print("\n=== Preprocessed sample ===")
        print(f"File: {sample_pkl}")
        print(f"Keys: {list(data.keys())}")
        if lat is not None:
            print(f"latent_feature shape: {tuple(lat.shape)}  # [C, T, H, W]")
            vae_stride = getattr(sana_cfg.vae, "vae_stride", [1, sana_cfg.vae.vae_downsample_rate, sana_cfg.vae.vae_downsample_rate])
            stride_t = vae_stride[0] if isinstance(vae_stride, (list, tuple)) and len(vae_stride) >= 1 else 1
            t_lat = lat.shape[1]
            est_frames = (t_lat - 1) * stride_t + 1
            print(f"Estimated pixel frames from latent T: {est_frames}")
        else:
            print("latent_feature not found in sample.")
    else:
        print("\nNo preprocessed pickle found.")

    if args.show_student:
        print("\n=== Student prompt embeds ===")
        device = torch.device(args.device)
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        student = build_student(cfg, device, dtype)
        student.eval()
        with torch.no_grad():
            out = student([args.prompt])
        print(f"Student prompt embeds shape: {tuple(out.shape)}  # [B, L, C]")


if __name__ == "__main__":
    main()
