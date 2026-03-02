#!/usr/bin/env python3
"""
Profile SANA-Video denoising performance (pure inference loop).

This script measures per-step runtime for the SANA transformer + CFG math
and optionally the scheduler step. It excludes model load and text encoding.
"""

import os
import sys
import time
import argparse
import torch
import torch.cuda as cuda

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
_sana_repo_root = os.path.join(project_root, "nets", "third_party", "sana")
if os.path.isdir(_sana_repo_root):
    sys.path.insert(0, _sana_repo_root)

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from tools.inference.sana_video_inference_fixed import (
    load_config_file,
    load_sana_models,
    encode_text,
    encode_negative_prompt,
)


def profile_step(model, latents, timestep, prompt_embeds, prompt_mask, cfg_scale):
    times = {}
    do_cfg = cfg_scale > 1.0

    if do_cfg:
        latent_input = torch.cat([latents] * 2)
        prompt_input = prompt_embeds
    else:
        latent_input = latents
        prompt_input = prompt_embeds

    cuda.synchronize()
    start = time.perf_counter()
    noise_pred = model(latent_input, timestep, prompt_input, mask=prompt_mask)
    cuda.synchronize()
    times["forward"] = time.perf_counter() - start

    if isinstance(noise_pred, (tuple, list)):
        noise_pred = noise_pred[0]

    if do_cfg:
        cuda.synchronize()
        start = time.perf_counter()
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        cuda.synchronize()
        times["cfg"] = time.perf_counter() - start
    else:
        times["cfg"] = 0.0

    times["total"] = times["forward"] + times["cfg"]
    return noise_pred, times


def main():
    parser = argparse.ArgumentParser(description="Profile SANA-Video denoising runtime")
    parser.add_argument("--ckpt_dir", type=str, default="omni_ckpts/sana_video_2b_480p")
    parser.add_argument("--config", type=str, default="configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml")
    parser.add_argument("--prompt", type=str, default="a cat playing with a wool beside the fireside")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--flow_shift", type=float, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--use_chi_prompt", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    config = load_config_file(args.config)
    if not args.use_chi_prompt:
        if hasattr(config, "text_encoder"):
            setattr(config.text_encoder, "chi_prompt", None)

    models = load_sana_models(
        config=config,
        checkpoint_dir=args.ckpt_dir,
        device=str(device),
        model_dtype=dtype,
        vae_dtype=torch.float32,
        latent_size=config.model.image_size // config.vae.vae_downsample_rate,
    )
    model = models["diffusion_model"]

    prompt_embeds, _, prompt_mask = encode_text(
        models["tokenizer"],
        models["text_encoder"],
        args.prompt,
        config,
        device=str(device),
    )
    prompt_embeds = prompt_embeds.unsqueeze(1)

    if args.cfg_scale > 1.0:
        negative_prompt = args.negative_prompt or ""
        neg_embeds, neg_mask = encode_negative_prompt(
            models["tokenizer"],
            models["text_encoder"],
            negative_prompt,
            config,
            device=str(device),
        )
        neg_embeds = neg_embeds.unsqueeze(1)
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
        prompt_mask = torch.cat([neg_mask, prompt_mask], dim=0)

    # Prepare latents
    size = tuple(map(int, args.size.split("*")))
    height, width = size[1], size[0]
    vae_stride = getattr(config.vae, "vae_stride", [1, config.vae.vae_downsample_rate])
    vae_stride_t = vae_stride[0] if isinstance(vae_stride, list) and len(vae_stride) >= 1 else 1
    latent_h = height // config.vae.vae_downsample_rate
    latent_w = width // config.vae.vae_downsample_rate
    latent_t = int(args.frame_num - 1) // vae_stride_t + 1
    latent_shape = (1, config.vae.vae_latent_dim, latent_t, latent_h, latent_w)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(latent_shape, device=device, dtype=dtype, generator=gen)

    scheduler = FlowMatchEulerDiscreteScheduler(shift=args.flow_shift or getattr(config.scheduler, "inference_flow_shift", 7.0))
    timesteps, _ = retrieve_timesteps(scheduler, args.sample_steps, device, None)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            t = timesteps[0].expand(latents.shape[0])
            _ = profile_step(model, latents, t, prompt_embeds, prompt_mask, args.cfg_scale)

    # Profile
    all_times = []
    with torch.no_grad():
        for t in timesteps:
            timestep = t.expand(latents.shape[0])
            noise_pred, step_times = profile_step(model, latents, timestep, prompt_embeds, prompt_mask, args.cfg_scale)
            cuda.synchronize()
            start = time.perf_counter()
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            cuda.synchronize()
            step_times["scheduler"] = time.perf_counter() - start
            step_times["total_with_scheduler"] = step_times["total"] + step_times["scheduler"]
            all_times.append(step_times)

    avg = {}
    for key in all_times[0].keys():
        avg[key] = sum(t[key] for t in all_times) / len(all_times)

    print("=" * 80)
    print("SANA-VIDEO PERFORMANCE PROFILE")
    print("=" * 80)
    print(f"Avg per step forward: {avg['forward'] * 1000:.2f} ms")
    print(f"Avg per step cfg: {avg['cfg'] * 1000:.2f} ms")
    print(f"Avg per step scheduler: {avg['scheduler'] * 1000:.2f} ms")
    print(f"Avg per step total (forward+cfg): {avg['total'] * 1000:.2f} ms")
    print(f"Avg per step total (incl scheduler): {avg['total_with_scheduler'] * 1000:.2f} ms")
    print(f"cfg_scale: {args.cfg_scale}, steps: {args.sample_steps}, size: {args.size}, frames: {args.frame_num}")
    print("=" * 80)


if __name__ == "__main__":
    main()
