#!/usr/bin/env python3
"""
Profile Mobile-OV-SANA component latencies (end-to-end, excluding model load).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.cuda as cuda

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from nets.omni.modules.sana_prompt_bridge import SanaPromptBridge
from tools.inference.sana_video_inference_fixed import (
    load_config_file,
    load_sana_models,
    encode_text,
)


def _sync():
    if torch.cuda.is_available():
        cuda.synchronize()


def _timeit(fn, *args, **kwargs):
    _sync()
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    _sync()
    return out, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Profile Mobile-OV-SANA latency")
    parser.add_argument("--bridge-ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a cat playing with a wool beside the fireside")
    parser.add_argument("--config", type=str, default="configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml")
    parser.add_argument("--ckpt-dir", type=str, default="omni_ckpts/sana_video_2b_480p")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--use_chi_prompt", action="store_true")
    parser.add_argument("--force-adapter-query-length", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    width, height = map(int, args.size.split("*"))

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
    diffusion_model = models["diffusion_model"]

    bridge = SanaPromptBridge(
        smolvlm2_ckpt_path="omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt",
        adapter_ckpt_dir="omni_ckpts/wan/wanxiang1_3b/adapter",
        adapter_in_channels=1152,
        adapter_out_channels=4096,
        adapter_query_length=64,
        smol_vh_num_queries=1,
        num_prompt_queries=config.text_encoder.model_max_length,
        caption_channels=getattr(config.text_encoder, "caption_channels", 2304),
        precision_dtype=dtype,
        device=device,
        force_adapter_query_length=args.force_adapter_query_length,
    )

    ckpt = torch.load(args.bridge_ckpt, map_location="cpu")
    state = ckpt.get("student_state", ckpt)
    bridge.smolvlm2_vision_head.load_state_dict(state["smolvlm2_vision_head"])
    bridge.adapter.load_state_dict(state["adapter"], strict=False)
    bridge.adapter_output_norm.load_state_dict(state["adapter_output_norm"])
    bridge.adapter_output_gate.data.copy_(state["adapter_output_gate"])
    bridge.resampler.load_state_dict(state["resampler"])

    prompt = args.prompt.strip()
    if args.use_chi_prompt:
        chi_list = getattr(config.text_encoder, "chi_prompt", None)
        if chi_list:
            prompt = "\n".join(chi_list) + prompt

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = bridge([prompt])

    timings = {}

    # Tokenizer + SmolVLM2 forward
    tokenizer = bridge._get_tokenizer()
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    with torch.no_grad():
        smol_out, t_smol = _timeit(
            bridge.smolvlm2_model,
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    timings["smolvlm2_forward"] = t_smol

    if hasattr(smol_out, "last_hidden_state"):
        hidden_states = smol_out.last_hidden_state
    elif hasattr(smol_out, "hidden_states") and isinstance(smol_out.hidden_states, (list, tuple)):
        hidden_states = smol_out.hidden_states[-1]
    else:
        hidden_states = smol_out[0]

    with torch.no_grad():
        vh_out, t_vh = _timeit(bridge.smolvlm2_vision_head, hidden_states)
        adapter_out, t_ad = _timeit(bridge.adapter, vh_out)
        if adapter_out.dim() == 2:
            adapter_out = adapter_out.unsqueeze(0)
        adapter_out = bridge.adapter_output_norm(adapter_out)
        adapter_out = adapter_out * bridge.adapter_output_gate
        prompt_embeds, t_resampler = _timeit(bridge.resampler, adapter_out)
    timings["vision_head"] = t_vh
    timings["adapter"] = t_ad
    timings["resampler"] = t_resampler

    # Mask from teacher tokenizer
    _, _, prompt_mask = encode_text(
        models["tokenizer"],
        models["text_encoder"],
        prompt,
        config,
        device=str(device),
    )
    prompt_embeds = prompt_embeds.unsqueeze(1)

    # Latents
    vae_stride = getattr(config.vae, "vae_stride", [1, config.vae.vae_downsample_rate])
    vae_stride_t = vae_stride[0] if isinstance(vae_stride, list) and len(vae_stride) >= 1 else 1
    latent_h = height // config.vae.vae_downsample_rate
    latent_w = width // config.vae.vae_downsample_rate
    latent_t = int(args.frame_num - 1) // vae_stride_t + 1
    latent_shape = (1, config.vae.vae_latent_dim, latent_t, latent_h, latent_w)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(latent_shape, device=device, dtype=dtype, generator=gen)

    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=getattr(config.scheduler, "inference_flow_shift", 7.0)
    )
    timesteps, _ = retrieve_timesteps(scheduler, args.steps, device, None)

    # Denoise loop timing (forward + scheduler)
    step_times = []
    with torch.no_grad():
        for t in timesteps:
            t_in = t.expand(latents.shape[0])
            noise_pred, t_fwd = _timeit(
                diffusion_model,
                latents,
                t_in,
                prompt_embeds,
                mask=prompt_mask,
            )
            _sync()
            start = time.perf_counter()
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            _sync()
            t_sched = time.perf_counter() - start
            step_times.append((t_fwd, t_sched))

    timings["sana_forward_per_step"] = float(np.mean([t[0] for t in step_times]))
    timings["scheduler_per_step"] = float(np.mean([t[1] for t in step_times]))
    timings["sana_total_steps"] = len(step_times)

    # VAE decode
    vae = models["vae"]
    if hasattr(vae, "decode"):
        _, t_decode = _timeit(vae.decode, latents.to(models.get("vae_dtype", latents.dtype)))
    else:
        from diffusion.model.builder import vae_decode
        _, t_decode = _timeit(
            vae_decode,
            config.vae.vae_type,
            vae,
            latents.to(models.get("vae_dtype", latents.dtype)),
        )
    timings["vae_decode"] = t_decode

    total = (
        timings["smolvlm2_forward"]
        + timings["vision_head"]
        + timings["adapter"]
        + timings["resampler"]
        + timings["sana_forward_per_step"] * timings["sana_total_steps"]
        + timings["scheduler_per_step"] * timings["sana_total_steps"]
        + timings["vae_decode"]
    )
    timings["approx_total_no_io"] = total

    print("=" * 80)
    print("MOBILE-OV-SANA LATENCY PROFILE (approx, no model load)")
    print("=" * 80)
    print(f"SmolVLM2 forward: {timings['smolvlm2_forward']*1000:.2f} ms")
    print(f"VisionHead: {timings['vision_head']*1000:.2f} ms")
    print(f"Adapter: {timings['adapter']*1000:.2f} ms")
    print(f"Resampler: {timings['resampler']*1000:.2f} ms")
    print(f"SANA forward per step: {timings['sana_forward_per_step']*1000:.2f} ms")
    print(f"Scheduler per step: {timings['scheduler_per_step']*1000:.2f} ms")
    print(f"VAE decode: {timings['vae_decode']*1000:.2f} ms")
    print(f"Approx total (no IO, {timings['sana_total_steps']} steps): {total:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
