#!/usr/bin/env python3
"""
Profile T5-only baseline to find performance bottlenecks.

Usage:
    python tools/inference/profile_t5_baseline.py \
        --ckpt_dir omni_ckpts/wan/wanxiang1_3b \
        --prompt "a cat playing with a wool beside the fireside" \
        --sample_steps 5
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import argparse
import logging
import math
import time
import torch
import torch.cuda as cuda
from nets.third_party.wan.configs import WAN_CONFIGS
from nets.third_party.wan.modules.t5 import T5EncoderModel
from nets.third_party.wan.modules.vae import WanVAE
from nets.third_party.wan.modules.model import WanModel
from nets.third_party.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def profile_step(wan_model, latents, timestep, context, context_null, context_lens, context_null_lens, 
                 seq_len, guide_scale, device):
    """Profile a single denoising step."""
    times = {}
    
    # Conditioned forward
    cuda.synchronize()
    start = time.time()
    velocity_pred_cond = wan_model(
        x=latents,
        t=timestep,
        context=context,
        seq_len=seq_len,
        context_lens=context_lens
    )
    cuda.synchronize()
    times['forward_cond'] = time.time() - start
    
    if isinstance(velocity_pred_cond, list):
        velocity_pred_cond = velocity_pred_cond[0]
    
    # Unconditioned forward (if CFG)
    if guide_scale > 0 and context_null is not None:
        cuda.synchronize()
        start = time.time()
        velocity_pred_uncond = wan_model(
            x=latents,
            t=timestep,
            context=context_null,
            seq_len=seq_len,
            context_lens=context_null_lens
        )
        cuda.synchronize()
        times['forward_uncond'] = time.time() - start
        
        if isinstance(velocity_pred_uncond, list):
            velocity_pred_uncond = velocity_pred_uncond[0]
        
        # CFG calculation
        cuda.synchronize()
        start = time.time()
        velocity_pred = velocity_pred_cond + guide_scale * (velocity_pred_cond - velocity_pred_uncond)
        cuda.synchronize()
        times['cfg_calc'] = time.time() - start
    else:
        velocity_pred = velocity_pred_cond
        times['forward_uncond'] = 0.0
        times['cfg_calc'] = 0.0
    
    times['total'] = times['forward_cond'] + times['forward_uncond'] + times['cfg_calc']
    
    return velocity_pred, times


def main():
    parser = argparse.ArgumentParser(description="Profile T5-only baseline")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a cat playing with a wool beside the fireside")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--sample_steps", type=int, default=5, help="Number of steps to profile")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = WAN_CONFIGS.get('t2v-1.3B', {})
    
    # Initialize models
    logger.info("Initializing models...")
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
        shard_fn=None
    )
    
    wan_model = WanModel.from_pretrained(args.ckpt_dir)
    wan_model.eval()
    wan_model.to(device)
    
    # Encode prompts
    logger.info("Encoding prompts...")
    text_encoder.model.to(device)
    context = text_encoder([args.prompt], device)
    n_prompt = ""
    context_null = text_encoder([n_prompt], device) if args.guide_scale > 0 else None
    text_encoder.model.cpu()
    
    # Calculate shapes
    size = tuple(map(int, args.size.split('*')))
    F = args.frame_num
    vae_stride = cfg.vae_stride
    patch_size = cfg.patch_size
    in_dim = wan_model.in_dim
    
    target_shape = (
        in_dim,
        (F - 1) // vae_stride[0] + 1,
        size[1] // vae_stride[1],
        size[0] // vae_stride[2]
    )
    
    sp_size = getattr(cfg, 'sp_size', 1)
    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (patch_size[1] * patch_size[2]) *
                        target_shape[1] / sp_size) * sp_size
    
    # Context lengths
    if isinstance(context, list):
        context_lens = torch.tensor([ctx.shape[0] for ctx in context], dtype=torch.long, device=device)
    else:
        context_lens = torch.tensor([context.shape[0]], dtype=torch.long, device=device)
    
    if context_null is not None:
        if isinstance(context_null, list):
            context_null_lens = torch.tensor([ctx.shape[0] for ctx in context_null], dtype=torch.long, device=device)
        else:
            context_null_lens = torch.tensor([context_null.shape[0]], dtype=torch.long, device=device)
    else:
        context_null_lens = None
    
    # Initialize scheduler
    num_train_timesteps = cfg.num_train_timesteps
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False
    )
    sample_scheduler.set_timesteps(args.sample_steps, device=device, shift=args.sample_shift)
    timesteps = sample_scheduler.timesteps
    
    # Generate noise
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.seed)
    noise = torch.randn(*target_shape, dtype=torch.float32, device=device, generator=seed_g)
    latents = [noise]
    
    # Warmup
    # FIX: Use bfloat16 like OmniVideo for faster inference
    param_dtype = torch.bfloat16  # Match OmniVideo's param_dtype
    logger.info("Warming up...")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=param_dtype):
        for _ in range(2):
            t = timesteps[0]
            timestep = torch.stack([t]).to(device)
            _, _ = profile_step(wan_model, latents, timestep, context, context_null, 
                              context_lens, context_null_lens, seq_len, args.guide_scale, device)
    
    # Profile
    logger.info(f"Profiling {args.sample_steps} steps...")
    all_times = []
    
    # FIX: Use bfloat16 like OmniVideo for faster inference
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=param_dtype):
        for step_idx, t in enumerate(timesteps):
            timestep = torch.stack([t]).to(device)
            
            velocity_pred, step_times = profile_step(
                wan_model, latents, timestep, context, context_null,
                context_lens, context_null_lens, seq_len, args.guide_scale, device
            )
            
            all_times.append(step_times)
            
            # Scheduler step
            temp_x0 = sample_scheduler.step(
                velocity_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g
            )[0]
            latents = [temp_x0.squeeze(0)]
            
            if step_idx == 0:
                logger.info(f"Step {step_idx+1} times: {step_times}")
    
    # Aggregate results
    avg_times = {}
    for key in all_times[0].keys():
        avg_times[key] = sum(t[key] for t in all_times) / len(all_times)
    
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE PROFILE")
    logger.info("="*80)
    logger.info(f"Average times per step:")
    logger.info(f"  Forward (conditioned): {avg_times['forward_cond']*1000:.2f}ms")
    if args.guide_scale > 0:
        logger.info(f"  Forward (unconditioned): {avg_times['forward_uncond']*1000:.2f}ms")
        logger.info(f"  CFG calculation: {avg_times['cfg_calc']*1000:.2f}ms")
    logger.info(f"  Total per step: {avg_times['total']*1000:.2f}ms")
    logger.info(f"\nConfiguration:")
    logger.info(f"  seq_len: {seq_len}")
    logger.info(f"  context_lens: {context_lens.tolist()}")
    logger.info(f"  guide_scale: {args.guide_scale}")
    logger.info(f"  target_shape: {target_shape}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
