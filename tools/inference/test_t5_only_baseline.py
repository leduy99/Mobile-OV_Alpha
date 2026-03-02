#!/usr/bin/env python3
"""
Test T5-only baseline for MobileOVModel (Phase 0 verification).

This script tests if MobileOVModel with T5-only (adapter/SmolVLM2 disabled)
produces output similar to WAN baseline, as required by expert's Phase 0.

Usage:
    python tools/inference/test_t5_only_baseline.py \
        --ckpt_dir omni_ckpts/wan/wanxiang1_3b \
        --prompt "a cat playing with a wool beside the fireside" \
        --size 832*480 \
        --output_dir output/t5_only_baseline_test
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import argparse
import logging
import math
import warnings
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV (cv2) not available. Video processing will be disabled.")
    CV2_AVAILABLE = False

from nets.third_party.wan.configs import WAN_CONFIGS
from nets.third_party.wan.modules.t5 import T5EncoderModel
from nets.third_party.wan.modules.vae import WanVAE
from nets.third_party.wan.modules.model import WanModel
from nets.third_party.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from nets.third_party.wan.utils.utils import cache_video, cache_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_video(video_tensor, output_path, fps=8):
    """Save video tensor to file."""
    if not CV2_AVAILABLE:
        logger.error("OpenCV not available, cannot save video")
        return
    
    # video_tensor: (C, N, H, W)
    video_np = video_tensor.permute(1, 2, 3, 0).cpu().numpy()  # (N, H, W, C)
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    if video_np.shape[-1] == 3:
        video_np = video_np[..., ::-1]
    
    height, width = video_np.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in video_np:
        out.write(frame)
    out.release()
    logger.info(f"Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test T5-only baseline for MobileOVModel")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to WAN checkpoint directory")
    parser.add_argument("--prompt", type=str, default="a cat playing with a wool beside the fireside", help="Text prompt")
    parser.add_argument("--size", type=str, default="832*480", help="Video size (W*H)")
    parser.add_argument("--output_dir", type=str, default="output/t5_only_baseline_test", help="Output directory")
    parser.add_argument("--sample_steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--frame_num", type=int, default=81, help="Number of frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample_shift", type=float, default=5.0, help="Scheduler shift parameter")
    parser.add_argument("--guide_scale", type=float, default=5.0, help="Classifier-Free Guidance scale (default: 5.0, matching OmniVideo)")
    parser.add_argument("--n_prompt", type=str, default="", help="Negative prompt for CFG (default: empty string)")
    
    args = parser.parse_args()
    
    # Parse size
    size = tuple(map(int, args.size.split('*')))
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    cfg = WAN_CONFIGS.get('t2v-1.3B', {})
    
    # Initialize T5 encoder
    logger.info("Initializing T5 encoder...")
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
        shard_fn=None
    )
    
    # Initialize VAE
    logger.info("Initializing VAE...")
    vae_stride = cfg.vae_stride
    patch_size = cfg.patch_size
    vae = WanVAE(
        vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
        device=device
    )
    
    # Initialize WAN model (T5-only, no adapter/SmolVLM2)
    logger.info("Initializing WAN model (T5-only baseline)...")
    wan_model = WanModel.from_pretrained(args.ckpt_dir)
    wan_model.eval()
    wan_model.to(device)
    
    # Initialize scheduler
    num_train_timesteps = cfg.num_train_timesteps
    sample_shift = args.sample_shift
    logger.info(f"Initializing FlowUniPCMultistepScheduler (shift={sample_shift})...")
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False
    )
    sample_scheduler.set_timesteps(
        args.sample_steps,
        device=device,
        shift=sample_shift
    )
    timesteps = sample_scheduler.timesteps
    
    # Encode text prompt with T5 (conditioned)
    logger.info(f"Encoding prompt with T5: '{args.prompt}'")
    text_encoder.model.to(device)
    context = text_encoder([args.prompt], device)
    
    # Encode negative prompt with T5 (unconditioned) for CFG
    if args.guide_scale > 0:
        n_prompt = args.n_prompt if args.n_prompt else ""  # Empty string if not provided
        logger.info(f"Encoding negative prompt with T5 for CFG (guide_scale={args.guide_scale}): '{n_prompt}'")
        context_null = text_encoder([n_prompt], device)
    else:
        context_null = None
        logger.info("CFG disabled (guide_scale=0)")
    
    text_encoder.model.cpu()  # Offload to save memory
    
    # Calculate target shape and seq_len
    # WAN expects input shape: [C_in, F, H, W] where C_in is the input channel dimension (in_dim, typically 16)
    F = args.frame_num
    # Get in_dim from WAN model config (typically 16)
    in_dim = wan_model.in_dim
    # VAE stride: [C, F_stride, H_stride, W_stride]
    # After VAE encoding, latent shape is [C, F', H', W'] where:
    # F' = (F - 1) // F_stride + 1
    # H' = H // H_stride
    # W' = W // W_stride
    # C = in_dim (input channels for WAN, typically 16)
    target_shape = (
        in_dim,  # C: input channels for WAN (typically 16)
        (F - 1) // vae_stride[0] + 1,  # F': frames after VAE
        size[1] // vae_stride[1],  # H': height after VAE
        size[0] // vae_stride[2]   # W': width after VAE
    )
    
    sp_size = getattr(cfg, 'sp_size', 1)
    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (patch_size[1] * patch_size[2]) *
                        target_shape[1] / sp_size) * sp_size
    
    logger.info(f"Target shape (C, F, H, W): {target_shape}, seq_len: {seq_len}")
    
    # Generate noise in VAE latent space
    # WAN expects List[Tensor] where each tensor is [C, F, H, W]
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.seed)
    noise = torch.randn(
        target_shape[0],  # C
        target_shape[1],  # F
        target_shape[2],  # H
        target_shape[3],  # W
        dtype=torch.float32,
        device=device,
        generator=seed_g
    )
    latents = [noise]  # WAN expects List[Tensor]
    
    # Calculate context_lens for proper attention masking
    # T5 context shape: List[Tensor] where each is [L, C]
    if isinstance(context, list):
        context_lens = torch.tensor([ctx.shape[0] for ctx in context], dtype=torch.long, device=device)
    else:
        context_lens = torch.tensor([context.shape[0]], dtype=torch.long, device=device)
    logger.info(f"Context length: {context_lens.tolist()}")
    
    # Calculate context_lens for null context (if CFG is enabled)
    if args.guide_scale > 0 and context_null is not None:
        if isinstance(context_null, list):
            context_null_lens = torch.tensor([ctx.shape[0] for ctx in context_null], dtype=torch.long, device=device)
        else:
            context_null_lens = torch.tensor([context_null.shape[0]], dtype=torch.long, device=device)
    else:
        context_null_lens = None
    
    # Denoising loop (T5-only, no adapter/SmolVLM2)
    # FIX: Use bfloat16 like OmniVideo for faster inference (bfloat16 is much faster than float32 on modern GPUs)
    param_dtype = torch.bfloat16  # Match OmniVideo's param_dtype
    logger.info(f"Starting denoising loop ({args.sample_steps} steps) with CFG (guide_scale={args.guide_scale})...")
    logger.info(f"Using autocast dtype: {param_dtype} (matching OmniVideo for optimal performance)")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=param_dtype):
        for t in tqdm(timesteps, desc="Sampling"):
            timestep = torch.stack([t]).to(device)
            
            # Forward pass with T5-only context (conditioned)
            # WAN model expects context as List[Tensor] where each tensor is [L, C]
            # Pass context_lens for proper attention masking
            velocity_pred_cond = wan_model(
                x=latents,
                t=timestep,
                context=context,  # T5 context (conditioned)
                seq_len=seq_len,
                context_lens=context_lens  # FIX: Pass context_lens for proper attention masking
            )
            
            if isinstance(velocity_pred_cond, list):
                velocity_pred_cond = velocity_pred_cond[0]
            
            # Forward pass with null context (unconditioned) for CFG
            if args.guide_scale > 0 and context_null is not None:
                velocity_pred_uncond = wan_model(
                    x=latents,
                    t=timestep,
                    context=context_null,  # T5 null context (unconditioned)
                    seq_len=seq_len,
                    context_lens=context_null_lens  # FIX: Pass context_lens for proper attention masking
                )
                
                if isinstance(velocity_pred_uncond, list):
                    velocity_pred_uncond = velocity_pred_uncond[0]
                
                # Apply Classifier-Free Guidance (CFG)
                # Formula: noise_pred = noise_pred_cond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                velocity_pred = velocity_pred_cond + args.guide_scale * (velocity_pred_cond - velocity_pred_uncond)
            else:
                # No CFG: use conditioned prediction directly
                velocity_pred = velocity_pred_cond
            
            # Scheduler step
            temp_x0 = sample_scheduler.step(
                velocity_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g
            )[0]
            latents = [temp_x0.squeeze(0)]
    
    # Decode with VAE
    logger.info("Decoding video with VAE...")
    x0 = latents
    videos = vae.decode(x0)
    video_tensor = videos[0]  # (C, N, H, W)
    logger.info(f"Video decoded: shape {video_tensor.shape}")
    
    # Save video
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c for c in args.prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')[:50]
    
    video_path = os.path.join(
        args.output_dir,
        f"t5_only_baseline_{args.size}_seed{args.seed}_steps{args.sample_steps}_"
        f"frames{args.frame_num}_{safe_prompt}_{timestamp}.mp4"
    )
    
    save_video(video_tensor, video_path, fps=8)
    logger.info(f"✅ T5-only baseline test complete! Video saved to: {video_path}")
    logger.info(f"📊 This is Phase 0 verification: T5-only output should match WAN baseline quality")
    logger.info(f"   If this looks good, proceed to Phase 1 (effect distillation)")
    logger.info(f"   If this looks bad, fix scheduler/latent scaling first")


if __name__ == "__main__":
    main()
