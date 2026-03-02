#!/usr/bin/env python3
"""
Inference script for MobileOVModel (SmolVLM2 + WAN + Adapter)

Usage:
    python generate_mobile_ov.py --prompt "a cat playing with a wool beside the fireside" --size 832*480
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import argparse
import logging
import math
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV (cv2) not available. Video processing will be disabled.")
    CV2_AVAILABLE = False
    cv2 = None

# Local imports
try:
    from nets.third_party.wan.configs import WAN_CONFIGS
    from nets.third_party.wan.modules.t5 import T5EncoderModel
    from nets.third_party.wan.modules.vae import WanVAE
    from nets.third_party.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from nets.omni.modules.mobile_ov_model import MobileOVModel
    from nets.third_party.wan.utils.utils import cache_video, cache_image
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def init_logging(rank=0):
    """Initialize logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MobileOVModel Inference: Generate video with SmolVLM2 + WAN"
    )
    
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--size", type=str, default="832*480",
        help="Output size in format 'width*height'"
    )
    parser.add_argument(
        "--frame_num", type=int, default=81,
        help="Number of frames to generate (should be 4n+1)"
    )
    parser.add_argument(
        "--sample_steps", type=int, default=50,
        help="Number of diffusion sampling steps"
    )
    parser.add_argument(
        "--sample_guide_scale", type=float, default=5.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--sample_shift", type=float, default=5.0,
        help="Noise schedule shift parameter"
    )
    parser.add_argument(
        "--base_seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="omni_ckpts/wan/wanxiang1_3b",
        help="Path to WAN checkpoint directory"
    )
    parser.add_argument(
        "--adapter_ckpt_dir", type=str, default="omni_ckpts/wan/wanxiang1_3b/adapter",
        help="Path to adapter checkpoint directory"
    )
    parser.add_argument(
        "--smolvlm2_ckpt_path", type=str, 
        default="omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt",
        help="Path to SmolVLM2 checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory for generated videos"
    )
    parser.add_argument(
        "--save_file", type=str, default=None,
        help="Output file path (auto-generated if not provided)"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device ID"
    )
    
    return parser.parse_args()


def save_video(video_tensor, output_path, fps=8):
    """Save video tensor to file."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required to save videos")
    
    # video_tensor shape: (C, N, H, W) -> (N, H, W, C)
    if video_tensor.dim() == 4:
        C, N, H, W = video_tensor.shape
        video_np = video_tensor.permute(1, 2, 3, 0).cpu().numpy()  # (N, H, W, C)
    else:
        raise ValueError(f"Unexpected video tensor shape: {video_tensor.shape}")
    
    # Convert to uint8
    video_np = np.clip((video_np + 1) / 2 * 255, 0, 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    if video_np.shape[-1] == 3:
        video_np = video_np[..., ::-1]  # RGB -> BGR
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for frame in video_np:
        out.write(frame)
    
    out.release()
    logging.info(f"Video saved to {output_path}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Initialize logging
    rank = int(os.getenv("RANK", 0))
    init_logging(rank)
    
    # Set device
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(args.device)
    
    # Parse size
    try:
        w, h = args.size.split('*')
        target_size = (int(h), int(w))  # (height, width)
    except:
        target_size = (480, 832)
        logging.warning(f"Invalid size format: {args.size}, using default {target_size}")
    
    logging.info(f"Target size: {target_size} (H x W)")
    logging.info(f"Prompt: {args.prompt}")
    
    # Load config
    cfg = WAN_CONFIGS.get('t2v-1.3B', {})
    
    # Initialize T5 encoder
    logging.info("Loading T5 encoder...")
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
        shard_fn=None
    )
    
    # Initialize VAE
    logging.info("Loading VAE...")
    vae_stride = cfg.vae_stride
    patch_size = cfg.patch_size
    vae = WanVAE(
        vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
        device=device
    )
    
    # Initialize MobileOVModel
    logging.info("Loading MobileOVModel...")
    model = MobileOVModel.from_pretrained(
        wan_ckpt_dir=args.ckpt_dir,
        adapter_ckpt_dir=args.adapter_ckpt_dir,
        smolvlm2_ckpt_path=args.smolvlm2_ckpt_path,
        use_precomputed_features=False,  # Use SmolVLM2 to encode prompts
        adapter_in_channels=1152,
        adapter_out_channels=4096,
        adapter_query_length=256,
        precision_dtype=torch.float32,
        device_id=args.device,
        rank=rank,
        use_visual_context_adapter=True,
        visual_context_adapter_patch_size=(1, 4, 4),
        max_context_len=2560,
    )
    model.eval()
    model.to(device)
    
    # Prepare generation parameters
    F = args.frame_num
    target_shape = (
        vae.model.z_dim,
        (F - 1) // vae_stride[0] + 1,
        target_size[0] // vae_stride[1],
        target_size[1] // vae_stride[2]
    )
    
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3]) / (patch_size[1] * patch_size[2]) * target_shape[1]
    )
    
    # Set random seed (used for scheduler step)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.base_seed)
    
    # Encode text prompt with T5
    # FIX: Không cần context_null vì đã tắt CFG
    logging.info("Encoding text prompt...")
    text_encoder.model.to(device)
    context = text_encoder([args.prompt], device)
    text_encoder.model.cpu()  # Offload to save memory
    
    # Generate noise
    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=device,
            generator=seed_g
        )
    ]
    
    # Initialize scheduler
    # FIX: Dùng FlowUniPCMultistepScheduler shift=5.0 như OmniVideo gốc để match pretrained checkpoint
    logging.info("Initializing scheduler (using FlowUniPCMultistepScheduler like OmniVideo original)...")
    num_train_timesteps = cfg.num_train_timesteps
    sample_shift = args.sample_shift if args.sample_shift > 0 else 5.0  # Use arg or default to 5.0 like OmniVideo
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1,  # Init với shift=1, sau đó set shift trong set_timesteps (giống OmniVideo)
        use_dynamic_shifting=False
    )
    sample_scheduler.set_timesteps(
        args.sample_steps,
        device=device,
        shift=sample_shift  # Set shift (default 5.0) như OmniVideo gốc
    )
    timesteps = sample_scheduler.timesteps
    
    # Generate video
    # FIX: Tắt CFG vì training không có CFG (classifier_free.ratio: 0.0)
    # Chỉ dùng conditional prediction, không có uncond branch
    logging.info(f"Generating video with {args.sample_steps} steps, shift={sample_shift}, scheduler=FlowUniPCMultistepScheduler (no CFG, guidance_scale=1.0)...")
    latents = noise
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
        for t in tqdm(timesteps, desc="Sampling"):
            # FIX: Move timestep to device
            timestep = torch.stack([t]).to(device)
            
            # Conditional prediction only (no CFG)
            # Note: model.forward expects x as a list and returns a tensor
            noise_pred = model(
                latents,  # List of tensors
                t=timestep,
                context=context,  # Can be None if disable_t5_context=True
                prompts=[args.prompt],  # Pass prompt for SmolVLM2 encoding
                seq_len=seq_len,
                condition_mode="full"
            )
            
            # Handle case where model returns a list (take first element)
            if isinstance(noise_pred, list):
                noise_pred = noise_pred[0]
            
            # No CFG - use conditional prediction directly
            # Training didn't use CFG, so we shouldn't use it in inference either
            
            # Scheduler step
            # FlowUniPCMultistepScheduler.step() signature: step(model_output, timestep, sample, return_dict=False, generator=None)
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),  # model_output
                t,  # timestep
                latents[0].unsqueeze(0),  # sample
                return_dict=False,
                generator=seed_g
            )[0]  # Returns tuple when return_dict=False
            latents = [temp_x0.squeeze(0)]
    
    # Decode with VAE
    logging.info("Decoding video with VAE...")
    x0 = latents
    videos = vae.decode(x0)
    video_tensor = videos[0]  # (C, N, H, W)
    
    # Save video
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.save_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in args.prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')[:50]
        args.save_file = os.path.join(
            args.output_dir,
            f"mobile_ov_{args.size}_seed{args.base_seed}_steps{args.sample_steps}_"
            f"cfg{args.sample_guide_scale}_frames{args.frame_num}_{safe_prompt}_{timestamp}.mp4"
        )
    
    save_video(video_tensor, args.save_file, fps=8)
    logging.info(f"✅ Generation complete! Video saved to: {args.save_file}")


if __name__ == "__main__":
    main()

