#!/usr/bin/env python3
"""
Inference script for MobileOVModel with TRAINED adapter and projection.
Extract frames from generated video for quality check.

Usage (single prompt):
    python inference_trained_extract_frames.py \
        --checkpoint_dir output/training_overfit_20260112_020243/checkpoint_epoch_44_step_99/epoch_44_step_99 \
        --prompt "a cat playing with a wool beside the fireside" \
        --size 832*480

Usage (test with training data prompts):
    python inference_trained_extract_frames.py \
        --checkpoint_dir output/training_overfit_20260112_020243/checkpoint_epoch_44_step_99/epoch_44_step_99 \
        --csv_path data/openvid_test/OpenVid-1M_test_subset.csv \
        --num_samples 10 \
        --size 832*480
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
import shutil
import pandas as pd
import yaml
from easydict import EasyDict

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


def extract_trained_components(checkpoint_dir, output_dir):
    """Extract adapter and projection from DeepSpeed checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # DeepSpeed checkpoint structure: {'module': {...}}
    if 'module' not in checkpoint:
        logging.error("Checkpoint does not have 'module' key")
        return False
    
    model_state = checkpoint['module']
    
    # Extract adapter weights
    adapter_keys = [k for k in model_state.keys() if k.startswith('adapter.')]
    if adapter_keys:
        logging.info(f"Found {len(adapter_keys)} adapter keys")
        adapter_state = {}
        for k, v in model_state.items():
            if k.startswith('adapter.'):
                # Remove first 'adapter.' prefix
                new_key = k.replace('adapter.', '', 1)
                adapter_state[new_key] = v
        
        os.makedirs(os.path.join(output_dir, "adapter"), exist_ok=True)
        adapter_path = os.path.join(output_dir, "adapter", "adapter_pytorch_model.bin")
        torch.save(adapter_state, adapter_path)
        logging.info(f"✅ Saved adapter to {adapter_path}")
    else:
        logging.warning("No adapter keys found in checkpoint")
        return False
    
    # Extract vision_head weights (replaces projection in new architecture)
    vision_head_keys = [k for k in model_state.keys() if k.startswith('smolvlm2_vision_head.')]
    if vision_head_keys:
        logging.info(f"Found {len(vision_head_keys)} vision_head keys")
        vision_head_state = {}
        for k, v in model_state.items():
            if k.startswith('smolvlm2_vision_head.'):
                # Remove first 'smolvlm2_vision_head.' prefix
                new_key = k.replace('smolvlm2_vision_head.', '', 1)
                vision_head_state[new_key] = v
        
        os.makedirs(os.path.join(output_dir, "smolvlm2_vision_head"), exist_ok=True)
        vision_head_path = os.path.join(output_dir, "smolvlm2_vision_head", "pytorch_model.bin")  # FIX: Use correct name that model expects
        torch.save(vision_head_state, vision_head_path)
        logging.info(f"✅ Saved vision_head to {vision_head_path}")
    else:
        # Fallback: try projection (for backward compatibility)
        projection_keys = [k for k in model_state.keys() if k.startswith('smolvlm2_projection.')]
        if projection_keys:
            logging.info(f"Found {len(projection_keys)} projection keys (fallback)")
            projection_state = {k.replace('smolvlm2_projection.', ''): v for k, v in model_state.items() if k.startswith('smolvlm2_projection.')}
            
            os.makedirs(os.path.join(output_dir, "smolvlm2_projection"), exist_ok=True)
            projection_path = os.path.join(output_dir, "smolvlm2_projection", "smolvlm2_projection_pytorch_model.bin")
            torch.save(projection_state, projection_path)
            logging.info(f"✅ Saved projection to {projection_path}")
        else:
            logging.warning("No vision_head or projection keys found in checkpoint")
            # Don't return False - adapter is more important than vision_head/projection
    
    return True
    
    return True


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


def extract_frames(video_path, output_dir, max_frames=20):
    """Extract frames from video for quality check."""
    if not CV2_AVAILABLE:
        logging.warning("OpenCV not available, skipping frame extraction")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logging.info(f"Extracting frames from {video_path}")
    logging.info(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Extract evenly spaced frames
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count in frame_indices:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            # Convert BGR to RGB for saving
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(frame_path, quality=95)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    logging.info(f"✅ Extracted {saved_count} frames to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MobileOVModel Inference with Trained Checkpoint: Generate video and extract frames"
    )
    
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to DeepSpeed checkpoint directory (e.g., output/training_xxx/checkpoint_epoch_X_step_Y/epoch_X_step_Y)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt for generation (required if --csv_path not provided)"
    )
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="Path to training CSV file to load prompts from (optional)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of prompts to test from CSV (None = all)"
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
        "--sample_shift", type=float, default=3.0,
        help="Noise schedule shift parameter (default 3.0 to match training)"
    )
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="Gamma gating parameter for clue/T5 context mixing (0.0=T5-only, 1.0=full clue). If None, will auto-detect from training config."
    )
    parser.add_argument(
        "--use_precomputed_features", type=str, default=None,
        help="Use precomputed features (true/false/auto). If 'auto' or None, will auto-detect from training config."
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
        "--smolvlm2_ckpt_path", type=str, 
        default="omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt",
        help="Path to SmolVLM2 checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/inference",
        help="Output directory for generated videos and frames"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--extract_frames", action="store_true",
        help="Extract frames from generated video"
    )
    parser.add_argument(
        "--max_frames", type=int, default=20,
        help="Maximum number of frames to extract"
    )
    
    return parser.parse_args()


def load_training_config(checkpoint_dir):
    """
    Load training config from checkpoint directory.
    Training saves config as YAML in parent directory: output/training_xxx/mobile_ov_openvid_overfit.yaml
    """
    # Checkpoint dir format: output/training_xxx/checkpoint_epoch_X_step_Y/epoch_X_step_Y
    # Config is in: output/training_xxx/mobile_ov_openvid_overfit.yaml
    checkpoint_parent = os.path.dirname(os.path.dirname(checkpoint_dir))
    
    # Try to find config YAML file
    config_files = [
        os.path.join(checkpoint_parent, "mobile_ov_openvid_overfit.yaml"),
        os.path.join(checkpoint_parent, "*.yaml"),
    ]
    
    config = None
    for config_pattern in config_files:
        if '*' in config_pattern:
            # Try glob pattern
            import glob
            matches = glob.glob(config_pattern)
            if matches:
                config_path = matches[0]
            else:
                continue
        else:
            config_path = config_pattern
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logging.info(f"Loaded training config from: {config_path}")
                break
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {e}")
                continue
    
    if config is None:
        logging.warning("Could not find training config, using defaults")
        return None
    
    return EasyDict(config)


def load_training_prompts(csv_path, num_samples=None):
    """Load prompts from training CSV file."""
    logging.info(f"Loading prompts from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, quoting=1, escapechar='\\')
    except:
        try:
            df = pd.read_csv(csv_path, quoting=1)
        except:
            df = pd.read_csv(csv_path)
    
    # Ensure required columns
    if 'caption' not in df.columns:
        raise ValueError(f"CSV file missing 'caption' column")
    
    prompts = []
    for idx, row in df.iterrows():
        if num_samples and len(prompts) >= num_samples:
            break
        
        caption = str(row['caption']).strip()
        if pd.isna(caption) or len(caption) < 10:
            continue
        
        video_name = str(row.get('video', '')).strip()
        prompts.append({
            'prompt': caption,
            'video_name': video_name,
            'index': idx
        })
    
    logging.info(f"Loaded {len(prompts)} prompts from training data")
    return prompts


def main():
    """Main inference function."""
    args = parse_args()
    
    # Validate arguments
    if args.csv_path is None and args.prompt is None:
        logging.error("Either --prompt or --csv_path must be provided")
        return
    
    # Load prompts
    if args.csv_path:
        prompts_data = load_training_prompts(args.csv_path, args.num_samples)
        if not prompts_data:
            logging.error("No prompts loaded from CSV")
            return
        # Convert to list of prompts for processing
        prompts_list = [p['prompt'] for p in prompts_data]
        prompts_meta = prompts_data  # Keep metadata for naming
    else:
        prompts_list = [args.prompt]
        prompts_meta = [{'prompt': args.prompt, 'video_name': '', 'index': 0}]
    
    logging.info(f"Will process {len(prompts_list)} prompt(s)")
    
    # Initialize logging
    rank = int(os.getenv("RANK", 0))
    init_logging(rank)
    
    # Set device
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(args.device)
    
    # Load training config to auto-detect settings
    training_config = load_training_config(args.checkpoint_dir)
    
    # Auto-detect use_precomputed_features from training config
    if args.use_precomputed_features is None or args.use_precomputed_features.lower() == 'auto':
        if training_config is not None:
            use_precomputed_features = getattr(
                training_config.training.model_settings, 
                'use_precomputed_features', 
                False
            )
            logging.info(f"Auto-detected use_precomputed_features={use_precomputed_features} from training config")
        else:
            use_precomputed_features = False  # Default to False (safer, matches current training)
            logging.warning("Could not auto-detect use_precomputed_features, defaulting to False")
    else:
        use_precomputed_features = args.use_precomputed_features.lower() in ('true', '1', 'yes')
        logging.info(f"Using use_precomputed_features={use_precomputed_features} from command line")
    
    # Auto-detect gamma from training config (if not provided)
    if args.gamma is None:
        if training_config is not None:
            # Check if gamma curriculum was used (gamma_start, gamma_end)
            gamma_start = getattr(training_config.training.model_settings, 'gamma_start', None)
            gamma_end = getattr(training_config.training.model_settings, 'gamma_end', None)
            if gamma_end is not None:
                args.gamma = gamma_end
                logging.info(f"Auto-detected gamma={args.gamma} from training config (gamma_end)")
            else:
                args.gamma = 1.0  # Default to full clue if no curriculum
                logging.info(f"No gamma curriculum found, defaulting to gamma=1.0 (full clue)")
        else:
            args.gamma = 1.0  # Default to full clue
            logging.warning("Could not auto-detect gamma, defaulting to 1.0 (full clue)")
    else:
        logging.info(f"Using gamma={args.gamma} from command line")
    
    # Extract trained components from checkpoint
    trained_components_dir = os.path.join(args.output_dir, "trained_components")
    os.makedirs(trained_components_dir, exist_ok=True)
    
    logging.info(f"Extracting trained components from {args.checkpoint_dir}")
    if not extract_trained_components(args.checkpoint_dir, trained_components_dir):
        logging.error("Failed to extract trained components")
        return
    
    # Parse size
    try:
        w, h = args.size.split('*')
        target_size = (int(h), int(w))  # (height, width)
    except:
        target_size = (480, 832)
        logging.warning(f"Invalid size format: {args.size}, using default {target_size}")
    
    logging.info(f"Target size: {target_size} (H x W)")
    
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
    
    # Initialize MobileOVModel with trained components
    logging.info("Loading MobileOVModel with trained adapter and projection...")
    adapter_ckpt_dir = os.path.join(trained_components_dir, "adapter")
    model = MobileOVModel.from_pretrained(
        wan_ckpt_dir=args.ckpt_dir,
        adapter_ckpt_dir=adapter_ckpt_dir,
        smolvlm2_ckpt_path=args.smolvlm2_ckpt_path,
        use_precomputed_features=use_precomputed_features,  # Match training setting
        adapter_in_channels=1152,
        adapter_out_channels=4096,
        adapter_query_length=64,  # Match checkpoint (checkpoint was trained with 64 as per config)
        precision_dtype=torch.float32,
        device_id=args.device,
        rank=rank,
        use_visual_context_adapter=True,
        visual_context_adapter_patch_size=(1, 4, 4),
        max_context_len=2560,
        disable_t5_context=False,  # FIX: Enable T5 to match OmniVideo architecture (concat with SmolVLM2)
        use_smol_vh=True,  # Use VisionHead-style resampler (matches OmniVideo)
        smol_vh_num_queries=1,  # Q=1 for T2V bring-up (can increase to 4 later)
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
    
    # Scheduler will be initialized for each prompt (needs reset between prompts)
    num_train_timesteps = cfg.num_train_timesteps
    # FIX: Default shift=5.0 to match OmniVideo gốc (pretrained checkpoint expects UniPC with shift=5.0)
    # Can fallback to 3.0 (training shift) if needed, but UniPC is more stable with 5.0
    sample_shift = args.sample_shift if args.sample_shift > 0 else 5.0
    
    # Process each prompt
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for prompt_idx, (prompt, meta) in enumerate(zip(prompts_list, prompts_meta)):
        logging.info(f"\n{'='*80}")
        logging.info(f"[{prompt_idx+1}/{len(prompts_list)}] Processing prompt: {prompt[:100]}...")
        logging.info(f"{'='*80}")
        
        # Set random seed (different for each prompt)
        seed_g = torch.Generator(device=device)
        seed = args.base_seed + prompt_idx
        seed_g.manual_seed(seed)
        
        # Initialize scheduler for this prompt (needs to be fresh for each prompt)
        # FIX: Use FlowUniPCMultistepScheduler to match OmniVideo gốc (pretrained checkpoint expects UniPC)
        # Note: Model predicts velocity field (noise - x0), which is independent of scheduler
        # UniPC is more stable and can handle prediction errors better than simple Euler step
        logging.info(f"Initializing FlowUniPCMultistepScheduler (shift={sample_shift}, matching OmniVideo gốc)...")
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=1,  # Init với shift=1, sau đó set shift trong set_timesteps (giống OmniVideo)
            use_dynamic_shifting=False
        )
        sample_scheduler.set_timesteps(
            args.sample_steps,
            device=device,
            shift=sample_shift  # Set shift (default 3.0 from training, but can try 5.0 like OmniVideo gốc)
        )
        timesteps = sample_scheduler.timesteps
        
        # Encode text prompt with T5 (optional, since disable_t5_context=True)
        logging.info("Encoding text prompt with T5...")
        text_encoder.model.to(device)
        context = text_encoder([prompt], device)
        text_encoder.model.cpu()  # Offload to save memory
        
        # OPTIMIZATION: Pre-encode SmolVLM2 prompts BEFORE denoising loop (1 time only)
        # This avoids encoding SmolVLM2 50 times (once per denoising step)
        # Similar to OmniVideo which uses pre-computed ar_vision_input
        logging.info("Pre-encoding prompts with SmolVLM2 (1 time, cached for all denoising steps)...")
        model.eval()
        model.to(device)
        
        # Pre-encode prompts with SmolVLM2 + VisionHead + Adapter
        # CRITICAL OPTIMIZATION: Pre-compute adapter output ONCE (not just VisionHead output)
        # Adapter output doesn't change between denoising steps, so we can cache it
        # FIX: Use bfloat16 like OmniVideo for faster inference
        param_dtype = torch.bfloat16  # Match OmniVideo's param_dtype
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=param_dtype):
            # Encode prompts with SmolVLM2 (1 time)
            smolvlm2_hidden = model.encode_prompts_with_smolvlm2([prompt], device)  # [1, L, 1024]
            
            # Process through VisionHead (1 time)
            if model.use_smol_vh and model.smolvlm2_vision_head is not None:
                vision_tokens = model.smolvlm2_vision_head(smolvlm2_hidden)  # [1, Q, 1152]
                
                # CRITICAL: Pre-compute adapter output ONCE (this is the expensive operation)
                # Adapter output is [1, 256, 4096] and doesn't change between steps
                precomputed_adapter_output = model.adapter(vision_tokens)  # [1, Q, 1152] -> [1, 64, 4096] (K=64, adapter_query_length)
                
                # Store both for flexibility (we'll use adapter_output directly)
                precomputed_vision_input = vision_tokens  # Keep for fallback
                precomputed_adapter_output_tensor = precomputed_adapter_output  # [1, 64, 4096] (K=64)
            else:
                precomputed_vision_input = None
                precomputed_adapter_output_tensor = None
        
        logging.info("✅ SmolVLM2 + VisionHead + Adapter pre-computed. Denoising loop will use cached adapter output.")
        
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
        
        # Generate video
        logging.info(f"Generating video with {args.sample_steps} steps, shift={sample_shift}...")
        logging.info(f"Prompt [{prompt_idx+1}/{len(prompts_list)}]: {prompt[:100]}...")
        latents = noise
        
        # FIX: Use bfloat16 like OmniVideo for faster inference
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=param_dtype):
            for t in tqdm(timesteps, desc=f"Sampling [{prompt_idx+1}/{len(prompts_list)}]"):
                timestep = torch.stack([t]).to(device)
                
                # Prepare forward kwargs to match training logic
                forward_kwargs = {
                    'x': latents,
                    't': timestep,
                    'context': context,  # T5 context (enabled with disable_t5_context=False)
                    'seq_len': seq_len,
                    'condition_mode': "full",
                    'gamma': args.gamma  # Gamma gating for clue/T5 mixing (0.0=T5-only, 1.0=full clue)
                }
                
                # OPTIMIZATION: Use pre-computed adapter output to skip adapter computation each step
                # This is the fastest path: adapter output already computed before denoising loop
                if precomputed_adapter_output_tensor is not None:
                    # Use pre-computed adapter output (fastest: skip adapter computation entirely)
                    forward_kwargs['precomputed_adapter_output'] = precomputed_adapter_output_tensor  # [1, 64, 4096] (K=64)
                    # Model will skip adapter computation and use this directly
                    if prompt_idx == 0 and t == timesteps[0]:
                        logging.info(f"🚀 Using pre-computed adapter output (shape: {precomputed_adapter_output_tensor.shape}) - SKIPPING adapter computation!")
                elif precomputed_vision_input is not None:
                    # Fallback: Use pre-computed vision input (still need to compute adapter each step)
                    forward_kwargs['ar_vision_input'] = [precomputed_vision_input]  # List format: [tensor]
                    # Model detects ar_vision_input and uses pre-computed path automatically
                else:
                    # Fallback: on-the-fly encoding (slowest: encodes every step)
                    forward_kwargs['prompts'] = [prompt]
                
                noise_pred = model(**forward_kwargs)
                
                if isinstance(noise_pred, list):
                    noise_pred = noise_pred[0]
                
                # FIX: FlowUniPCMultistepScheduler.step() signature: step(model_output, timestep, sample, return_dict=False, generator=None)
                # Returns: tuple (prev_sample, ...) if return_dict=False, lấy [0]
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),  # model_output
                    t,  # timestep
                    latents[0].unsqueeze(0),  # sample
                    return_dict=False,
                    generator=seed_g
                )[0]  # FlowUniPC returns tuple, take first element
                latents = [temp_x0.squeeze(0)]
        
        # Decode with VAE
        logging.info(f"[{prompt_idx+1}/{len(prompts_list)}] Decoding video with VAE...")
        x0 = latents
        videos = vae.decode(x0)
        video_tensor = videos[0]  # (C, N, H, W)
        logging.info(f"[{prompt_idx+1}/{len(prompts_list)}] Video decoded: shape {video_tensor.shape}")
        
        # Save video with descriptive name
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')[:50]
        safe_video_name = "".join(c for c in meta['video_name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30] if meta['video_name'] else ''
        
        if args.csv_path:
            # Use training data metadata for naming
            video_path = os.path.join(
                args.output_dir,
                f"train_sample_{meta['index']:03d}_{safe_video_name}_{safe_prompt[:30]}_seed{seed}.mp4"
            )
        else:
            # Use original naming
            video_path = os.path.join(
                args.output_dir,
                f"mobile_ov_{args.size}_seed{seed}_steps{args.sample_steps}_"
                f"frames{args.frame_num}_{safe_prompt}_{timestamp}.mp4"
            )
        
        save_video(video_tensor, video_path, fps=8)
        logging.info(f"✅ Generation complete! Video saved to: {video_path}")
        
        # Extract frames if requested
        if args.extract_frames:
            frames_dir = os.path.join(args.output_dir, f"frames_{safe_prompt}_{timestamp}_{prompt_idx}")
            extract_frames(video_path, frames_dir, max_frames=args.max_frames)
            logging.info(f"✅ Frames extracted to: {frames_dir}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"✅ All {len(prompts_list)} video(s) generated successfully!")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"{'='*80}")


if __name__ == "__main__":
    main()
