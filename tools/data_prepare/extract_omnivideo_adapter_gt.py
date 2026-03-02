#!/usr/bin/env python3
"""
Extract OmniVideo adapter output as groundtruth for MobileOV training.

This script:
1. Loads OmniVideo model (AR VisionHead + Adapter)
2. For each pkl file in dataset:
   - Loads vlm_last_hidden_states
   - Passes through AR VisionHead → Adapter
   - Saves adapter output as groundtruth
3. Updates pkl files with new key: 'adapter_output_gt'

Usage:
    python tools/data_prepare/extract_omnivideo_adapter_gt.py \
        --dataset_file_path data/openvid_subset/file_paths.txt \
        --wan_ckpt_dir omni_ckpts/wan/wanxiang1_3b \
        --adapter_ckpt_dir omni_ckpts/adapter \
        --vision_head_ckpt_dir omni_ckpts/vision_head \
        --output_dir data/openvid_subset_with_gt \
        --device 0 \
        --batch_size 8
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import argparse
import logging
import pickle as pkl
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import shutil

from nets.omni.modules.omni_video_model import OmniVideoMixedConditionModel
from nets.third_party.wan.configs import WAN_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_omnivideo_model(wan_ckpt_dir, adapter_ckpt_dir, vision_head_ckpt_dir, device_id, precision_dtype=torch.bfloat16):
    """Load OmniVideo model with AR VisionHead and Adapter."""
    logger.info("Loading OmniVideo model...")
    logger.info(f"  WAN checkpoint: {wan_ckpt_dir}")
    
    # Handle adapter checkpoint path (can be directory or file)
    if os.path.isdir(adapter_ckpt_dir):
        # If directory, look for adapter_pytorch_model.bin
        adapter_file = os.path.join(adapter_ckpt_dir, "adapter_pytorch_model.bin")
        if os.path.exists(adapter_file):
            adapter_ckpt_dir = adapter_file
        else:
            # Try to use directory as-is (adapter will handle it)
            pass
    logger.info(f"  Adapter checkpoint: {adapter_ckpt_dir}")
    logger.info(f"  VisionHead checkpoint: {vision_head_ckpt_dir}")
    
    model = OmniVideoMixedConditionModel.from_pretrained(
        wan_ckpt_dir=wan_ckpt_dir,
        adapter_ckpt_dir=adapter_ckpt_dir,
        vision_head_ckpt_dir=vision_head_ckpt_dir,
        learnable_query_length=4,  # OmniVideo default
        adapter_in_channels=1152,
        adapter_out_channels=4096,
        adapter_query_length=256,  # Match checkpoint (will truncate to 64 later if needed)
        precision_dtype=precision_dtype,
        device_id=device_id,  # device_id should be int, not device object
        rank=0,
        dit_fsdp=False,
        use_usp=False,
        use_visual_context_adapter=False,
        max_context_len=None,
    )
    
    model.eval()
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("✅ OmniVideo model loaded successfully")
    
    return model


def extract_adapter_output(model, aligned_emb=None, vlm_hidden_states=None, device_id=0, precision_dtype=torch.bfloat16):
    """
    Extract adapter output from OmniVideo model.
    
    Args:
        model: OmniVideoMixedConditionModel instance
        aligned_emb: Optional Tensor of shape [1, 1152] (pre-computed VisionHead output) - PREFERRED
        vlm_hidden_states: Optional Tensor of shape [1, L, D] or [L, D] (VLM hidden states) - requires VisionHead
        device_id: int, GPU device ID
        precision_dtype: torch.dtype for computation
        
    Returns:
        adapter_output: Tensor of shape [K, 4096] where K=64 (adapter_query_length)
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=precision_dtype):
        # Prefer aligned_emb (pre-computed VisionHead output) if available
        if aligned_emb is not None:
            # aligned_emb is already VisionHead output: [1, 1152] or [1152]
            if isinstance(aligned_emb, torch.Tensor):
                ar_vision_input = aligned_emb.to(device=device, dtype=precision_dtype)
            else:
                ar_vision_input = torch.tensor(aligned_emb, device=device, dtype=precision_dtype)
            
            # Ensure correct shape: [1, 1152]
            if ar_vision_input.dim() == 1:
                ar_vision_input = ar_vision_input.unsqueeze(0)  # [1152] -> [1, 1152]
            elif ar_vision_input.dim() == 2:
                if ar_vision_input.shape[0] > 1:
                    # Take first sample if batch > 1
                    ar_vision_input = ar_vision_input[0:1]  # [1, 1152]
            else:
                raise ValueError(f"Unexpected aligned_emb shape: {ar_vision_input.shape}")
            
            # Reshape to [1, 1, 1152] for adapter input
            ar_vision_input = ar_vision_input.unsqueeze(1)  # [1, 1152] -> [1, 1, 1152]
            
        elif vlm_hidden_states is not None:
            # Fallback: use vlm_hidden_states and pass through VisionHead
            if model.ar_vision_head is None:
                raise ValueError("AR VisionHead is None and aligned_emb not provided! Cannot extract adapter output.")
            
            # Ensure vlm_hidden_states is on device and correct dtype
            if isinstance(vlm_hidden_states, torch.Tensor):
                vlm_hidden_states = vlm_hidden_states.to(device=device, dtype=precision_dtype)
            else:
                vlm_hidden_states = torch.tensor(vlm_hidden_states, device=device, dtype=precision_dtype)
            
            # Ensure correct shape: [1, L, D] or [L, D]
            if vlm_hidden_states.dim() == 2:
                vlm_hidden_states = vlm_hidden_states.unsqueeze(0)  # [L, D] -> [1, L, D]
            elif vlm_hidden_states.dim() == 3:
                if vlm_hidden_states.shape[0] > 1:
                    # Take first sample if batch > 1
                    vlm_hidden_states = vlm_hidden_states[0:1]
            else:
                raise ValueError(f"Unexpected vlm_hidden_states shape: {vlm_hidden_states.shape}")
            
            # Pass through AR VisionHead
            ar_vision_output = model.ar_vision_head(vlm_hidden_states)  # [1, 4, 1152] or [1, Q, 1152]
            
            # For T2V tasks, take first frame's vision output
            # OmniVideo does: ar_vision_output[:, 0:1, :] for T2V
            if ar_vision_output.shape[1] > 1:
                ar_vision_output = ar_vision_output[:, 0:1, :]  # [1, 1, 1152]
            
            ar_vision_input = ar_vision_output
        else:
            raise ValueError("Either aligned_emb or vlm_hidden_states must be provided!")
        
        # Pass through Adapter
        adapter_output = model.adapter(ar_vision_input)  # [1, 1, 1152] -> [1, 256, 4096] (or [1, 64, 4096])
        
        # Extract single sample: [1, K, 4096] -> [K, 4096]
        if adapter_output.dim() == 3:
            adapter_output = adapter_output[0]  # [K, 4096] where K=256 or 64
        elif adapter_output.dim() == 2:
            adapter_output = adapter_output  # Already [K, 4096]
        else:
            raise ValueError(f"Unexpected adapter_output shape: {adapter_output.shape}")
        
        # Truncate to 64 tokens to match OmniVideo's effective context (if needed)
        # OmniVideo uses adapter_query_length=256 but often takes only [0] or truncates to 64
        if adapter_output.shape[0] > 64:
            adapter_output = adapter_output[:64, :]  # Truncate to 64 tokens
        
        # Convert to CPU and float32 for storage
        adapter_output = adapter_output.cpu().float()
        
        return adapter_output


def process_pkl_file(pkl_path, model, device_id, precision_dtype, output_dir=None):
    """
    Process a single pkl file to extract adapter groundtruth.
    
    Args:
        pkl_path: Path to input pkl file
        model: OmniVideoMixedConditionModel instance
        device_id: int, GPU device ID
        precision_dtype: torch.dtype
        output_dir: Optional output directory (if None, overwrites original)
        
    Returns:
        success: bool
        error_msg: str or None
    """
    try:
        # Load pkl file
        with open(pkl_path, 'rb') as f:
            data = pkl.load(f)
        
        # Prefer aligned_emb (pre-computed VisionHead output) if available
        # Otherwise use vlm_last_hidden_states (requires VisionHead)
        aligned_emb = data.get('aligned_emb', None)
        vlm_hidden_states = data.get('vlm_last_hidden_states', None)
        
        if aligned_emb is None and vlm_hidden_states is None:
            return False, f"Missing both 'aligned_emb' and 'vlm_last_hidden_states' in {pkl_path}"
        
        # Extract adapter output
        # Prefer aligned_emb (doesn't require VisionHead)
        adapter_output_gt = extract_adapter_output(
            model, 
            aligned_emb=aligned_emb,
            vlm_hidden_states=vlm_hidden_states,
            device_id=device_id, 
            precision_dtype=precision_dtype
        )
        
        # Add to data dict
        data['adapter_output_gt'] = adapter_output_gt
        
        # Save updated pkl file
        if output_dir is not None:
            # Save to output directory
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(pkl_path))
        else:
            # Overwrite original
            output_path = pkl_path
        
        with open(output_path, 'wb') as f:
            pkl.dump(data, f)
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Extract OmniVideo adapter output as groundtruth")
    parser.add_argument("--dataset_file_path", type=str, required=True,
                        help="Path to text file listing pkl file paths")
    parser.add_argument("--wan_ckpt_dir", type=str, required=True,
                        help="Path to WAN checkpoint directory")
    parser.add_argument("--adapter_ckpt_dir", type=str, required=True,
                        help="Path to Adapter checkpoint directory")
    parser.add_argument("--vision_head_ckpt_dir", type=str, required=True,
                        help="Path to VisionHead checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for updated pkl files (if None, overwrites original)")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing (currently processes one by one)")
    parser.add_argument("--precision", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Precision dtype for computation")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set precision dtype
    precision_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    precision_dtype = precision_map[args.precision]
    logger.info(f"Using precision: {precision_dtype}")
    
    # Load OmniVideo model
    # Convert device to device_id (int)
    device_id = args.device if isinstance(args.device, int) else (device.index if hasattr(device, 'index') else 0)
    model = load_omnivideo_model(
        wan_ckpt_dir=args.wan_ckpt_dir,
        adapter_ckpt_dir=args.adapter_ckpt_dir,
        vision_head_ckpt_dir=args.vision_head_ckpt_dir if args.vision_head_ckpt_dir != "None" else None,
        device_id=device_id,
        precision_dtype=precision_dtype
    )
    
    # Load dataset file paths
    logger.info(f"Loading dataset file paths from {args.dataset_file_path}")
    with open(args.dataset_file_path, 'r') as f:
        pkl_paths = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Found {len(pkl_paths)} pkl files to process")
    
    # Create output directory if specified
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
        # Also create file_paths.txt in output directory
        output_file_paths_txt = os.path.join(args.output_dir, "file_paths.txt")
    else:
        logger.info("Will overwrite original pkl files")
        output_file_paths_txt = None
    
    # Process each pkl file
    success_count = 0
    error_count = 0
    output_paths = []
    
    logger.info("Processing pkl files...")
    for pkl_path in tqdm(pkl_paths, desc="Extracting adapter GT"):
        if not os.path.exists(pkl_path):
            logger.warning(f"File not found: {pkl_path}")
            error_count += 1
            continue
        
        # Determine output path
        if args.output_dir is not None:
            output_path = os.path.join(args.output_dir, os.path.basename(pkl_path))
        else:
            output_path = pkl_path
        
        # Process file
        success, error_msg = process_pkl_file(
            pkl_path=pkl_path,
            model=model,
            device_id=device_id,
            precision_dtype=precision_dtype,
            output_dir=args.output_dir
        )
        
        if success:
            success_count += 1
            output_paths.append(output_path)
        else:
            error_count += 1
            logger.warning(f"Failed to process {pkl_path}: {error_msg}")
    
    # Save output file_paths.txt if output directory specified
    if output_file_paths_txt is not None and output_paths:
        with open(output_file_paths_txt, 'w') as f:
            for path in output_paths:
                f.write(f"{path}\n")
        logger.info(f"Saved output file paths to {output_file_paths_txt}")
    
    # Summary
    logger.info("="*80)
    logger.info("EXTRACTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total files: {len(pkl_paths)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Success rate: {success_count/len(pkl_paths)*100:.1f}%")
    logger.info("="*80)
    
    if success_count > 0:
        logger.info("✅ Adapter groundtruth extraction complete!")
        logger.info(f"   Updated pkl files now contain 'adapter_output_gt' key")
        logger.info(f"   Shape: [64, 4096] (adapter_query_length=64, adapter_out_channels=4096)")
        if args.output_dir is not None:
            logger.info(f"   Output directory: {args.output_dir}")
    else:
        logger.error("❌ No files were processed successfully!")


if __name__ == "__main__":
    main()
