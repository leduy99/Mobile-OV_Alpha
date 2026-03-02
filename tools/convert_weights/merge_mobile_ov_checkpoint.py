#!/usr/bin/env python3
"""
Merge SmolVLM2 and Omni-Video checkpoints into a unified MobileOVModel checkpoint.

This script:
1. Loads SmolVLM2 checkpoint (converted .pt file)
2. Loads Omni-Video checkpoint (WAN + Adapter, skips VisionHead)
3. Initializes projection layer (random or from pretrained if available)
4. Saves unified checkpoint that can be loaded by MobileOVModel

Usage:
    python tools/convert_weights/merge_mobile_ov_checkpoint.py \
        --smolvlm2_ckpt path/to/smolvlm2_500m.pt \
        --omnivideo_ckpt_dir path/to/omnivideo_ckpt \
        --output_dir path/to/mobile_ov_ckpt \
        [--adapter_in_channels 1152] \
        [--adapter_out_channels 4096] \
        [--init_projection random|zeros|xavier]
"""

import argparse
import os
import sys
import logging
import torch
import torch.nn as nn

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from nets.smolvlm2 import load_smolvlm2_from_ckpt
from nets.third_party.wan.modules.model import WanModel
from nets.omni.modules.adapter import DM_Adapter
from nets.omni.modules.visual_context_adapter import VisualContextAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_smolvlm2_checkpoint(ckpt_path: str, device: str = "cpu"):
    """Load SmolVLM2 checkpoint and extract model state dict"""
    logger.info(f"Loading SmolVLM2 checkpoint from {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        else:
            raise ValueError("Checkpoint must contain 'model' key")
    else:
        model = checkpoint
    
    # Extract state dict
    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
    else:
        # If it's already a state dict
        state_dict = model
    
    # Get hidden size from model config
    hidden_size = None
    if hasattr(model, 'config') and hasattr(model.config, 'text_config'):
        hidden_size = model.config.text_config.hidden_size
    elif hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    
    logger.info(f"SmolVLM2 hidden_size: {hidden_size}")
    
    return state_dict, hidden_size


def load_omnivideo_checkpoint(
    ckpt_dir: str,
    adapter_in_channels: int = 1152,
    adapter_out_channels: int = 4096,
    adapter_query_length: int = 64,
    device: str = "cpu"
):
    """Load Omni-Video checkpoint (WAN + Adapter, skip VisionHead)"""
    logger.info(f"Loading Omni-Video checkpoint from {ckpt_dir}")
    
    # Load WAN model
    wan_model = WanModel.from_pretrained(ckpt_dir)
    wan_state_dict = wan_model.state_dict()
    
    # Load Adapter
    adapter_ckpt_path = os.path.join(ckpt_dir, "adapter", "adapter_pytorch_model.bin")
    if not os.path.exists(adapter_ckpt_path):
        # Try alternative path
        adapter_ckpt_path = os.path.join(ckpt_dir, "adapter_pytorch_model.bin")
    
    adapter_state_dict = None
    if os.path.exists(adapter_ckpt_path):
        logger.info(f"Loading adapter from {adapter_ckpt_path}")
        adapter_checkpoint = torch.load(adapter_ckpt_path, map_location=device)
        if isinstance(adapter_checkpoint, dict) and 'state_dict' in adapter_checkpoint:
            adapter_state_dict = adapter_checkpoint['state_dict']
        else:
            adapter_state_dict = adapter_checkpoint
    else:
        logger.warning(f"Adapter checkpoint not found at {adapter_ckpt_path}, will initialize randomly")
    
    # Load Visual Context Adapter (optional)
    visual_adapter_ckpt_path = os.path.join(ckpt_dir, "visual_context_adapter", "visual_context_adapter_pytorch_model.bin")
    if not os.path.exists(visual_adapter_ckpt_path):
        visual_adapter_ckpt_path = os.path.join(ckpt_dir, "visual_context_adapter_pytorch_model.bin")
    
    visual_adapter_state_dict = None
    if os.path.exists(visual_adapter_ckpt_path):
        logger.info(f"Loading visual context adapter from {visual_adapter_ckpt_path}")
        visual_adapter_checkpoint = torch.load(visual_adapter_ckpt_path, map_location=device)
        if isinstance(visual_adapter_checkpoint, dict) and 'state_dict' in visual_adapter_checkpoint:
            visual_adapter_state_dict = visual_adapter_checkpoint['state_dict']
        else:
            visual_adapter_state_dict = visual_adapter_checkpoint
    else:
        logger.info("Visual context adapter not found, will be None")
    
    return {
        'wan_model': wan_state_dict,
        'adapter': adapter_state_dict,
        'visual_context_adapter': visual_adapter_state_dict,
    }


def initialize_projection_layer(
    hidden_size: int,
    adapter_in_channels: int,
    init_method: str = "xavier"
):
    """Initialize projection layer with specified method"""
    projection = nn.Linear(hidden_size, adapter_in_channels, bias=False)
    
    if init_method == "random":
        nn.init.normal_(projection.weight, mean=0.0, std=0.02)
        logger.info("Initialized projection with random normal")
    elif init_method == "zeros":
        nn.init.zeros_(projection.weight)
        logger.info("Initialized projection with zeros")
    elif init_method == "xavier":
        nn.init.xavier_uniform_(projection.weight)
        logger.info("Initialized projection with xavier uniform")
    elif init_method == "kaiming":
        nn.init.kaiming_uniform_(projection.weight)
        logger.info("Initialized projection with kaiming uniform")
    else:
        raise ValueError(f"Unknown init method: {init_method}")
    
    return projection.state_dict()


def merge_checkpoints(
    smolvlm2_ckpt_path: str,
    omnivideo_ckpt_dir: str,
    output_dir: str,
    adapter_in_channels: int = 1152,
    adapter_out_channels: int = 4096,
    adapter_query_length: int = 64,
    init_projection: str = "xavier",
    device: str = "cpu",
):
    """
    Merge SmolVLM2 and Omni-Video checkpoints into unified MobileOVModel checkpoint.
    
    Args:
        smolvlm2_ckpt_path: Path to converted SmolVLM2 checkpoint (.pt file)
        omnivideo_ckpt_dir: Directory containing Omni-Video checkpoint (WAN + Adapter)
        output_dir: Output directory for merged checkpoint
        adapter_in_channels: Adapter input channels
        adapter_out_channels: Adapter output channels
        adapter_query_length: Adapter query length
        init_projection: Initialization method for projection layer
        device: Device to load checkpoints on
    """
    logger.info("=" * 60)
    logger.info("Merging SmolVLM2 and Omni-Video checkpoints")
    logger.info("=" * 60)
    
    # 1. Load SmolVLM2 checkpoint
    smolvlm2_state_dict, smolvlm2_hidden_size = load_smolvlm2_checkpoint(
        smolvlm2_ckpt_path, device
    )
    
    if smolvlm2_hidden_size is None:
        logger.warning("Could not detect SmolVLM2 hidden_size, using default 1024")
        smolvlm2_hidden_size = 1024
    
    # 2. Load Omni-Video checkpoint
    omnivideo_state_dicts = load_omnivideo_checkpoint(
        omnivideo_ckpt_dir,
        adapter_in_channels,
        adapter_out_channels,
        adapter_query_length,
        device
    )
    
    # 3. Initialize projection layer
    projection_state_dict = initialize_projection_layer(
        smolvlm2_hidden_size,
        adapter_in_channels,
        init_projection
    )
    
    # 4. Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Save SmolVLM2 model (save the full model object for easy loading)
    smolvlm2_save_path = os.path.join(output_dir, "smolvlm2_model.pt")
    logger.info(f"Saving SmolVLM2 model to {smolvlm2_save_path}")
    
    # Load full model to save it
    smolvlm2_model, _ = load_smolvlm2_from_ckpt(smolvlm2_ckpt_path, device=torch.device(device))
    torch.save({
        'model': smolvlm2_model,
        'hidden_size': smolvlm2_hidden_size,
    }, smolvlm2_save_path)
    
    # Save WAN model
    wan_save_dir = os.path.join(output_dir, "wan_model")
    os.makedirs(wan_save_dir, exist_ok=True)
    logger.info(f"Saving WAN model to {wan_save_dir}")
    
    # Create a temporary WAN model to use save_pretrained
    temp_wan = WanModel.from_pretrained(omnivideo_ckpt_dir)
    temp_wan.load_state_dict(omnivideo_state_dicts['wan_model'])
    temp_wan.save_pretrained(wan_save_dir)
    del temp_wan
    
    # Save Adapter
    if omnivideo_state_dicts['adapter'] is not None:
        adapter_save_dir = os.path.join(output_dir, "adapter")
        os.makedirs(adapter_save_dir, exist_ok=True)
        adapter_save_path = os.path.join(adapter_save_dir, "adapter_pytorch_model.bin")
        logger.info(f"Saving adapter to {adapter_save_path}")
        torch.save(omnivideo_state_dicts['adapter'], adapter_save_path)
    else:
        logger.warning("Adapter state dict is None, skipping save")
    
    # Save Visual Context Adapter (if exists)
    if omnivideo_state_dicts['visual_context_adapter'] is not None:
        visual_adapter_save_dir = os.path.join(output_dir, "visual_context_adapter")
        os.makedirs(visual_adapter_save_dir, exist_ok=True)
        visual_adapter_save_path = os.path.join(visual_adapter_save_dir, "visual_context_adapter_pytorch_model.bin")
        logger.info(f"Saving visual context adapter to {visual_adapter_save_path}")
        torch.save(omnivideo_state_dicts['visual_context_adapter'], visual_adapter_save_path)
    
    # Save projection layer
    projection_save_dir = os.path.join(output_dir, "smolvlm2_projection")
    os.makedirs(projection_save_dir, exist_ok=True)
    projection_save_path = os.path.join(projection_save_dir, "smolvlm2_projection_pytorch_model.bin")
    logger.info(f"Saving projection layer to {projection_save_path}")
    torch.save(projection_state_dict, projection_save_path)
    
    # Save metadata
    metadata = {
        'smolvlm2_hidden_size': smolvlm2_hidden_size,
        'adapter_in_channels': adapter_in_channels,
        'adapter_out_channels': adapter_out_channels,
        'adapter_query_length': adapter_query_length,
        'init_projection': init_projection,
        'smolvlm2_ckpt_path': smolvlm2_ckpt_path,
        'omnivideo_ckpt_dir': omnivideo_ckpt_dir,
    }
    
    import json
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("=" * 60)
    logger.info("Checkpoint merge completed successfully!")
    logger.info(f"Unified checkpoint saved to: {output_dir}")
    logger.info("=" * 60)
    logger.info("\nTo load this checkpoint, use:")
    logger.info(f"  model = MobileOVModel.from_pretrained(")
    logger.info(f"      wan_ckpt_dir='{wan_save_dir}',")
    logger.info(f"      adapter_ckpt_dir='{adapter_save_dir if omnivideo_state_dicts['adapter'] else None}',")
    logger.info(f"      smolvlm2_ckpt_path='{smolvlm2_save_path}',")
    logger.info(f"      adapter_in_channels={adapter_in_channels},")
    logger.info(f"      adapter_out_channels={adapter_out_channels},")
    logger.info(f"      adapter_query_length={adapter_query_length},")
    logger.info(f"  )")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge SmolVLM2 and Omni-Video checkpoints into unified MobileOVModel checkpoint"
    )
    
    parser.add_argument(
        "--smolvlm2_ckpt",
        type=str,
        required=True,
        help="Path to converted SmolVLM2 checkpoint (.pt file)"
    )
    
    parser.add_argument(
        "--omnivideo_ckpt_dir",
        type=str,
        required=True,
        help="Directory containing Omni-Video checkpoint (should contain wan_model/ and adapter/)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged checkpoint"
    )
    
    parser.add_argument(
        "--adapter_in_channels",
        type=int,
        default=1152,
        help="Adapter input channels (default: 1152)"
    )
    
    parser.add_argument(
        "--adapter_out_channels",
        type=int,
        default=4096,
        help="Adapter output channels (default: 4096)"
    )
    
    parser.add_argument(
        "--adapter_query_length",
        type=int,
        default=64,
        help="Adapter query length (default: 64)"
    )
    
    parser.add_argument(
        "--init_projection",
        type=str,
        default="xavier",
        choices=["random", "zeros", "xavier", "kaiming"],
        help="Initialization method for projection layer (default: xavier)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load checkpoints on (default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.smolvlm2_ckpt):
        raise FileNotFoundError(f"SmolVLM2 checkpoint not found: {args.smolvlm2_ckpt}")
    
    if not os.path.exists(args.omnivideo_ckpt_dir):
        raise FileNotFoundError(f"Omni-Video checkpoint directory not found: {args.omnivideo_ckpt_dir}")
    
    # Merge checkpoints
    merge_checkpoints(
        smolvlm2_ckpt_path=args.smolvlm2_ckpt,
        omnivideo_ckpt_dir=args.omnivideo_ckpt_dir,
        output_dir=args.output_dir,
        adapter_in_channels=args.adapter_in_channels,
        adapter_out_channels=args.adapter_out_channels,
        adapter_query_length=args.adapter_query_length,
        init_projection=args.init_projection,
        device=args.device,
    )


if __name__ == "__main__":
    main()



