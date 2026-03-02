#!/usr/bin/env python3
"""
Extract adapter and projection layer from DeepSpeed training checkpoint.

Usage:
    python extract_trained_checkpoint.py \
        --checkpoint_dir output/test_training/checkpoint_epoch_0_step_3/epoch_0_step_3 \
        --output_dir output/trained_components
"""

import argparse
import os
import sys
import torch
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_from_deepspeed_checkpoint(checkpoint_dir, output_dir):
    """Extract adapter and projection from DeepSpeed checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # DeepSpeed checkpoint structure: {'module': {...}}
    if 'module' not in checkpoint:
        logger.error("Checkpoint does not have 'module' key")
        return False
    
    model_state = checkpoint['module']
    
    # Extract adapter weights
    # Handle both 'adapter.xxx' and 'adapter.adapter.xxx' formats
    # Keep 'adapter.' prefix for compatibility with DM_Adapter.load_ckpt()
    adapter_keys = [k for k in model_state.keys() if k.startswith('adapter.')]
    if adapter_keys:
        logger.info(f"Found {len(adapter_keys)} adapter keys")
        adapter_state = {}
        for k, v in model_state.items():
            if k.startswith('adapter.'):
                # Remove first 'adapter.' prefix, keep the rest (which should start with 'adapter.')
                new_key = k.replace('adapter.', '', 1)  # Remove first occurrence
                # The result should be like 'adapter.encoder.xxx' or 'adapter.decoder.xxx'
                adapter_state[new_key] = v
        
        os.makedirs(os.path.join(output_dir, "adapter"), exist_ok=True)
        adapter_path = os.path.join(output_dir, "adapter", "adapter_pytorch_model.bin")
        torch.save(adapter_state, adapter_path)
        logger.info(f"✅ Saved adapter to {adapter_path}")
        logger.info(f"   Adapter has {len(adapter_state)} parameters")
    else:
        logger.warning("No adapter keys found in checkpoint")
    
    # Extract projection layer weights
    projection_keys = [k for k in model_state.keys() if k.startswith('smolvlm2_projection.')]
    if projection_keys:
        logger.info(f"Found {len(projection_keys)} projection keys")
        projection_state = {k.replace('smolvlm2_projection.', ''): v for k, v in model_state.items() if k.startswith('smolvlm2_projection.')}
        
        os.makedirs(os.path.join(output_dir, "smolvlm2_projection"), exist_ok=True)
        projection_path = os.path.join(output_dir, "smolvlm2_projection", "smolvlm2_projection_pytorch_model.bin")
        torch.save(projection_state, projection_path)
        logger.info(f"✅ Saved projection to {projection_path}")
        logger.info(f"   Projection has {len(projection_state)} parameters")
    else:
        logger.warning("No projection keys found in checkpoint")
    
    # List all keys for debugging
    logger.info("\nAll model keys:")
    for k in sorted(model_state.keys())[:20]:
        logger.info(f"  {k}")
    if len(model_state.keys()) > 20:
        logger.info(f"  ... and {len(model_state.keys()) - 20} more")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract trained components from DeepSpeed checkpoint")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to DeepSpeed checkpoint directory (epoch_X_step_Y)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for extracted components"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = extract_from_deepspeed_checkpoint(args.checkpoint_dir, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

