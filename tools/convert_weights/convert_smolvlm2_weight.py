#!/usr/bin/env python3
"""
Convert SmolVLM2-500M weights from HuggingFace format to pure PyTorch checkpoint.

This script should be run ONCE to convert the model weights.
After conversion, the repo will not need transformers library.

Usage:
    conda activate smolvlm2  # env with transformers
    python tools/convert_weights/convert_smolvlm2_weight.py \
        --model-id HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --output-path omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt
"""

import argparse
import os
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_smolvlm2_weight(model_id: str, output_path: str, device: str = "cuda"):
    """
    Convert SmolVLM2 model from HuggingFace to pure PyTorch checkpoint.
    
    Args:
        model_id: HuggingFace model identifier
        output_path: Path to save the converted checkpoint
        device: Device to load model on
    """
    try:
        from transformers import AutoModel, AutoTokenizer, AutoModelForImageTextToText
    except ImportError:
        raise ImportError(
            "transformers library is required for conversion. "
            "Please install it: pip install transformers"
        )
    
    logger.info(f"Loading SmolVLM2 model from HuggingFace: {model_id}")
    
    # CRITICAL: Use AutoModelForImageTextToText instead of AutoModel
    # This ensures we get the full conditional generation model with lm_head
    # for unified model (text generation capability)
    logger.info("Loading as AutoModelForImageTextToText (with lm_head for text generation)...")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.float16,
        )
        logger.info("✅ Loaded as AutoModelForImageTextToText (has lm_head and generate method)")
        if hasattr(model, 'lm_head'):
            logger.info("  - ✅ Model has lm_head")
        if hasattr(model, 'generate'):
            logger.info("  - ✅ Model has generate() method")
    except Exception as e:
        logger.warning(f"Failed to load as AutoModelForImageTextToText: {e}")
        logger.info("Falling back to AutoModel (encoder-only, no text generation)...")
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.float16,
        )
        logger.warning("⚠️ Using AutoModel - model will NOT have text generation capability")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract state_dict and config
    state_dict = model.state_dict()
    config = model.config if hasattr(model, "config") else None
    
    # Convert config to dict (for safe serialization)
    config_dict = None
    if config is not None:
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif hasattr(config, "__dict__"):
            config_dict = config.__dict__
        else:
            # Fallback: try to serialize as dict
            try:
                import json
                config_dict = json.loads(json.dumps(config, default=str))
            except:
                logger.warning("Could not serialize config to dict, will save as object")
                config_dict = config
    
    # Save tokenizer vocab (safer than full object)
    tokenizer_vocab = None
    tokenizer_config = None
    if tokenizer is not None:
        try:
            # Save tokenizer vocab and config
            if hasattr(tokenizer, "get_vocab"):
                tokenizer_vocab = tokenizer.get_vocab()
            if hasattr(tokenizer, "init_kwargs"):
                tokenizer_config = tokenizer.init_kwargs
        except Exception as e:
            logger.warning(f"Could not extract tokenizer vocab: {e}")
    
    # Prepare checkpoint with state_dict (Option B1)
    checkpoint = {
        # Format 1: Safe (no transformers needed at runtime)
        "state_dict": state_dict,
        "config_dict": config_dict,
        "config": config,  # Save config object directly (for easy loading, no internet needed)
        "tokenizer_vocab": tokenizer_vocab,
        "tokenizer_config": tokenizer_config,
        
        # Format 2: Fallback (requires transformers, for backward compatibility)
        "model": model,
        "tokenizer": tokenizer,
        
        "model_id": model_id,
        "checkpoint_format": "state_dict",  # Mark format version
    }
    
    # Save checkpoint
    logger.info(f"Saving converted checkpoint to {output_path}")
    logger.info("  - Format: state_dict (safe, no transformers needed at runtime)")
    logger.info(f"  - State dict keys: {len(state_dict)}")
    torch.save(checkpoint, output_path)
    
    # Verify checkpoint and check for generation capability
    logger.info("Verifying checkpoint...")
    loaded_checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    
    if "state_dict" in loaded_checkpoint:
        logger.info("✓ Checkpoint saved successfully (state_dict format)")
        logger.info(f"  - State dict keys: {len(loaded_checkpoint['state_dict'])}")
        
        # Check if model has lm_head (for text generation)
        state_dict = loaded_checkpoint['state_dict']
        has_lm_head = any('lm_head' in k for k in state_dict.keys())
        if has_lm_head:
            lm_head_keys = [k for k in state_dict.keys() if 'lm_head' in k]
            logger.info(f"  - ✅ lm_head found in state_dict (text generation enabled)")
            logger.info(f"    - lm_head keys: {lm_head_keys[:3]}...")
        else:
            logger.warning("  - ⚠️ No lm_head in state_dict (text generation NOT available)")
            logger.warning("    - This checkpoint is encoder-only (SmolVLMModel)")
            logger.warning("    - For unified model with text generation, use AutoModelForImageTextToText")
        
        if "config" in loaded_checkpoint:
            logger.info(f"  - Config object included: ✓")
        if "config_dict" in loaded_checkpoint:
            logger.info(f"  - Config dict included: ✓")
        if "tokenizer_vocab" in loaded_checkpoint:
            logger.info(f"  - Tokenizer vocab included: ✓")
        if "model" in loaded_checkpoint:
            logger.info(f"  - Fallback model object included: ✓")
            # Check fallback model
            fallback_model = loaded_checkpoint['model']
            if hasattr(fallback_model, 'lm_head'):
                logger.info("  - ✅ Fallback model has lm_head")
            if hasattr(fallback_model, 'generate'):
                logger.info("  - ✅ Fallback model has generate() method")
    elif "model" in loaded_checkpoint:
        logger.info("✓ Checkpoint saved successfully (fallback format)")
        logger.info(f"  - Model type: {type(loaded_checkpoint['model'])}")
        fallback_model = loaded_checkpoint['model']
        if hasattr(fallback_model, 'lm_head'):
            logger.info("  - ✅ Model has lm_head (text generation enabled)")
        else:
            logger.warning("  - ⚠️ Model does NOT have lm_head (text generation NOT available)")
        if hasattr(fallback_model, 'generate'):
            logger.info("  - ✅ Model has generate() method")
    else:
        logger.warning("⚠ Checkpoint format may be incorrect")
    
    # Get model size
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Checkpoint size: {model_size_mb:.2f} MB")
    
    logger.info("Conversion completed successfully!")
    logger.info(f"You can now use this checkpoint with nets.smolvlm2 without transformers library.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SmolVLM2-500M weights from HuggingFace to pure PyTorch"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the converted checkpoint (.pt file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load model on (cuda or cpu)",
    )
    
    args = parser.parse_args()
    
    convert_smolvlm2_weight(
        model_id=args.model_id,
        output_path=args.output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()


