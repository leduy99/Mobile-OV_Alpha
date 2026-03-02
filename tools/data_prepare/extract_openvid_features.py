#!/usr/bin/env python3
"""
Extract VAE and T5 features from OpenVid-1M videos for training.

This script processes OpenVid-1M videos and extracts:
- Video latent features (using VAE)
- Text embeddings (using T5 encoder)

Usage:
    python tools/data_prepare/extract_openvid_features.py \
        --csv_path data/openvid/OpenVid-1M.csv \
        --video_dir data/openvid/videos \
        --output_dir data/openvid/preprocessed \
        --ckpt_dir omni_ckpts/wan \
        --frame_num 21 \
        --target_size 512,512 \
        --max_samples 1000
"""

import os
import sys
import argparse
import logging
import pickle
import pandas as pd
import torch
from pathlib import Path

# Add project root to path (repo root)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from nets.third_party.wan.text2video import T5EncoderModel
from nets.third_party.wan.configs import WAN_CONFIGS
from nets.third_party.wan.modules.vae import WanVAE
from tools.data_prepare.vae_feature_extract import read_video_frames

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_features_from_openvid(
    csv_path: str,
    video_dir: str,
    output_dir: str,
    ckpt_dir: str,
    frame_num: int = 21,
    target_size: tuple = (512, 512),
    sampling_rate: int = 1,
    skip_num: int = 0,
    max_samples: int = None,
    device: str = "cuda:0",
):
    """
    Extract features from OpenVid-1M videos.
    
    Args:
        csv_path: Path to OpenVid-1M.csv
        video_dir: Directory containing video files
        output_dir: Directory to save preprocessed features
        ckpt_dir: Directory containing WAN checkpoints
        frame_num: Number of frames to extract
        target_size: Target video size (height, width)
        sampling_rate: Frame sampling rate
        skip_num: Number of frames to skip at start
        max_samples: Maximum number of videos to process
        device: Device to use for processing
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    
    # Load CSV
    logger.info(f"Loading OpenVid-1M CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Load T5 encoder
    logger.info("Loading T5 encoder...")
    cfg = WAN_CONFIGS.get('t2v-1.3B', {})
    text_encoder = T5EncoderModel(
        text_len=cfg.get('text_len', 256),
        dtype=cfg.get('t5_dtype', torch.bfloat16),
        device=torch.device('cpu'),  # T5 on CPU to save GPU memory
        checkpoint_path=os.path.join(ckpt_dir, cfg.get('t5_checkpoint', 'models_t5_umt5-xxl-enc-bf16.pth')),
        tokenizer_path=os.path.join(ckpt_dir, cfg.get('t5_tokenizer', 'google/umt5-xxl')),
        shard_fn=None
    )
    
    # Load VAE
    logger.info("Loading VAE...")
    vae = WanVAE(
        vae_pth=os.path.join(ckpt_dir, cfg.get('vae_checkpoint', 'Wan2.1_VAE.pth')),
        device=device
    )
    
    # Process videos
    processed = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        if max_samples and processed >= max_samples:
            break
        
        video_name = row['video']
        caption = row['caption']
        
        # Find video file
        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            if not os.path.exists(video_path):
                skipped += 1
                if skipped % 100 == 0:
                    logger.warning(f"Skipped {skipped} videos (file not found)")
                continue
        
        # Check if already processed
        video_basename = os.path.splitext(video_name)[0]
        output_path = os.path.join(output_dir, f"{video_basename}_features.pkl")
        if os.path.exists(output_path):
            processed += 1
            if processed % 100 == 0:
                logger.info(f"Already processed: {processed} videos")
            continue
        
        try:
            # Read video frames
            frames_tensor = read_video_frames(
                video_path, 
                frame_num, 
                sampling_rate, 
                skip_num, 
                target_size
            )
            if frames_tensor is None:
                skipped += 1
                continue
            
            # Encode text
            if not caption or len(caption) < 10:
                skipped += 1
                continue
            
            with torch.no_grad():
                text_emb = text_encoder([caption], torch.device('cpu'))
                text_emb = [t.cpu() for t in text_emb]
            
            # Extract VAE features
            with torch.no_grad():
                frames_tensor = frames_tensor.to(device)
                latent_feature = vae.encode(frames_tensor.transpose(0, 1).unsqueeze(0))
                latent_feature = latent_feature[0].cpu()
            
            # Save features
            data = {
                'latent_feature': latent_feature,
                'prompt': caption,
                'text_emb': text_emb,
                'video_path': video_path,
                'frame_num': frame_num,
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {processed} videos, skipped {skipped}")
                
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            skipped += 1
            continue
    
    logger.info(f"Feature extraction completed!")
    logger.info(f"Processed: {processed}, Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Extract features from OpenVid-1M videos")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to OpenVid-1M.csv")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for preprocessed features")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing WAN checkpoints")
    parser.add_argument("--frame_num", type=int, default=21, help="Number of frames to extract")
    parser.add_argument("--target_size", type=str, default="512,512", help="Target size (height,width)")
    parser.add_argument("--sampling_rate", type=int, default=1, help="Frame sampling rate")
    parser.add_argument("--skip_num", type=int, default=0, help="Frames to skip at start")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    target_size = tuple(int(x) for x in args.target_size.split(','))
    
    extract_features_from_openvid(
        csv_path=args.csv_path,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        ckpt_dir=args.ckpt_dir,
        frame_num=args.frame_num,
        target_size=target_size,
        sampling_rate=args.sampling_rate,
        skip_num=args.skip_num,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
