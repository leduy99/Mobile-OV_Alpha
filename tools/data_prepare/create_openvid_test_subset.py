#!/usr/bin/env python3
"""
Create a small test subset from OpenVid-1M CSV for quick testing.

This script downloads only the CSV file and creates a small subset for testing.
No video files are downloaded.

Usage:
    python tools/data_prepare/create_openvid_test_subset.py \
        --output_dir data/openvid_test \
        --num_samples 100
"""

import os
import sys
import argparse
import logging
import subprocess
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_csv(output_dir: str) -> str:
    """Download OpenVid-1M.csv file."""
    csv_url = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv"
    csv_path = os.path.join(output_dir, "OpenVid-1M.csv")
    
    if os.path.exists(csv_path):
        logger.info(f"CSV file already exists: {csv_path}")
        return csv_path
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Downloading CSV from {csv_url}...")
    
    try:
        subprocess.run(['wget', '-c', csv_url, '-O', csv_path], check=True)
        logger.info(f"✓ Downloaded CSV to {csv_path}")
        return csv_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download CSV: {e}")
        return None
    except FileNotFoundError:
        logger.error("wget not found. Trying with requests...")
        try:
            import requests
            response = requests.get(csv_url, stream=True)
            with open(csv_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"✓ Downloaded CSV to {csv_path}")
            return csv_path
        except ImportError:
            logger.error("Neither wget nor requests available. Please install one of them.")
            return None


def create_test_subset(csv_path: str, output_path: str, num_samples: int = 100):
    """Create a small test subset from the full CSV."""
    logger.info(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Full dataset has {len(df)} samples")
    
    # Take first N samples for testing
    subset_df = df.head(num_samples).copy()
    
    # Save subset
    subset_df.to_csv(output_path, index=False)
    logger.info(f"✓ Created test subset with {len(subset_df)} samples: {output_path}")
    
    # Print some sample captions
    logger.info("\nSample captions from subset:")
    for idx, row in subset_df.head(5).iterrows():
        caption = row['caption'][:100] + "..." if len(row['caption']) > 100 else row['caption']
        logger.info(f"  [{idx}] {caption}")


def create_dummy_preprocessed_data(output_dir: str, csv_path: str, num_samples: int = 100):
    """
    Create dummy preprocessed pickle files for testing.
    This allows testing the training pipeline without actual video processing.
    """
    import pickle
    import torch
    
    logger.info("Creating dummy preprocessed data for testing...")
    
    # Load subset CSV
    df = pd.read_csv(csv_path)
    subset_df = df.head(num_samples)
    
    preprocessed_dir = os.path.join(output_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    created = 0
    for idx, row in subset_df.iterrows():
        video_name = row['video']
        caption = row['caption']
        
        video_basename = os.path.splitext(video_name)[0]
        output_path = os.path.join(preprocessed_dir, f"{video_basename}_features.pkl")
        
        if os.path.exists(output_path):
            continue
        
        # Create dummy data with realistic shapes
        dummy_data = {
            'latent_feature': torch.randn(16, 21, 32, 32),  # [C, T, H, W] - VAE latent
            'prompt': caption,
            'text_emb': [torch.randn(1, 256, 4096)],  # T5 embeddings (list of tensors)
            'video_path': f"dummy/{video_name}",
            'frame_num': 21,
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        created += 1
        if created % 10 == 0:
            logger.info(f"Created {created} dummy preprocessed files...")
    
    logger.info(f"✓ Created {created} dummy preprocessed files in {preprocessed_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create test subset from OpenVid-1M")
    parser.add_argument("--output_dir", type=str, default="data/openvid_test", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for test subset")
    parser.add_argument("--create_dummy", action="store_true", help="Create dummy preprocessed data for testing")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Download CSV
    logger.info("=== Step 1: Downloading CSV ===")
    csv_path = download_csv(args.output_dir)
    if not csv_path:
        logger.error("Failed to download CSV. Exiting.")
        return
    
    # Step 2: Create test subset
    logger.info("\n=== Step 2: Creating test subset ===")
    subset_path = os.path.join(args.output_dir, "OpenVid-1M_test_subset.csv")
    create_test_subset(csv_path, subset_path, args.num_samples)
    
    # Step 3: Create dummy preprocessed data (optional)
    if args.create_dummy:
        logger.info("\n=== Step 3: Creating dummy preprocessed data ===")
        create_dummy_preprocessed_data(args.output_dir, subset_path, args.num_samples)
    
    logger.info("\n=== Setup Complete ===")
    logger.info(f"Test subset CSV: {subset_path}")
    logger.info(f"Number of samples: {args.num_samples}")
    if args.create_dummy:
        logger.info(f"Dummy preprocessed data: {os.path.join(args.output_dir, 'preprocessed')}")
    
    logger.info("\nNext step: Update config to use this test subset:")
    logger.info(f"  data_file: {subset_path}")
    logger.info(f"  video_dir: {args.output_dir}/videos  # (dummy, not needed if using preprocessed)")
    logger.info(f"  preprocessed_dir: {os.path.join(args.output_dir, 'preprocessed')}")


if __name__ == "__main__":
    main()

