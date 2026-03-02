#!/usr/bin/env python3
"""
Script to generate dummy data for testing MobileOVModel training.
Creates 5-10 pickle files with the correct format for OmniVideoDataset.
"""

import os
import pickle
import torch

def create_dummy_dataset(output_dir, num_samples=8, frames=8, height=32, width=32, emb_dim=4096, hidden_dim=1152):
    """
    Creates a dummy dataset of pickle files for testing the OmniVideoDataset.
    
    Args:
        output_dir (str): Directory to save the dummy pickle files
        num_samples (int): Number of samples to generate
        frames (int): Number of frames in each video
        height (int): Height of each frame
        width (int): Width of each frame
        emb_dim (int): Dimension of text embeddings
        hidden_dim (int): Hidden dimension for text features
        
    Returns:
        str: Path to the text file containing paths to the generated pickle files
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    
    for i in range(num_samples):
        # Create dummy data matching OmniVideoDataset format
        dummy_data = {
            # T5 encoder typically outputs a list of tensors
            'text_emb': torch.randn(200, emb_dim),  # [seq_len, emb_dim]
            'aligned_emb': torch.randn(1, hidden_dim),  # [1, hidden_dim]
            'prompt': f"This is a dummy prompt for sample {i}: A beautiful scene with dynamic motion",
            'latent_feature': torch.randn(16, frames, height, width),  # [C, F, H, W]
            'vlm_last_hidden_states': torch.randn(1, 267, 4096),  # Optional, for backward compatibility
        }
        
        # Save to pickle file
        file_path = os.path.join(output_dir, f"dummy_sample_{i:04d}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        file_paths.append(file_path)
        print(f"  Created: {file_path}")
    
    # Create a text file with the file paths
    txt_file_path = os.path.join(output_dir, "file_paths.txt")
    with open(txt_file_path, 'w') as f:
        for path in file_paths:
            f.write(f"{path}\n")
    
    return txt_file_path

def main():
    # Configuration
    output_dir = "examples/finetune_data/dummy_t2v"
    num_samples = 8  # Between 5-10 as requested
    
    # Parameters matching typical OmniVideo data format
    frames = 8  # Small number for testing
    height = 32
    width = 32
    emb_dim = 4096  # T5 embedding dimension
    hidden_dim = 1152  # Adapter input dimension
    
    print(f"Generating {num_samples} dummy samples...")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create dummy dataset
    txt_file_path = create_dummy_dataset(
        output_dir=output_dir,
        num_samples=num_samples,
        frames=frames,
        height=height,
        width=width,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim
    )
    
    print()
    print(f"✓ Successfully created {num_samples} dummy samples!")
    print(f"✓ Data file list: {txt_file_path}")
    print()
    print("You can now use this path in your config:")
    print(f"  data_file: \"{txt_file_path}\"")

if __name__ == "__main__":
    main()
