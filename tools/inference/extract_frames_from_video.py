#!/usr/bin/env python3
"""
Extract frames from a video file.

Usage:
    python tools/inference/extract_frames_from_video.py \
        --video_path output/t5_only_baseline_test/t5_only_baseline_832*480_seed42_steps50_frames81_a_cat_playing_with_a_wool_beside_the_fireside_20260117_232538.mp4 \
        --output_dir output/t5_only_baseline_test/frames \
        --max_frames 20
"""

import sys
import os
import argparse
import cv2
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_frames(video_path, output_dir, max_frames=None, frame_interval=1):
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract (None = extract all)
        frame_interval: Extract every Nth frame (1 = all frames)
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties:")
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  Duration: {total_frames/fps:.2f}s")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    logger.info(f"Extracting frames (interval={frame_interval}, max={max_frames})...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame if it matches interval
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB for saving
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)  # cv2.imwrite expects BGR
            saved_count += 1
            
            if saved_count % 10 == 0:
                logger.info(f"  Saved {saved_count} frames...")
        
        frame_count += 1
        
        # Stop if max_frames reached
        if max_frames is not None and saved_count >= max_frames:
            break
    
    cap.release()
    
    logger.info(f"✅ Extracted {saved_count} frames to {output_dir}")
    logger.info(f"   Total frames in video: {total_frames}")
    logger.info(f"   Frames extracted: {saved_count}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for frames (default: video_dir/frames)")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to extract (default: all)")
    parser.add_argument("--frame_interval", type=int, default=1, help="Extract every Nth frame (default: 1 = all frames)")
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        video_dir = os.path.dirname(args.video_path)
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        args.output_dir = os.path.join(video_dir, f"{video_name}_frames")
    
    extract_frames(args.video_path, args.output_dir, args.max_frames, args.frame_interval)


if __name__ == "__main__":
    main()
