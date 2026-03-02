#!/usr/bin/env python3
"""
Extract frames from video file for inspection.

Usage:
    python extract_frames.py --video_path <video_path> --output_dir <output_dir>
"""

import argparse
import os
import sys
import cv2

def extract_frames(video_path, output_dir, frame_prefix="frame"):
    """
    Extract all frames from a video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_prefix: Prefix for frame filenames
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {frame_count}")
    print(f"  Duration: {frame_count/fps:.2f} seconds")
    print(f"\nExtracting frames to: {output_dir}")
    
    # Extract frames
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        frame_filename = os.path.join(output_dir, f"{frame_prefix}_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
        
        if frame_idx % 5 == 0:
            print(f"  Extracted {saved_count}/{frame_count} frames...", end='\r')
        
        frame_idx += 1
    
    cap.release()
    print(f"\n✅ Successfully extracted {saved_count} frames to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument(
        "--video_path", type=str, required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for frames (default: <video_name>_frames)"
    )
    parser.add_argument(
        "--frame_prefix", type=str, default="frame",
        help="Prefix for frame filenames (default: frame)"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        args.output_dir = os.path.join(
            os.path.dirname(args.video_path),
            f"{video_name}_frames"
        )
    
    success = extract_frames(args.video_path, args.output_dir, args.frame_prefix)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

