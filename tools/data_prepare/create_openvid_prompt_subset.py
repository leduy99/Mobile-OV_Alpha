#!/usr/bin/env python3
"""
Create a prompt-only subset from OpenVid-1M CSV for Q1 distillation.
"""

import argparse
import logging
import os
import random

import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Create OpenVid prompt-only subset")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to OpenVid-1M.csv")
    parser.add_argument("--output-dir", type=str, default="data/openvid_q1", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of prompts to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before sampling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading CSV: %s", args.csv_path)
    df = pd.read_csv(args.csv_path)
    if "caption" not in df.columns:
        raise ValueError("CSV missing 'caption' column")

    captions = df["caption"].dropna().astype(str).tolist()
    if args.shuffle:
        random.Random(args.seed).shuffle(captions)

    if args.num_samples:
        captions = captions[: args.num_samples]

    subset_csv = os.path.join(args.output_dir, "OpenVid_prompt_subset.csv")
    subset_txt = os.path.join(args.output_dir, "prompts.txt")

    pd.DataFrame({"caption": captions}).to_csv(subset_csv, index=False)
    with open(subset_txt, "w", encoding="utf-8") as f:
        for caption in captions:
            f.write(caption.strip().replace("\n", " ") + "\n")

    logger.info("Saved subset CSV: %s", subset_csv)
    logger.info("Saved prompts list: %s", subset_txt)
    logger.info("Total prompts: %d", len(captions))


if __name__ == "__main__":
    main()
