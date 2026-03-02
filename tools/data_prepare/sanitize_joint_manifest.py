#!/usr/bin/env python3
"""
Sanitize a joint image+video manifest for iv-joint training.

Input schema (expected):
  video, caption, preprocessed_path, video_path, dataset, modality, sample_idx

Typical usage:
  python tools/data_prepare/sanitize_joint_manifest.py \
    --input-csv data/mix/manifests/joint_msrvtt_laion_coyo_ivjoint_ready.csv \
    --output-csv data/mix/manifests/joint_msrvtt_coyo_ivjoint_clean_v1.csv \
    --image-datasets coyo_700m \
    --min-image-bytes 2000 \
    --dedup-image-path
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Set

import pandas as pd


REQUIRED_COLUMNS = {
    "video",
    "caption",
    "preprocessed_path",
    "video_path",
    "dataset",
    "modality",
    "sample_idx",
}


def _parse_set(raw: Optional[str]) -> Optional[Set[str]]:
    if raw is None:
        return None
    vals = {x.strip().lower() for x in str(raw).split(",") if x.strip()}
    return vals if vals else None


def _norm_modality_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["modality"] = out["modality"].astype(str).str.strip().str.lower()
    out["dataset"] = out["dataset"].astype(str).str.strip().str.lower()
    out["caption"] = out["caption"].fillna("").astype(str).str.strip()
    out["video_path"] = out["video_path"].fillna("").astype(str).str.strip()
    out["preprocessed_path"] = out["preprocessed_path"].fillna("").astype(str).str.strip()
    return out


def _exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _size_or_neg(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="data/mix/manifests/joint_msrvtt_laion_coyo_ivjoint_ready.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="data/mix/manifests/joint_msrvtt_coyo_ivjoint_clean_v1.csv",
    )
    parser.add_argument(
        "--image-datasets",
        default="coyo_700m",
        help="Comma-separated image datasets to keep (e.g. coyo_700m,laion). Empty keeps all.",
    )
    parser.add_argument(
        "--min-image-bytes",
        type=int,
        default=2000,
        help="Drop image rows whose raw image file is smaller than this threshold.",
    )
    parser.add_argument(
        "--dedup-image-path",
        action="store_true",
        help="Keep first row per image video_path for image modality.",
    )
    parser.add_argument(
        "--drop-missing-preprocessed",
        action="store_true",
        help="Drop rows whose preprocessed_path does not exist.",
    )
    parser.add_argument(
        "--drop-empty-caption",
        action="store_true",
        help="Drop rows with empty caption.",
    )
    parser.add_argument(
        "--reindex-sample-idx",
        action="store_true",
        help="Rewrite sample_idx to contiguous global IDs [0..N-1] after filtering.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    missing_cols = sorted(list(REQUIRED_COLUMNS - set(df.columns)))
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns: {missing_cols}")

    df = _norm_modality_col(df)
    image_datasets = _parse_set(args.image_datasets)

    video_df = df[df["modality"] == "video"].copy()
    image_df = df[df["modality"] == "image"].copy()

    # Optional image dataset filter.
    if image_datasets is not None:
        image_df = image_df[image_df["dataset"].isin(image_datasets)].copy()

    # Drop missing preprocessed files.
    if args.drop_missing_preprocessed:
        video_df = video_df[video_df["preprocessed_path"].map(_exists)].copy()
        image_df = image_df[image_df["preprocessed_path"].map(_exists)].copy()

    # Drop tiny/malformed images.
    if args.min_image_bytes > 0:
        image_sizes = image_df["video_path"].map(_size_or_neg)
        image_df = image_df[image_sizes >= int(args.min_image_bytes)].copy()

    # Remove empty captions if requested.
    if args.drop_empty_caption:
        video_df = video_df[video_df["caption"].str.len() > 0].copy()
        image_df = image_df[image_df["caption"].str.len() > 0].copy()

    # Dedup image rows by raw image path.
    if args.dedup_image_path:
        image_df = image_df.drop_duplicates(subset=["video_path"], keep="first").copy()

    out_df = pd.concat([video_df, image_df], axis=0, ignore_index=True)
    if args.reindex_sample_idx:
        out_df["sample_idx"] = pd.Series(range(len(out_df)), dtype="int64")

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    # Summary.
    print(f"[done] input={args.input_csv}")
    print(f"[done] output={args.output_csv}")
    print(f"[done] total={len(out_df)}")
    print(f"[done] reindex_sample_idx={bool(args.reindex_sample_idx)}")
    print("[done] modality counts:")
    print(out_df["modality"].value_counts(dropna=False).to_string())
    print("[done] dataset counts:")
    print(out_df["dataset"].value_counts(dropna=False).to_string())
    if len(image_df) > 0:
        print(f"[done] image unique video_path={image_df['video_path'].nunique()}")
        print(f"[done] image duplicate video_path={int(image_df['video_path'].duplicated().sum())}")


if __name__ == "__main__":
    main()
