#!/usr/bin/env python3
"""Build a joint OpenVid + merged-image manifest for 5v1i training.

This script is intentionally generic:
- video side comes from an OpenVid-style manifest with `caption` and `sample_idx`
- image side comes from a train-ready merged image manifest
- output schema matches OpenVidDataset direct-preprocessed mode
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


REQUIRED_OUTPUT_COLUMNS = [
    "video",
    "caption",
    "preprocessed_path",
    "video_path",
    "dataset",
    "modality",
    "sample_idx",
]


def _resolve_repo_relative(path_text: str, cwd: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (cwd / path).resolve()
    return str(path)


def _first_present(row: pd.Series, keys: List[str]) -> str:
    for key in keys:
        value = row.get(key)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _load_image_rows(image_manifest: Path, cwd: Path) -> pd.DataFrame:
    df = pd.read_csv(image_manifest).copy()
    missing = [c for c in REQUIRED_OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Image manifest missing required columns: {missing}")

    df["caption"] = df["caption"].fillna("").astype(str).str.strip()
    df = df[df["caption"] != ""].copy()
    df["modality"] = "image"
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce")
    df = df[df["sample_idx"].notna()].copy()
    df["sample_idx"] = df["sample_idx"].astype(int)
    df["preprocessed_path"] = df["preprocessed_path"].fillna("").astype(str).map(lambda p: _resolve_repo_relative(p, cwd))
    df["video_path"] = df["video_path"].fillna("").astype(str).map(lambda p: _resolve_repo_relative(p, cwd))
    return df[REQUIRED_OUTPUT_COLUMNS].copy()


def _load_openvid_rows(openvid_manifest: Path, openvid_preprocessed_dir: Path, cwd: Path) -> pd.DataFrame:
    df = pd.read_csv(openvid_manifest).copy()
    if "caption" not in df.columns or "sample_idx" not in df.columns:
        raise RuntimeError("OpenVid manifest must contain at least 'caption' and 'sample_idx'")

    df["caption"] = df["caption"].fillna("").astype(str).str.strip()
    df = df[df["caption"] != ""].copy()

    if "modality" in df.columns:
        df["modality"] = df["modality"].fillna("").astype(str).str.strip().str.lower()
        df = df[df["modality"].eq("video")].copy()
    else:
        df["modality"] = "video"

    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce")
    df = df[df["sample_idx"].notna()].copy()
    df["sample_idx"] = df["sample_idx"].astype(int)

    if "preprocessed_path" not in df.columns:
        df["preprocessed_path"] = ""
    df["preprocessed_path"] = df["preprocessed_path"].fillna("").astype(str).str.strip()
    missing_pre = df["preprocessed_path"].eq("")
    df.loc[missing_pre, "preprocessed_path"] = df.loc[missing_pre, "sample_idx"].map(
        lambda idx: str((openvid_preprocessed_dir / f"sample_{int(idx):08d}.pkl").resolve())
    )
    df["preprocessed_path"] = df["preprocessed_path"].map(lambda p: _resolve_repo_relative(p, cwd))

    if "video_path" not in df.columns:
        df["video_path"] = ""
    df["video_path"] = df["video_path"].fillna("").astype(str).str.strip()
    if "media_path" in df.columns:
        missing_video_path = df["video_path"].eq("")
        df.loc[missing_video_path, "video_path"] = (
            df.loc[missing_video_path, "media_path"].fillna("").astype(str).str.strip()
        )
    df["video_path"] = df["video_path"].map(lambda p: _resolve_repo_relative(p, cwd))

    if "dataset" not in df.columns:
        df["dataset"] = "openvid_1m_current"
    df["dataset"] = df["dataset"].fillna("").astype(str).str.strip()
    df.loc[df["dataset"].eq(""), "dataset"] = "openvid_1m_current"

    if "video" not in df.columns:
        df["video"] = ""
    df["video"] = df["video"].fillna("").astype(str).str.strip()

    missing_video_name = df["video"].eq("")
    if missing_video_name.any():
        df.loc[missing_video_name, "video"] = df.loc[missing_video_name].apply(
            lambda row: os.path.basename(
                _first_present(row, ["video_path", "media_path", "image_path", "source_id"])
            ),
            axis=1,
        )
    df.loc[df["video"].eq(""), "video"] = df.loc[df["video"].eq(""), "sample_idx"].map(
        lambda idx: f"sample_{int(idx):08d}.pkl"
    )

    df["modality"] = "video"
    return df[REQUIRED_OUTPUT_COLUMNS].copy()


def build_joint_manifest(
    image_manifest: Path,
    openvid_manifest: Path,
    openvid_preprocessed_dir: Path,
    output_prefix: Path,
) -> Dict[str, object]:
    cwd = Path.cwd()
    image_df = _load_image_rows(image_manifest=image_manifest, cwd=cwd)
    video_df = _load_openvid_rows(
        openvid_manifest=openvid_manifest,
        openvid_preprocessed_dir=openvid_preprocessed_dir,
        cwd=cwd,
    )

    openvid_offset = int(image_df["sample_idx"].max()) + 1 if len(image_df) else 0
    video_df["sample_idx"] = video_df["sample_idx"] + openvid_offset

    combined_df = pd.concat([video_df, image_df], ignore_index=True)
    if int(combined_df["sample_idx"].duplicated().sum()) != 0:
        dup_count = int(combined_df["sample_idx"].duplicated().sum())
        raise RuntimeError(f"Combined manifest has duplicate sample_idx values: {dup_count}")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    combined_path = output_prefix.with_suffix(".csv")
    video_path = output_prefix.with_name(output_prefix.name + "_video.csv")
    image_path = output_prefix.with_name(output_prefix.name + "_image.csv")
    summary_path = output_prefix.with_suffix(".summary.json")

    combined_df.to_csv(combined_path, index=False)
    video_df.to_csv(video_path, index=False)
    image_df.to_csv(image_path, index=False)

    summary = {
        "combined_csv": str(combined_path),
        "video_csv": str(video_path),
        "image_csv": str(image_path),
        "rows_combined": int(len(combined_df)),
        "rows_video": int(len(video_df)),
        "rows_image": int(len(image_df)),
        "dataset_counts": {
            str(k): int(v) for k, v in combined_df["dataset"].value_counts().to_dict().items()
        },
        "modality_counts": {
            str(k): int(v) for k, v in combined_df["modality"].value_counts().to_dict().items()
        },
        "sample_idx_min": int(combined_df["sample_idx"].min()) if len(combined_df) else -1,
        "sample_idx_max": int(combined_df["sample_idx"].max()) if len(combined_df) else -1,
        "openvid_source_manifest": str(openvid_manifest.resolve()),
        "openvid_preprocessed_dir": str(openvid_preprocessed_dir.resolve()),
        "image_source_manifest": str(image_manifest.resolve()),
        "openvid_offset": int(openvid_offset),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", required=True, help="Merged JourneyDB + Short-Caption train-ready CSV")
    parser.add_argument("--openvid-manifest", required=True, help="OpenVid manifest CSV")
    parser.add_argument(
        "--openvid-preprocessed-dir",
        required=True,
        help="Directory containing OpenVid latent .pkl files",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Writes <prefix>.csv, <prefix>_video.csv, <prefix>_image.csv and <prefix>.summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_joint_manifest(
        image_manifest=Path(args.image_manifest),
        openvid_manifest=Path(args.openvid_manifest),
        openvid_preprocessed_dir=Path(args.openvid_preprocessed_dir),
        output_prefix=Path(args.output_prefix),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
