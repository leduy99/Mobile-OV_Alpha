#!/usr/bin/env python3
"""Build a train-ready joint manifest from current OpenVid latents plus LAION/COYO images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def _basename_from_row(row: pd.Series) -> str:
    for key in ("video_path", "media_path", "source_id", "video"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return os.path.basename(value.strip())
    return f"sample_{int(row['sample_idx']):08d}.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-mix-manifest",
        default="data/mix/manifests/joint_v4plus_openvid_partial_allavail_20260227.csv",
        help="Existing mixed manifest containing usable LAION/COYO rows.",
    )
    parser.add_argument(
        "--openvid-manifest",
        default="data/openvid_1m/manifests/openvid_manifest_0_112.csv",
        help="OpenVid unified manifest with sample_idx values.",
    )
    parser.add_argument(
        "--openvid-preprocessed-dir",
        default="data/openvid_1m/encoded/wan_vae_fp16_stream",
        help="Directory containing OpenVid latent .pkl files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="data/mix/manifests/joint_openvid_current_laion_coyo_20260315",
        help="Output prefix; writes combined/video/image CSVs plus summary JSON.",
    )
    args = parser.parse_args()

    existing_mix_manifest = Path(args.existing_mix_manifest).resolve()
    openvid_manifest = Path(args.openvid_manifest).resolve()
    openvid_preprocessed_dir = Path(args.openvid_preprocessed_dir).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    mix_df = pd.read_csv(existing_mix_manifest)
    keep_image = mix_df["dataset"].isin(["laion", "coyo_700m"])
    keep_image &= mix_df["modality"].astype(str).str.lower().eq("image")
    image_df = mix_df.loc[keep_image, ["video", "caption", "preprocessed_path", "video_path", "dataset", "modality", "sample_idx"]].copy()
    image_df["preprocessed_path"] = image_df["preprocessed_path"].astype(str)
    image_df = image_df[image_df["preprocessed_path"].map(lambda p: Path(p).exists())].copy()

    openvid_df = pd.read_csv(openvid_manifest)
    openvid_df = openvid_df.loc[openvid_df["modality"].astype(str).str.lower().eq("video")].copy()

    openvid_offset = int(image_df["sample_idx"].max()) + 1 if len(image_df) else 0
    openvid_df["preprocessed_path"] = openvid_df["sample_idx"].map(
        lambda idx: str(openvid_preprocessed_dir / f"sample_{int(idx):08d}.pkl")
    )
    openvid_df = openvid_df.loc[openvid_df["preprocessed_path"].map(lambda p: Path(p).exists())].copy()
    openvid_df["video"] = openvid_df.apply(_basename_from_row, axis=1)
    openvid_df["dataset"] = "openvid_1m_current"
    openvid_df["modality"] = "video"
    openvid_df["sample_idx"] = openvid_df["sample_idx"].astype(int) + openvid_offset
    openvid_df["video_path"] = openvid_df["video_path"].where(
        openvid_df["video_path"].notna(), openvid_df.get("media_path")
    )
    openvid_df = openvid_df[["video", "caption", "preprocessed_path", "video_path", "dataset", "modality", "sample_idx"]].copy()

    combined_df = pd.concat([openvid_df, image_df], ignore_index=True)
    if int(combined_df["sample_idx"].duplicated().sum()) != 0:
        dup_count = int(combined_df["sample_idx"].duplicated().sum())
        raise RuntimeError(f"Combined manifest has duplicate sample_idx values: {dup_count}")

    combined_path = output_prefix.with_suffix(".csv")
    video_path = output_prefix.with_name(output_prefix.name + "_video.csv")
    image_path = output_prefix.with_name(output_prefix.name + "_image.csv")
    summary_path = output_prefix.with_suffix(".summary.json")

    combined_df.to_csv(combined_path, index=False)
    openvid_df.to_csv(video_path, index=False)
    image_df.to_csv(image_path, index=False)

    summary = {
        "combined_csv": str(combined_path),
        "video_csv": str(video_path),
        "image_csv": str(image_path),
        "rows_combined": int(len(combined_df)),
        "rows_video": int(len(openvid_df)),
        "rows_image": int(len(image_df)),
        "dataset_counts": {str(k): int(v) for k, v in combined_df["dataset"].value_counts().to_dict().items()},
        "modality_counts": {str(k): int(v) for k, v in combined_df["modality"].value_counts().to_dict().items()},
        "sample_idx_min": int(combined_df["sample_idx"].min()) if len(combined_df) else -1,
        "sample_idx_max": int(combined_df["sample_idx"].max()) if len(combined_df) else -1,
        "openvid_source_manifest": str(openvid_manifest),
        "openvid_preprocessed_dir": str(openvid_preprocessed_dir),
        "existing_mix_manifest": str(existing_mix_manifest),
        "openvid_offset": int(openvid_offset),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
