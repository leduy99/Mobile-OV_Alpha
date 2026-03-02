#!/usr/bin/env python3
"""
Build a joint image+video training manifest with direct preprocessed paths.

Output schema:
  video,caption,preprocessed_path,video_path,dataset,modality,sample_idx

This schema is compatible with OpenVidDataset direct_preprocessed_mode.
"""

import argparse
import os
from typing import Dict, List

import pandas as pd


def _norm_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _build_msrvtt_rows(
    msrvtt_csv: str,
    msrvtt_preprocessed_dir: str,
    msrvtt_video_root: str,
) -> List[Dict]:
    df = pd.read_csv(msrvtt_csv)
    rows: List[Dict] = []
    for idx, row in df.iterrows():
        video = str(row.get("video", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if not video or not caption:
            continue
        base = os.path.splitext(os.path.basename(video))[0]
        pkl_path = os.path.join(msrvtt_preprocessed_dir, f"{base}_features.pkl")
        if not os.path.exists(pkl_path):
            continue
        video_path = os.path.join(msrvtt_video_root, video)
        if not os.path.exists(video_path):
            alt_video_path = os.path.join(msrvtt_video_root, base + ".mp4")
            video_path = alt_video_path if os.path.exists(alt_video_path) else ""
        rows.append(
            {
                "video": video,
                "caption": caption,
                "preprocessed_path": _norm_path(pkl_path),
                "video_path": _norm_path(video_path) if video_path else "",
                "dataset": "msrvtt",
                "modality": "video",
                "sample_idx": int(idx),
            }
        )
    return rows


def _build_laion_coyo_rows(
    source_manifest_csv: str,
    encoded_dir: str,
) -> List[Dict]:
    df = pd.read_csv(source_manifest_csv)
    rows: List[Dict] = []
    for _, row in df.iterrows():
        try:
            sample_idx = int(row.get("sample_idx"))
        except Exception:
            continue
        caption = str(row.get("caption", "")).strip()
        if not caption:
            continue
        pkl_name = f"sample_{sample_idx:08d}.pkl"
        pkl_path = os.path.join(encoded_dir, pkl_name)
        if not os.path.exists(pkl_path):
            continue
        dataset = str(row.get("dataset", "laion_coyo")).strip() or "laion_coyo"
        modality = str(row.get("modality", "image")).strip() or "image"
        video_name = str(row.get("source_id", "")).strip()
        if not video_name:
            video_name = os.path.splitext(pkl_name)[0]
        media_path = str(row.get("media_path", "")).strip()
        video_path = str(row.get("video_path", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        raw_path = media_path or video_path or image_path
        rows.append(
            {
                "video": video_name,
                "caption": caption,
                "preprocessed_path": _norm_path(pkl_path),
                "video_path": _norm_path(raw_path) if raw_path else "",
                "dataset": dataset,
                "modality": modality,
                "sample_idx": sample_idx,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--msrvtt-csv",
        default="data/msrvtt/OpenVid_extracted_subset_unique.csv",
    )
    parser.add_argument(
        "--msrvtt-preprocessed-dir",
        default="data/msrvtt/preprocessed",
    )
    parser.add_argument(
        "--msrvtt-video-root",
        default="data/msrvtt/videos",
    )
    parser.add_argument(
        "--laion-coyo-manifest",
        default="data/laion_coyo/manifests/laion_coyo_selected_media_existing_58k.csv",
    )
    parser.add_argument(
        "--laion-coyo-encoded-dir",
        default="data/laion_coyo/encoded/wan_vae_ivjoint_prep_58k_v2",
    )
    parser.add_argument(
        "--output-csv",
        default="data/mix/manifests/joint_msrvtt_laion_coyo_ivjoint_ready.csv",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle final manifest rows.",
    )
    args = parser.parse_args()

    msrvtt_rows = _build_msrvtt_rows(
        msrvtt_csv=args.msrvtt_csv,
        msrvtt_preprocessed_dir=args.msrvtt_preprocessed_dir,
        msrvtt_video_root=args.msrvtt_video_root,
    )
    laion_coyo_rows = _build_laion_coyo_rows(
        source_manifest_csv=args.laion_coyo_manifest,
        encoded_dir=args.laion_coyo_encoded_dir,
    )
    all_rows = msrvtt_rows + laion_coyo_rows
    out_df = pd.DataFrame(all_rows)
    if args.shuffle:
        out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"[done] output={args.output_csv}")
    print(f"[done] total={len(out_df)}")
    print(f"[done] msrvtt={len(msrvtt_rows)}")
    print(f"[done] laion_coyo={len(laion_coyo_rows)}")
    if len(out_df) > 0:
        print("[done] sample rows:")
        print(out_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()

