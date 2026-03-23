#!/usr/bin/env python3
"""Build a train-ready LAION / COYO manifest from a source manifest + encoded dir.

Output schema:
  video,caption,preprocessed_path,video_path,dataset,modality,sample_idx

This is the missing bridge between:
- raw/materialized LAION / COYO source manifests
- WAN VAE encoded `sample_XXXXXXXX.pkl` files
- the current OpenVid + LAION / COYO joint manifest builder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _to_int(value, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def _pick_existing_path(row: pd.Series) -> str:
    for key in ("media_path", "video_path", "image_path"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _pick_video_name(row: pd.Series, sample_idx: int) -> str:
    for key in ("source_id", "video"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return f"sample_{sample_idx:08d}"


def build_rows(
    source_manifest: Path,
    encoded_dir: Path,
    keep_datasets: Optional[set[str]],
    modality_filter: str,
) -> tuple[List[Dict], Dict]:
    df = pd.read_csv(source_manifest)
    if "dataset" not in df.columns:
        df["dataset"] = ""
    if "modality" not in df.columns:
        df["modality"] = "image"
    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["modality"] = df["modality"].astype(str).str.strip().str.lower()

    rows: List[Dict] = []
    skipped_missing_idx = 0
    skipped_missing_caption = 0
    skipped_dataset = 0
    skipped_modality = 0
    skipped_missing_pkl = 0

    for _, row in df.iterrows():
        sample_idx = _to_int(row.get("sample_idx"), default=-1)
        if sample_idx < 0:
            skipped_missing_idx += 1
            continue

        dataset = str(row.get("dataset", "")).strip().lower()
        modality = str(row.get("modality", "image")).strip().lower() or "image"

        if keep_datasets is not None and dataset not in keep_datasets:
            skipped_dataset += 1
            continue
        if modality_filter != "all" and modality != modality_filter:
            skipped_modality += 1
            continue

        caption = str(row.get("caption", "")).strip()
        if not caption:
            skipped_missing_caption += 1
            continue

        preprocessed_path = encoded_dir / f"sample_{sample_idx:08d}.pkl"
        if not preprocessed_path.exists():
            skipped_missing_pkl += 1
            continue

        rows.append(
            {
                "video": _pick_video_name(row, sample_idx),
                "caption": caption,
                "preprocessed_path": str(preprocessed_path.resolve()),
                "video_path": _pick_existing_path(row),
                "dataset": dataset or "laion_coyo",
                "modality": modality,
                "sample_idx": int(sample_idx),
            }
        )

    summary = {
        "input_rows": int(len(df)),
        "output_rows": int(len(rows)),
        "skipped_missing_idx": int(skipped_missing_idx),
        "skipped_missing_caption": int(skipped_missing_caption),
        "skipped_dataset": int(skipped_dataset),
        "skipped_modality": int(skipped_modality),
        "skipped_missing_pkl": int(skipped_missing_pkl),
        "datasets_out": sorted({str(r["dataset"]) for r in rows}),
        "modalities_out": sorted({str(r["modality"]) for r in rows}),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        default="data/laion_coyo/manifests/laion_coyo_recovered_unique.csv",
    )
    parser.add_argument(
        "--encoded-dir",
        default="data/laion_coyo/encoded/wan_vae_openvid_mix_sana_ar",
    )
    parser.add_argument(
        "--output-csv",
        default="data/laion_coyo/manifests/laion_coyo_encoded_for_openvid_mix.csv",
    )
    parser.add_argument(
        "--datasets",
        default="laion,coyo_700m",
        help="Comma-separated dataset names to keep. Empty keeps all.",
    )
    parser.add_argument(
        "--modality",
        choices=["image", "video", "all"],
        default="image",
        help="Keep only rows with this modality.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_datasets = {x.strip().lower() for x in str(args.datasets).split(",") if x.strip()}
    if not keep_datasets:
        keep_datasets = None

    source_manifest = Path(args.source_manifest).resolve()
    encoded_dir = Path(args.encoded_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows, summary = build_rows(
        source_manifest=source_manifest,
        encoded_dir=encoded_dir,
        keep_datasets=keep_datasets,
        modality_filter=args.modality,
    )

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        out_df = out_df.sort_values("sample_idx").reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)

    summary.update(
        {
            "source_manifest": str(source_manifest),
            "encoded_dir": str(encoded_dir),
            "output_csv": str(output_csv),
        }
    )
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
