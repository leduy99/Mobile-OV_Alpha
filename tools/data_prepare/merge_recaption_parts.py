#!/usr/bin/env python3
"""Merge recaption shard CSVs into one OpenVid manifest with caption variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


AUGMENT_COLUMNS = [
    "caption_base",
    "caption_short",
    "caption_medium",
    "caption_long",
    "caption_source",
    "caption_status",
    "caption_error",
    "caption_raw_response",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Original OpenVid-style CSV.")
    parser.add_argument("--parts-dir", required=True, help="Directory containing part_*.csv recaption shards.")
    parser.add_argument("--output-csv", required=True, help="Merged output CSV.")
    parser.add_argument("--part-glob", default="part_*.csv")
    return parser.parse_args()


def read_parts(parts: List[Path]) -> pd.DataFrame:
    frames = []
    for path in parts:
        try:
            frames.append(pd.read_csv(path, on_bad_lines="skip"))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "recaption_row_id" not in out.columns:
        raise ValueError("Recaption parts are missing recaption_row_id")
    out["recaption_row_id"] = pd.to_numeric(out["recaption_row_id"], errors="coerce")
    out = out[out["recaption_row_id"].notna()].copy()
    out["recaption_row_id"] = out["recaption_row_id"].astype(int)
    return out.drop_duplicates(subset=["recaption_row_id"], keep="last")


def nonempty_or_fallback(value, fallback: str) -> str:
    if pd.isna(value):
        return fallback
    text = " ".join(str(value).split()).strip()
    return text or fallback


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    parts_dir = Path(args.parts_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    parts = sorted(parts_dir.glob(args.part_glob))

    base_df = pd.read_csv(input_csv).reset_index(drop=True)
    base_df["recaption_row_id"] = range(len(base_df))
    part_df = read_parts(parts)
    part_df = part_df.set_index("recaption_row_id") if len(part_df) else pd.DataFrame()

    for col in AUGMENT_COLUMNS:
        if col not in base_df.columns:
            base_df[col] = ""

    matched = 0
    for idx, row in base_df.iterrows():
        fallback = nonempty_or_fallback(row.get("caption", ""), "")
        if len(part_df) and int(row["recaption_row_id"]) in part_df.index:
            matched += 1
            rec = part_df.loc[int(row["recaption_row_id"])]
            for col in AUGMENT_COLUMNS:
                if col in rec:
                    base_df.at[idx, col] = rec[col]
        base_df.at[idx, "caption_short"] = nonempty_or_fallback(base_df.at[idx, "caption_short"], fallback)
        base_df.at[idx, "caption_medium"] = nonempty_or_fallback(base_df.at[idx, "caption_medium"], fallback)
        base_df.at[idx, "caption_long"] = nonempty_or_fallback(base_df.at[idx, "caption_long"], fallback)
        if not nonempty_or_fallback(base_df.at[idx, "caption_status"], ""):
            base_df.at[idx, "caption_status"] = "missing"
        if not nonempty_or_fallback(base_df.at[idx, "caption_source"], ""):
            base_df.at[idx, "caption_source"] = "fallback"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(output_csv, index=False)
    summary = {
        "input_csv": str(input_csv),
        "parts_dir": str(parts_dir),
        "output_csv": str(output_csv),
        "parts": [str(p) for p in parts],
        "rows_input": int(len(base_df)),
        "rows_matched_parts": int(matched),
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
