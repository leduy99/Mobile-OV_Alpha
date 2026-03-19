#!/usr/bin/env python3
"""
Merge worker shard manifests produced by materialize_unified_manifest.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge worker shard manifests into one CSV.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def _sample_idx(row: Dict[str, str]) -> int:
    try:
        return int(row.get("sample_idx", "0"))
    except Exception:
        return 0


def main() -> int:
    args = parse_args()
    input_paths = sorted(Path(p) for p in glob.glob(args.input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No shard manifests matched: {args.input_glob}")

    fieldnames: List[str] = []
    rows: List[Dict[str, str]] = []
    for path in input_paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not fieldnames:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)

    rows.sort(key=_sample_idx)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_input_files": len(input_paths),
        "num_rows": len(rows),
        "output_csv": str(args.output_csv),
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
