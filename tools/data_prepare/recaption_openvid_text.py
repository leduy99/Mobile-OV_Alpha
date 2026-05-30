#!/usr/bin/env python3
"""Text-only OpenVid recaptioning.

This script rewrites existing OpenVid captions into short/medium/long variants.
It does not load or decode videos; the expensive model is a text LLM such as
Qwen/Qwen3.6-35B-A3B. Use shards to run one independent process per GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data_prepare.captioners.base import CAPTION_KEYS, fallback_caption_result, normalize_caption_text
from tools.data_prepare.captioners.mock import MockTextCaptioner
from tools.data_prepare.captioners.qwen_text import QwenTextCaptioner


LOGGER = logging.getLogger("recaption_openvid_text")
EXTRA_COLUMNS = [
    "recaption_row_id",
    "caption_base",
    *CAPTION_KEYS,
    "caption_source",
    "caption_status",
    "caption_error",
    "caption_raw_response",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="OpenVid-style CSV with a caption column.")
    parser.add_argument("--output-dir", required=True, help="Directory for shard output CSVs.")
    parser.add_argument("--caption-column", default="caption", help="Input caption column.")
    parser.add_argument("--captioner", choices=["qwen", "mock"], default="qwen")
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows for this shard.")
    parser.add_argument("--save-every", type=int, default=1, help="Flush every N written rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing shard output.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry rows previously marked failed.")
    parser.add_argument("--fail-on-error", action="store_true", help="Abort instead of writing fallback captions on recaption errors.")
    parser.add_argument(
        "--no-global-resume",
        action="store_true",
        help="Only skip rows already present in this shard file. By default all part_*.csv files are used.",
    )
    parser.add_argument("--resume-part-glob", default="part_*.csv", help="Part glob used for global resume.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def build_captioner(args: argparse.Namespace):
    if args.captioner == "mock":
        return MockTextCaptioner()
    return QwenTextCaptioner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        trust_remote_code=args.trust_remote_code,
    )


def shard_rows(df: pd.DataFrame, num_shards: int, shard_id: int, limit: int | None) -> Iterable[tuple[int, pd.Series]]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")
    count = 0
    for row_id, row in df.iterrows():
        if int(row_id) % int(num_shards) != int(shard_id):
            continue
        yield int(row_id), row
        count += 1
        if limit is not None and count >= int(limit):
            break


def load_processed_rows(path: Path, retry_failed: bool) -> set[int]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["recaption_row_id", "caption_status"], on_bad_lines="skip")
    except Exception:
        return set()
    if retry_failed:
        df = df[df["caption_status"].fillna("").astype(str).str.lower().ne("failed")]
    return {int(x) for x in pd.to_numeric(df["recaption_row_id"], errors="coerce").dropna().astype(int)}


def load_processed_rows_from_parts(output_dir: Path, part_glob: str, retry_failed: bool) -> set[int]:
    processed: set[int] = set()
    for path in sorted(output_dir.glob(part_glob)):
        processed.update(load_processed_rows(path, retry_failed=retry_failed))
    return processed


def make_fieldnames(input_columns: List[str]) -> List[str]:
    fieldnames = list(input_columns)
    for col in EXTRA_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)
    return fieldnames


def row_to_output(
    row_id: int,
    row: pd.Series,
    caption_column: str,
    captioner,
    *,
    fail_on_error: bool = False,
) -> Dict[str, object]:
    row_dict = row.to_dict()
    base_caption = normalize_caption_text(row.get(caption_column, ""))
    if not base_caption:
        if fail_on_error:
            raise ValueError(f"empty caption column for row_id={row_id}: {caption_column}")
        result = fallback_caption_result("")
        status = "failed"
        error = f"empty caption column: {caption_column}"
    else:
        try:
            result = captioner.caption_from_text(base_caption)
            status = "ok"
            error = ""
        except Exception as exc:
            if fail_on_error:
                raise RuntimeError(f"Recaption failed for row_id={row_id}: {exc}") from exc
            LOGGER.warning("Recaption failed for row_id=%s: %s", row_id, exc)
            result = fallback_caption_result(base_caption)
            status = "failed"
            error = str(exc)[:1000]

    row_dict.update(result.as_dict())
    row_dict["recaption_row_id"] = int(row_id)
    row_dict["caption_base"] = base_caption
    row_dict["caption_source"] = getattr(captioner, "source_name", type(captioner).__name__)
    row_dict["caption_status"] = status
    row_dict["caption_error"] = error
    return row_dict


def write_summary(path: Path, *, input_csv: Path, output_csv: Path, rows_seen: int, rows_written: int, skipped: int) -> None:
    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows_seen": int(rows_seen),
        "rows_written": int(rows_written),
        "rows_skipped_existing": int(skipped),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"part_{int(args.shard_id):05d}_of_{int(args.num_shards):05d}.csv"
    summary_json = output_csv.with_suffix(".summary.json")

    if args.overwrite and output_csv.exists():
        output_csv.unlink()

    df = pd.read_csv(input_csv).reset_index(drop=True)
    if args.caption_column not in df.columns:
        raise ValueError(f"Input CSV missing caption column: {args.caption_column}")

    if args.no_global_resume:
        processed = load_processed_rows(output_csv, retry_failed=bool(args.retry_failed))
    else:
        processed = load_processed_rows_from_parts(
            output_dir,
            part_glob=args.resume_part_glob,
            retry_failed=bool(args.retry_failed),
        )
    LOGGER.info("Resume index loaded: processed_rows=%d global_resume=%s", len(processed), not args.no_global_resume)
    captioner = build_captioner(args)
    fieldnames = make_fieldnames(list(df.columns))
    append = output_csv.exists() and not args.overwrite
    rows_seen = rows_written = skipped = 0

    with output_csv.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not append:
            writer.writeheader()
        for row_id, row in shard_rows(df, args.num_shards, args.shard_id, args.limit):
            rows_seen += 1
            if row_id in processed:
                skipped += 1
                continue
            writer.writerow(row_to_output(row_id, row, args.caption_column, captioner, fail_on_error=bool(args.fail_on_error)))
            rows_written += 1
            if rows_written % max(1, int(args.save_every)) == 0:
                f.flush()
                os.fsync(f.fileno())

    write_summary(
        summary_json,
        input_csv=input_csv,
        output_csv=output_csv,
        rows_seen=rows_seen,
        rows_written=rows_written,
        skipped=skipped,
    )
    LOGGER.info("Wrote %s rows=%d skipped=%d", output_csv, rows_written, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
