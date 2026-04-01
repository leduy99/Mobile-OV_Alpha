#!/usr/bin/env python3
"""Build a LAION / COYO source manifest from parquet metadata shards.

This script fills the current gap in the repo's LAION / COYO pipeline:
it turns raw metadata shards (local parquet files or parquet files stored in a
Hugging Face dataset repo) into the source-manifest CSV consumed by
`materialize_unified_manifest.py`.

The output schema matches the manifest-based LAION / COYO flow already used in
this repository:
  sample_idx,dataset,modality,caption,media_path,video_path,image_path,
  source_url,source_id,split,part_user,part_remote,span_start,span_end,
  width,height,extra_json

Typical usage for a "full" download is:
1. build the source manifest from all parquet shards for COYO and/or LAION
2. materialize raw media with `materialize_unified_manifest.py`
3. continue with the existing encode + train-manifest flow
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pyarrow.dataset as ds
from huggingface_hub import hf_hub_download, list_repo_files


LOG = logging.getLogger("bootstrap_laion_coyo_source_manifest")

OUTPUT_FIELDNAMES = [
    "sample_idx",
    "dataset",
    "modality",
    "caption",
    "media_path",
    "video_path",
    "image_path",
    "source_url",
    "source_id",
    "split",
    "part_user",
    "part_remote",
    "span_start",
    "span_end",
    "width",
    "height",
    "extra_json",
]

PRESETS = {
    "coyo_700m": {
        "dataset_name": "coyo_700m",
        "caption_column": "text",
        "url_column": "url",
        "id_column": "id",
        "width_column": "width",
        "height_column": "height",
        "extra_columns": ["clip_similarity", "watermark_score", "nsfw_score"],
    },
    "laion": {
        "dataset_name": "laion",
        "caption_column": "TEXT",
        "url_column": "URL",
        "id_column": "SAMPLE_ID",
        "width_column": "WIDTH",
        "height_column": "HEIGHT",
        "extra_columns": ["similarity", "LICENSE", "NSFW"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        action="append",
        default=[],
        help="Local parquet glob(s), for example 'metadata/coyo/*.parquet'. Can be repeated.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default="",
        help="Optional Hugging Face dataset repo id that stores parquet metadata shards.",
    )
    parser.add_argument(
        "--hf-filename-glob",
        action="append",
        default=[],
        help="Filename glob(s) inside the HF dataset repo, for example 'data/*.parquet'. Can be repeated.",
    )
    parser.add_argument(
        "--hf-exclude-glob",
        action="append",
        default=[],
        help="Filename glob(s) to exclude inside the HF dataset repo. Can be repeated.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="data/laion_coyo/parquet_cache",
        help="Directory used to cache parquet shards downloaded from HF.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HF token. Public dataset repos usually do not need this.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="",
        help="Optional preset for common COYO / LAION parquet schemas.",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Dataset label written into the manifest, for example 'coyo_700m' or 'laion'.",
    )
    parser.add_argument("--caption-column", default="", help="Parquet column that stores captions.")
    parser.add_argument("--url-column", default="", help="Parquet column that stores image URLs.")
    parser.add_argument("--id-column", default="", help="Optional parquet column for a stable sample id.")
    parser.add_argument("--width-column", default="", help="Optional parquet column for width.")
    parser.add_argument("--height-column", default="", help="Optional parquet column for height.")
    parser.add_argument(
        "--extra-columns",
        default="",
        help="Comma-separated extra parquet columns to copy into extra_json.",
    )
    parser.add_argument(
        "--modality",
        default="image",
        choices=["image", "video"],
        help="Modality written into the output manifest. LAION / COYO current path should use 'image'.",
    )
    parser.add_argument(
        "--default-split",
        default="train",
        help="Split value written into the manifest when no split column is used.",
    )
    parser.add_argument(
        "--start-sample-idx",
        type=int,
        default=0,
        help="Starting sample_idx for sequential numbering.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for debugging. 0 keeps all rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Arrow scan batch size when reading parquet files.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50000,
        help="Progress log interval in kept output rows.",
    )
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _split_csv_arg(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _apply_preset(args: argparse.Namespace) -> None:
    if not args.preset:
        return
    preset = PRESETS[args.preset]
    if not args.dataset_name:
        args.dataset_name = preset["dataset_name"]
    if not args.caption_column:
        args.caption_column = preset["caption_column"]
    if not args.url_column:
        args.url_column = preset["url_column"]
    if not args.id_column:
        args.id_column = preset["id_column"]
    if not args.width_column:
        args.width_column = preset["width_column"]
    if not args.height_column:
        args.height_column = preset["height_column"]
    if not args.extra_columns:
        args.extra_columns = ",".join(preset["extra_columns"])


def _expand_local_files(globs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in globs:
        matches = [Path(p) for p in sorted(glob.glob(pattern))]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        out.extend(Path(m).resolve() for m in matches if Path(m).suffix.lower() == ".parquet")
    unique = sorted({p for p in out})
    return unique


def _resolve_hf_files(
    repo_id: str,
    include_globs: List[str],
    exclude_globs: List[str],
    token: Optional[str],
    cache_dir: Path,
) -> List[Path]:
    files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    parquet_files = [f for f in files if f.endswith(".parquet")]
    if include_globs:
        parquet_files = [
            f for f in parquet_files
            if any(fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern) for pattern in include_globs)
        ]
    if exclude_globs:
        parquet_files = [
            f for f in parquet_files
            if not any(fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern) for pattern in exclude_globs)
        ]
    parquet_files = sorted(parquet_files)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched in HF dataset repo {repo_id}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_paths: List[Path] = []
    for idx, remote_name in enumerate(parquet_files, start=1):
        LOG.info("Downloading HF parquet %d/%d: %s", idx, len(parquet_files), remote_name)
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote_name,
            local_dir=str(cache_dir),
            token=token,
            resume_download=True,
        )
        local_paths.append(Path(local).resolve())
    return local_paths


def _resolve_input_parquets(args: argparse.Namespace) -> List[Path]:
    local_files = _expand_local_files(args.input_glob)
    hf_files: List[Path] = []
    if args.hf_repo_id:
        hf_files = _resolve_hf_files(
            repo_id=args.hf_repo_id,
            include_globs=args.hf_filename_glob,
            exclude_globs=args.hf_exclude_glob,
            token=args.hf_token,
            cache_dir=Path(args.hf_cache_dir).resolve(),
        )
    files = sorted({*local_files, *hf_files})
    if not files:
        raise FileNotFoundError("No parquet input files found. Pass --input-glob and/or --hf-repo-id.")
    return files


def _normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _normalize_int(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def _stable_source_id(url: str, sample_idx: int) -> str:
    if url:
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()
        return str(int(digest[:15], 16))
    return str(sample_idx)


def _iter_rows(parquet_files: List[Path], columns: List[str], batch_size: int) -> Iterator[tuple[dict, str]]:
    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    for fragment in dataset.get_fragments():
        source_path = getattr(fragment, "path", "") or ""
        scanner = fragment.scanner(columns=columns, batch_size=batch_size)
        for batch in scanner.to_batches():
            data = batch.to_pydict()
            if not data:
                continue
            keys = list(data.keys())
            if not keys:
                continue
            row_count = len(data[keys[0]])
            for idx in range(row_count):
                row = {key: data[key][idx] for key in keys}
                yield row, source_path


def main() -> int:
    setup_logging()
    args = parse_args()
    _apply_preset(args)

    if not args.dataset_name:
        raise ValueError("--dataset-name is required unless --preset provides it")
    if not args.caption_column or not args.url_column:
        raise ValueError("--caption-column and --url-column are required unless --preset provides them")

    extra_columns = _split_csv_arg(args.extra_columns)
    parquet_files = _resolve_input_parquets(args)

    needed_columns = [args.caption_column, args.url_column]
    for optional in [args.id_column, args.width_column, args.height_column, *extra_columns]:
        if optional:
            needed_columns.append(optional)
    needed_columns = list(dict.fromkeys(needed_columns))

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    LOG.info(
        "Building source manifest: parquet_files=%d dataset=%s modality=%s output=%s",
        len(parquet_files),
        args.dataset_name,
        args.modality,
        output_csv,
    )

    processed = 0
    written = 0
    skipped_missing_caption = 0
    skipped_missing_url = 0
    sample_idx = args.start_sample_idx
    source_shards = set()

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for row, source_path in _iter_rows(parquet_files, needed_columns, args.batch_size):
            processed += 1
            if source_path:
                source_shards.add(source_path)

            caption = _normalize_text(row.get(args.caption_column))
            if not caption:
                skipped_missing_caption += 1
                continue

            source_url = _normalize_text(row.get(args.url_column))
            if not source_url:
                skipped_missing_url += 1
                continue

            source_id = _normalize_int(row.get(args.id_column)) if args.id_column else ""
            if not source_id:
                source_id = _stable_source_id(source_url, sample_idx)

            width = _normalize_int(row.get(args.width_column)) if args.width_column else ""
            height = _normalize_int(row.get(args.height_column)) if args.height_column else ""

            extra = {}
            if args.hf_repo_id:
                extra["repo_id"] = args.hf_repo_id
            if source_path:
                extra["parquet_file"] = source_path
            for key in extra_columns:
                value = row.get(key)
                if value is None:
                    continue
                if str(value).strip().lower() == "nan":
                    continue
                extra[key] = value

            writer.writerow(
                {
                    "sample_idx": sample_idx,
                    "dataset": args.dataset_name,
                    "modality": args.modality,
                    "caption": caption,
                    "media_path": "",
                    "video_path": "",
                    "image_path": "",
                    "source_url": source_url,
                    "source_id": source_id,
                    "split": args.default_split,
                    "part_user": -1,
                    "part_remote": -1,
                    "span_start": "",
                    "span_end": "",
                    "width": width,
                    "height": height,
                    "extra_json": json.dumps(extra, ensure_ascii=False),
                }
            )
            written += 1
            sample_idx += 1

            if args.log_every > 0 and written > 0 and written % args.log_every == 0:
                LOG.info("Manifest progress | processed=%d written=%d skipped_caption=%d skipped_url=%d", processed, written, skipped_missing_caption, skipped_missing_url)

            if args.max_rows > 0 and written >= args.max_rows:
                LOG.info("Reached --max-rows=%d; stopping early.", args.max_rows)
                break

    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_csv": str(output_csv),
        "dataset": args.dataset_name,
        "modality": args.modality,
        "input_parquet_files": len(parquet_files),
        "input_parquet_preview": [str(p) for p in parquet_files[:10]],
        "processed_rows": processed,
        "written_rows": written,
        "skipped_missing_caption": skipped_missing_caption,
        "skipped_missing_url": skipped_missing_url,
        "caption_column": args.caption_column,
        "url_column": args.url_column,
        "id_column": args.id_column,
        "width_column": args.width_column,
        "height_column": args.height_column,
        "extra_columns": extra_columns,
        "hf_repo_id": args.hf_repo_id,
        "source_shards_seen": len(source_shards),
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Done | written=%d output=%s", written, output_csv)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
