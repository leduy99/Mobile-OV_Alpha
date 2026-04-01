#!/usr/bin/env python3
"""Bootstrap Mobile-O-SFT into the repo's image source-manifest format.

This script streams one or more WebDataset tar shards from
`Amshaker/Mobile-O-SFT`, extracts paired image/text samples, writes local image
files, and emits a source manifest CSV compatible with:

- `tools/data_prepare/encode_laion_coyo_images_sana_ar.py`
- `tools/data_prepare/build_laion_coyo_encoded_manifest.py`
- `tools/train_stage1_teacher_free.py`

It is designed to work for:
- smoke tests: stream a single shard and stop after a few samples
- full processing: stream all dataset tar shards and write the full local corpus
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import requests
from huggingface_hub import hf_hub_url, list_repo_files
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp"}
TEXT_EXTS = {"txt", "text", "caption"}
DEFAULT_REPO_ID = "Amshaker/Mobile-O-SFT"
DEFAULT_DATASET_NAME = "mobile_o_sft"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--filenames",
        default="object_2.tar",
        help="Comma-separated tar filenames to process, or 'all' for every .tar shard.",
    )
    parser.add_argument(
        "--output-root",
        default="data/mobile_o_sft",
        help="Root directory for raw images and manifests.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional manifest CSV path. Defaults to <output-root>/manifests/mobile_o_sft_source.csv",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after writing this many paired samples total. Omit for full dataset.",
    )
    parser.add_argument(
        "--start-sample-idx",
        type=int,
        default=0,
        help="Starting sample_idx value in the output manifest.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name to write into the manifest.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds for shard requests.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of tar shards to process in parallel. Use 1 for sequential processing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image files and manifest rows.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. Public Mobile-O-SFT does not require one.",
    )
    return parser.parse_args()


def _resolve_filenames(repo_id: str, filenames_arg: str) -> List[str]:
    if filenames_arg.strip().lower() != "all":
        return [x.strip() for x in filenames_arg.split(",") if x.strip()]
    files = list_repo_files(repo_id, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".tar"))


def _member_key(name: str) -> Tuple[str, str]:
    base = Path(name).name
    if "." not in base:
        return base, ""
    stem, ext = base.rsplit(".", 1)
    return stem, ext.lower()


def _iter_wds_pairs(url: str, token: Optional[str], timeout: int) -> Iterator[Tuple[str, Dict]]:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    pending: Dict[str, Dict] = {}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        tf = tarfile.open(fileobj=response.raw, mode="r|*")
        for member in tf:
            if not member.isfile():
                continue
            key, ext = _member_key(member.name)
            if ext not in IMAGE_EXTS and ext not in TEXT_EXTS:
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            record = pending.setdefault(key, {})
            if ext in IMAGE_EXTS:
                record["image_bytes"] = payload
                record["image_ext"] = ext
                record["member_name"] = member.name
            elif ext in TEXT_EXTS:
                record["caption"] = payload.decode("utf-8", errors="replace").strip()
            if "image_bytes" in record and "caption" in record:
                yield key, record
                pending.pop(key, None)


def _save_image(image_bytes: bytes, ext: str, out_path: Path) -> Tuple[int, int]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG" if ext in {"jpg", "jpeg"} else ext.upper()
    save_kwargs = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = 95
    image.save(out_path, **save_kwargs)
    width, height = image.size
    return width, height


def _source_id(shard_name: str, key: str) -> str:
    return f"{Path(shard_name).stem}:{key}"


def _process_shard(
    repo_id: str,
    shard_name: str,
    images_dir: Path,
    dataset_name: str,
    token: Optional[str],
    timeout: int,
    overwrite: bool,
) -> Tuple[str, List[Dict], int]:
    shard_url = hf_hub_url(repo_id, shard_name, repo_type="dataset")
    shard_rows: List[Dict] = []
    total_seen = 0

    for key, record in _iter_wds_pairs(shard_url, token=token, timeout=timeout):
        total_seen += 1
        caption = str(record.get("caption", "")).strip()
        if not caption:
            continue

        ext = str(record.get("image_ext", "jpg")).lower()
        out_path = images_dir / shard_name.replace(".tar", "") / f"{key}.{ext}"
        if out_path.exists() and not overwrite:
            width = height = -1
            try:
                with Image.open(out_path) as im:
                    width, height = im.size
            except Exception:
                pass
        else:
            width, height = _save_image(record["image_bytes"], ext, out_path)

        shard_rows.append(
            {
                "dataset": dataset_name,
                "modality": "image",
                "caption": caption,
                "image_path": str(out_path),
                "media_path": str(out_path),
                "video_path": "",
                "source_id": _source_id(shard_name, key),
                "source_url": shard_url,
                "width": width,
                "height": height,
                "extra_json": json.dumps(
                    {
                        "shard": shard_name,
                        "member_name": record.get("member_name", ""),
                        "key": key,
                        "caption_sha1": hashlib.sha1(caption.encode("utf-8")).hexdigest(),
                    },
                    ensure_ascii=False,
                ),
            }
        )

    return shard_name, shard_rows, total_seen


def _write_manifest(rows: List[Dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_idx",
        "dataset",
        "modality",
        "caption",
        "image_path",
        "media_path",
        "video_path",
        "source_id",
        "source_url",
        "width",
        "height",
        "extra_json",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    images_dir = output_root / "raw" / "images"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else output_root / "manifests" / "mobile_o_sft_source.csv"

    rows: List[Dict] = []
    filenames = _resolve_filenames(args.repo_id, args.filenames)
    sample_idx = int(args.start_sample_idx)
    total_seen = 0
    total_written = 0

    effective_jobs = max(1, int(args.jobs))
    if args.max_samples is not None and effective_jobs > 1:
        print(
            json.dumps(
                {
                    "note": "max_samples is set; forcing sequential processing to keep exact sample cap.",
                    "requested_jobs": effective_jobs,
                }
            )
        )
        effective_jobs = 1

    shard_results: Dict[str, Tuple[List[Dict], int]] = {}
    if effective_jobs == 1:
        for shard_name in filenames:
            _, shard_rows, shard_seen = _process_shard(
                repo_id=args.repo_id,
                shard_name=shard_name,
                images_dir=images_dir,
                dataset_name=args.dataset_name,
                token=args.hf_token,
                timeout=args.timeout,
                overwrite=args.overwrite,
            )
            shard_results[shard_name] = (shard_rows, shard_seen)
            if args.max_samples is not None:
                current_rows = sum(len(x[0]) for x in shard_results.values())
                if current_rows >= args.max_samples:
                    break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_jobs) as executor:
            future_to_shard = {
                executor.submit(
                    _process_shard,
                    args.repo_id,
                    shard_name,
                    images_dir,
                    args.dataset_name,
                    args.hf_token,
                    args.timeout,
                    args.overwrite,
                ): shard_name
                for shard_name in filenames
            }
            for future in concurrent.futures.as_completed(future_to_shard):
                shard_name, shard_rows, shard_seen = future.result()
                shard_results[shard_name] = (shard_rows, shard_seen)

    for shard_name in filenames:
        if shard_name not in shard_results:
            continue
        shard_rows, shard_seen = shard_results[shard_name]
        total_seen += shard_seen
        for row in shard_rows:
            row["sample_idx"] = sample_idx
            rows.append(row)
            total_written += 1
            sample_idx += 1
            if args.max_samples is not None and total_written >= args.max_samples:
                _write_manifest(rows, output_csv)
                summary = {
                    "repo_id": args.repo_id,
                    "filenames": filenames,
                    "output_root": str(output_root),
                    "output_csv": str(output_csv),
                    "total_seen_pairs": total_seen,
                    "total_written": total_written,
                    "jobs": effective_jobs,
                    "stopped_early": True,
                }
                output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary, indent=2))
                return

    _write_manifest(rows, output_csv)
    summary = {
        "repo_id": args.repo_id,
        "filenames": filenames,
        "output_root": str(output_root),
        "output_csv": str(output_csv),
        "total_seen_pairs": total_seen,
        "total_written": total_written,
        "jobs": effective_jobs,
        "stopped_early": False,
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
