#!/usr/bin/env python3
"""Shared bootstrap logic for Mobile-O WebDataset image/text corpora."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import logging
import tarfile
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Optional, Tuple

import requests
from huggingface_hub import hf_hub_url, list_repo_files
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp"}
TEXT_EXTS = {"txt", "text", "caption"}
MANIFEST_FIELDNAMES = [
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
SHARD_FIELDNAMES = [x for x in MANIFEST_FIELDNAMES if x != "sample_idx"]
FAILURE_FIELDNAMES = [
    "dataset",
    "shard",
    "key",
    "member_name",
    "image_ext",
    "caption_sha1",
    "source_url",
    "reason",
]


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args(
    *,
    description: str,
    default_repo_id: str,
    default_output_root: str,
    default_dataset_name: str,
    default_filenames: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo-id", default=default_repo_id)
    parser.add_argument(
        "--input-root",
        default=None,
        help="Optional local directory containing .tar shards. When set, local tar files are used instead of Hugging Face.",
    )
    parser.add_argument(
        "--filenames",
        default=default_filenames,
        help="Comma-separated tar filenames to process, or 'all' for every .tar shard.",
    )
    parser.add_argument(
        "--output-root",
        default=default_output_root,
        help="Root directory for raw images and manifests.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional manifest CSV path. Defaults to <output-root>/manifests/<dataset>_source.csv",
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
        default=default_dataset_name,
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
        "--log-every",
        type=int,
        default=1000,
        help="Emit an in-shard progress log every N paired samples. Set 0 to disable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image files and manifest rows.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. Public Mobile-O datasets do not require one.",
    )
    return parser.parse_args()


def _resolve_filenames(repo_id: str, filenames_arg: str, input_root: Optional[Path]) -> List[str]:
    if input_root is not None:
        if not input_root.exists():
            raise FileNotFoundError(f"Local input root does not exist: {input_root}")
        if not input_root.is_dir():
            raise NotADirectoryError(f"Local input root is not a directory: {input_root}")

    if filenames_arg.strip().lower() != "all":
        filenames = [x.strip() for x in filenames_arg.split(",") if x.strip()]
        source_desc = str(input_root) if input_root is not None else f"dataset repo {repo_id}"
        LOGGER.info("Using %d explicitly requested shard(s) from %s", len(filenames), source_desc)
        return filenames

    if input_root is not None:
        LOGGER.info("Resolving all tar shards from local directory %s", input_root)
        filenames = sorted(p.name for p in input_root.glob("*.tar"))
        LOGGER.info("Resolved %d tar shard(s) from %s", len(filenames), input_root)
        return filenames

    LOGGER.info("Resolving all tar shards from dataset repo %s", repo_id)
    files = list_repo_files(repo_id, repo_type="dataset")
    filenames = sorted(f for f in files if f.endswith(".tar"))
    LOGGER.info("Resolved %d tar shard(s) from %s", len(filenames), repo_id)
    return filenames


def _member_key(name: str) -> Tuple[str, str]:
    base = Path(name).name
    if "." not in base:
        return base, ""
    stem, ext = base.rsplit(".", 1)
    return stem, ext.lower()


def _iter_wds_pairs_from_tarfile(tf: tarfile.TarFile) -> Iterator[Tuple[str, Dict]]:
    pending: Dict[str, Dict] = {}
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


def _iter_wds_pairs_from_stream(stream: BinaryIO) -> Iterator[Tuple[str, Dict]]:
    with tarfile.open(fileobj=stream, mode="r|*") as tf:
        yield from _iter_wds_pairs_from_tarfile(tf)


def _iter_wds_pairs_from_url(url: str, token: Optional[str], timeout: int) -> Iterator[Tuple[str, Dict]]:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        yield from _iter_wds_pairs_from_stream(response.raw)


def _iter_wds_pairs_from_path(tar_path: Path) -> Iterator[Tuple[str, Dict]]:
    with tar_path.open("rb") as stream:
        yield from _iter_wds_pairs_from_stream(stream)


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


def _shard_cache_paths(output_csv: Path, shard_name: str) -> Dict[str, Path]:
    shard_cache_dir = output_csv.parent / f"{output_csv.stem}_shards"
    safe_name = shard_name.replace("/", "__").replace("\\", "__")
    return {
        "rows_csv": shard_cache_dir / f"{safe_name}.csv",
        "summary_json": shard_cache_dir / f"{safe_name}.summary.json",
        "failures_csv": shard_cache_dir / f"{safe_name}.failures.csv",
    }


def _existing_image_size(out_path: Path) -> Tuple[int, int]:
    with Image.open(out_path) as im:
        return im.size


def _process_shard(
    repo_id: str,
    input_root: Optional[Path],
    shard_name: str,
    images_dir: Path,
    dataset_name: str,
    token: Optional[str],
    timeout: int,
    overwrite: bool,
    log_every: int,
    max_samples: Optional[int] = None,
) -> Tuple[str, List[Dict], int, List[Dict]]:
    source_mode = "hf"
    source_ref = ""
    pair_iter: Iterator[Tuple[str, Dict]]
    if input_root is not None:
        shard_path = Path(shard_name)
        if not shard_path.is_absolute():
            shard_path = input_root / shard_path
        source_mode = "local"
        source_ref = str(shard_path.resolve())
        pair_iter = _iter_wds_pairs_from_path(shard_path)
    else:
        source_ref = hf_hub_url(repo_id, shard_name, repo_type="dataset")
        pair_iter = _iter_wds_pairs_from_url(source_ref, token=token, timeout=timeout)
    shard_rows: List[Dict] = []
    failure_rows: List[Dict] = []
    total_seen = 0
    shard_stem = Path(shard_name).stem

    LOGGER.info("Starting shard %s (%s)", shard_name, source_mode)

    for key, record in pair_iter:
        total_seen += 1
        caption = str(record.get("caption", "")).strip()
        if not caption:
            continue

        ext = str(record.get("image_ext", "jpg")).lower()
        out_path = images_dir / shard_stem / f"{key}.{ext}"
        try:
            if out_path.exists() and not overwrite:
                try:
                    width, height = _existing_image_size(out_path)
                except Exception as exc:
                    LOGGER.warning(
                        "Shard %s key=%s existing image is unreadable; rewriting from tar payload (%s)",
                        shard_name,
                        key,
                        exc,
                    )
                    width, height = _save_image(record["image_bytes"], ext, out_path)
            else:
                width, height = _save_image(record["image_bytes"], ext, out_path)
        except Exception as exc:
            failure_rows.append(
                {
                    "dataset": dataset_name,
                    "shard": shard_name,
                    "key": key,
                    "member_name": record.get("member_name", ""),
                    "image_ext": ext,
                    "caption_sha1": hashlib.sha1(caption.encode("utf-8")).hexdigest(),
                    "source_url": source_ref,
                    "reason": str(exc),
                }
            )
            LOGGER.warning(
                "Shard %s key=%s skipped due to image decode/save error: %s",
                shard_name,
                key,
                exc,
            )
            continue

        shard_rows.append(
            {
                "dataset": dataset_name,
                "modality": "image",
                "caption": caption,
                "image_path": str(out_path),
                "media_path": str(out_path),
                "video_path": "",
                "source_id": _source_id(shard_name, key),
                "source_url": source_ref,
                "width": width,
                "height": height,
                "extra_json": json.dumps(
                    {
                        "shard": shard_name,
                        "source_mode": source_mode,
                        "member_name": record.get("member_name", ""),
                        "key": key,
                        "caption_sha1": hashlib.sha1(caption.encode("utf-8")).hexdigest(),
                    },
                    ensure_ascii=False,
                ),
            }
        )

        if log_every > 0 and total_seen % log_every == 0:
            LOGGER.info(
                "Shard %s progress: seen_pairs=%d written_rows=%d last_key=%s",
                shard_name,
                total_seen,
                len(shard_rows),
                key,
            )

        if max_samples is not None and len(shard_rows) >= max_samples:
            LOGGER.info(
                "Shard %s reached local max_samples=%d; stopping early",
                shard_name,
                max_samples,
            )
            break

    LOGGER.info(
        "Finished shard %s: seen_pairs=%d written_rows=%d failed_rows=%d",
        shard_name,
        total_seen,
        len(shard_rows),
        len(failure_rows),
    )
    return shard_name, shard_rows, total_seen, failure_rows


def _write_csv_rows(rows: List[Dict], output_csv: Path, fieldnames: List[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_shard_cache(
    output_csv: Path,
    shard_name: str,
    shard_rows: List[Dict],
    shard_seen: int,
    shard_failures: List[Dict],
) -> Dict:
    cache_paths = _shard_cache_paths(output_csv, shard_name)
    rows_csv = cache_paths["rows_csv"]
    summary_json = cache_paths["summary_json"]
    failures_csv = cache_paths["failures_csv"]

    _write_csv_rows(shard_rows, rows_csv, SHARD_FIELDNAMES)
    if shard_failures:
        _write_csv_rows(shard_failures, failures_csv, FAILURE_FIELDNAMES)
    elif failures_csv.exists():
        failures_csv.unlink()

    summary = {
        "status": "finished",
        "shard_name": shard_name,
        "rows_csv": str(rows_csv),
        "failures_csv": str(failures_csv) if shard_failures else None,
        "seen_pairs": int(shard_seen),
        "written_rows": int(len(shard_rows)),
        "failed_rows": int(len(shard_failures)),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_failures(failure_rows: List[Dict], output_csv: Path) -> Optional[Path]:
    if not failure_rows:
        return None
    fail_csv = output_csv.with_name(f"{output_csv.stem}_failures.csv")
    with fail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_FIELDNAMES)
        writer.writeheader()
        for row in failure_rows:
            writer.writerow(row)
    return fail_csv


def _load_shard_cache(output_csv: Path, shard_name: str) -> Optional[Dict]:
    cache_paths = _shard_cache_paths(output_csv, shard_name)
    summary_json = cache_paths["summary_json"]
    rows_csv = cache_paths["rows_csv"]
    if not summary_json.exists() or not rows_csv.exists():
        return None

    try:
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Ignoring invalid shard cache summary %s (%s)", summary_json, exc)
        return None

    if summary.get("status") != "finished":
        return None
    if str(summary.get("shard_name", "")) != str(shard_name):
        LOGGER.warning("Ignoring mismatched shard cache summary %s", summary_json)
        return None
    if not Path(str(summary.get("rows_csv", rows_csv))).exists():
        return None
    return summary


def _append_failures_from_csv(fail_csv_path: Path, writer: csv.DictWriter) -> int:
    count = 0
    if not fail_csv_path.exists():
        return count
    with fail_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            writer.writerow({key: row.get(key, "") for key in FAILURE_FIELDNAMES})
            count += 1
    return count


def _assemble_outputs(
    *,
    output_csv: Path,
    shard_names: List[str],
    shard_summaries: Dict[str, Dict],
    start_sample_idx: int,
    max_samples: Optional[int],
) -> Dict:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_seen = 0
    total_failed = 0
    sample_idx = int(start_sample_idx)
    stopped_early = False

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for shard_name in shard_names:
            summary = shard_summaries[shard_name]
            total_seen += int(summary.get("seen_pairs", 0))
            total_failed += int(summary.get("failed_rows", 0))
            rows_csv = Path(str(summary["rows_csv"]))
            with rows_csv.open("r", newline="", encoding="utf-8") as shard_f:
                reader = csv.DictReader(shard_f)
                for row in reader:
                    out_row = {key: row.get(key, "") for key in SHARD_FIELDNAMES}
                    out_row["sample_idx"] = sample_idx
                    writer.writerow({"sample_idx": out_row["sample_idx"], **{k: out_row[k] for k in SHARD_FIELDNAMES}})
                    sample_idx += 1
                    total_written += 1
                    if max_samples is not None and total_written >= max_samples:
                        stopped_early = True
                        break
            if stopped_early:
                break

    fail_csv = output_csv.with_name(f"{output_csv.stem}_failures.csv")
    failure_count_written = 0
    with fail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_FIELDNAMES)
        writer.writeheader()
        for shard_name in shard_names:
            summary = shard_summaries[shard_name]
            fail_path = summary.get("failures_csv")
            if not fail_path:
                continue
            failure_count_written += _append_failures_from_csv(Path(str(fail_path)), writer)
    if failure_count_written == 0:
        fail_csv.unlink()
        fail_csv_out: Optional[Path] = None
    else:
        fail_csv_out = fail_csv

    return {
        "total_seen_pairs": total_seen,
        "total_written": total_written,
        "total_failed": total_failed,
        "failed_csv": str(fail_csv_out) if fail_csv_out is not None else None,
        "stopped_early": stopped_early,
    }


def run_bootstrap(
    *,
    description: str,
    default_repo_id: str,
    default_output_root: str,
    default_dataset_name: str,
    default_filenames: str,
) -> None:
    _configure_logging()
    args = parse_args(
        description=description,
        default_repo_id=default_repo_id,
        default_output_root=default_output_root,
        default_dataset_name=default_dataset_name,
        default_filenames=default_filenames,
    )

    output_root = Path(args.output_root).resolve()
    input_root = Path(args.input_root).resolve() if args.input_root else None
    images_dir = output_root / "raw" / "images"
    default_csv_name = f"{args.dataset_name}_source.csv"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else output_root / "manifests" / default_csv_name

    filenames = _resolve_filenames(args.repo_id, args.filenames, input_root)
    sample_idx = int(args.start_sample_idx)
    source_mode = "local" if input_root is not None else "hf"
    source_desc = str(input_root) if input_root is not None else args.repo_id
    reused_shards = 0
    processed_shards = 0
    selected_filenames: List[str] = []
    shard_summaries: Dict[str, Dict] = {}
    pending_filenames: List[str] = []
    planned_written_rows = 0

    effective_jobs = max(1, int(args.jobs))
    if args.max_samples is not None and effective_jobs > 1:
        LOGGER.info(
            "max_samples=%s requires exact ordering; forcing sequential processing instead of jobs=%d",
            args.max_samples,
            effective_jobs,
        )
        effective_jobs = 1

    LOGGER.info(
        "Bootstrap start: source_mode=%s source=%s shards=%d output_root=%s jobs=%d max_samples=%s overwrite=%s",
        source_mode,
        source_desc,
        len(filenames),
        output_root,
        effective_jobs,
        args.max_samples,
        bool(args.overwrite),
    )

    if effective_jobs == 1:
        for shard_idx, shard_name in enumerate(filenames, start=1):
            if args.max_samples is not None and planned_written_rows >= int(args.max_samples):
                LOGGER.info("Reached max_samples=%d before starting shard %s", args.max_samples, shard_name)
                break

            cached_summary = None if args.overwrite else _load_shard_cache(output_csv, shard_name)
            if cached_summary is not None:
                shard_summaries[shard_name] = cached_summary
                selected_filenames.append(shard_name)
                reused_shards += 1
                planned_written_rows += int(cached_summary.get("written_rows", 0))
                LOGGER.info(
                    "Reusing cached shard %s (%d/%d): written_rows=%d failed_rows=%d",
                    shard_name,
                    shard_idx,
                    len(filenames),
                    int(cached_summary.get("written_rows", 0)),
                    int(cached_summary.get("failed_rows", 0)),
                )
                if args.max_samples is not None and planned_written_rows >= int(args.max_samples):
                    LOGGER.info("Reached max_samples=%d while reading cached shards", args.max_samples)
                    break
                continue

            _, shard_rows, shard_seen, shard_failures = _process_shard(
                repo_id=args.repo_id,
                input_root=input_root,
                shard_name=shard_name,
                images_dir=images_dir,
                dataset_name=args.dataset_name,
                token=args.hf_token,
                timeout=args.timeout,
                overwrite=args.overwrite,
                log_every=int(args.log_every),
                max_samples=None,
            )
            shard_summary = _write_shard_cache(
                output_csv=output_csv,
                shard_name=shard_name,
                shard_rows=shard_rows,
                shard_seen=shard_seen,
                shard_failures=shard_failures,
            )
            shard_summaries[shard_name] = shard_summary
            selected_filenames.append(shard_name)
            processed_shards += 1
            planned_written_rows += int(shard_summary.get("written_rows", 0))
            LOGGER.info(
                "Collected shard %s (%d/%d): seen_pairs=%d buffered_rows=%d failed_rows=%d",
                shard_name,
                shard_idx,
                len(filenames),
                shard_seen,
                len(shard_rows),
                len(shard_failures),
            )
            if args.max_samples is not None and planned_written_rows >= int(args.max_samples):
                LOGGER.info("Reached max_samples=%d while reading shards", args.max_samples)
                break
    else:
        for shard_name in filenames:
            cached_summary = None if args.overwrite else _load_shard_cache(output_csv, shard_name)
            if cached_summary is not None:
                shard_summaries[shard_name] = cached_summary
                selected_filenames.append(shard_name)
                reused_shards += 1
                LOGGER.info(
                    "Reusing cached shard %s: written_rows=%d failed_rows=%d",
                    shard_name,
                    int(cached_summary.get("written_rows", 0)),
                    int(cached_summary.get("failed_rows", 0)),
                )
            else:
                pending_filenames.append(shard_name)

        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_jobs) as executor:
            future_to_shard = {
                executor.submit(
                    _process_shard,
                    args.repo_id,
                    input_root,
                    shard_name,
                    images_dir,
                    args.dataset_name,
                    args.hf_token,
                    args.timeout,
                    args.overwrite,
                    int(args.log_every),
                ): shard_name
                for shard_name in pending_filenames
            }
            for completed_idx, future in enumerate(concurrent.futures.as_completed(future_to_shard), start=1):
                shard_name, shard_rows, shard_seen, shard_failures = future.result()
                shard_summary = _write_shard_cache(
                    output_csv=output_csv,
                    shard_name=shard_name,
                    shard_rows=shard_rows,
                    shard_seen=shard_seen,
                    shard_failures=shard_failures,
                )
                shard_summaries[shard_name] = shard_summary
                selected_filenames.append(shard_name)
                processed_shards += 1
                LOGGER.info(
                    "Collected shard %s (%d/%d completed): seen_pairs=%d buffered_rows=%d failed_rows=%d",
                    shard_name,
                    completed_idx,
                    len(pending_filenames),
                    shard_seen,
                    len(shard_rows),
                    len(shard_failures),
                )

    ordered_selected = [shard_name for shard_name in filenames if shard_name in shard_summaries and shard_name in set(selected_filenames)]
    assembly = _assemble_outputs(
        output_csv=output_csv,
        shard_names=ordered_selected,
        shard_summaries=shard_summaries,
        start_sample_idx=sample_idx,
        max_samples=args.max_samples,
    )
    summary = {
        "source_mode": source_mode,
        "input_root": str(input_root) if input_root is not None else None,
        "repo_id": args.repo_id,
        "filenames": ordered_selected,
        "output_root": str(output_root),
        "output_csv": str(output_csv),
        "total_seen_pairs": assembly["total_seen_pairs"],
        "total_written": assembly["total_written"],
        "total_failed": assembly["total_failed"],
        "failed_csv": assembly["failed_csv"],
        "jobs": effective_jobs,
        "stopped_early": assembly["stopped_early"],
        "reused_shards": reused_shards,
        "processed_shards": processed_shards,
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote manifest: %s", output_csv)
    if assembly["failed_csv"] is not None:
        LOGGER.warning("Wrote bootstrap failures CSV: %s", assembly["failed_csv"])
    print(json.dumps(summary, indent=2))
