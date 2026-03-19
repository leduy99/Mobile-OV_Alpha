#!/usr/bin/env python3
"""
Materialize media files for a unified manifest shard.

This script is designed for multi-process downloads:
- image rows are partitioned by sample id
- video rows are partitioned by video id so clips from the same source
  reuse the same full-video download within one worker
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


LOGGER = logging.getLogger("materialize_unified_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize media files for a unified manifest worker shard.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--modality", choices=["image", "video", "all"], default="all")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--url-timeout-s", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--cleanup-full-video", action="store_true")
    parser.add_argument("--cookies-file", type=Path, default=None)
    parser.add_argument("--cookies-from-browser", default="")
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _stable_partition(value: str, num_workers: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, num_workers)


def _safe_name(value: str, default_stem: str, default_ext: str) -> str:
    text = (value or "").strip()
    if not text:
        return f"{default_stem}{default_ext}"
    name = Path(text).name.replace("/", "_")
    stem = Path(name).stem or default_stem
    ext = Path(name).suffix or default_ext
    return f"{stem}{ext}"


def _extension_from_url(url: str, default_ext: str = ".jpg") -> str:
    try:
        ext = Path(urlparse(url).path).suffix.lower()
        if ext and len(ext) <= 5:
            return ext
    except Exception:
        pass
    return default_ext


def _download_url_to_path(url: str, dst_path: Path, timeout_s: int, max_retries: int) -> bool:
    if not url:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(max_retries + 1):
        try:
            with urlopen(req, timeout=timeout_s) as resp, dst_path.open("wb") as out_f:
                shutil.copyfileobj(resp, out_f, length=4 * 1024 * 1024)
            return dst_path.exists() and dst_path.stat().st_size > 0
        except (URLError, TimeoutError, OSError, ValueError, Exception):
            if dst_path.exists():
                try:
                    dst_path.unlink()
                except OSError:
                    pass
            if attempt >= max_retries:
                return False
            time.sleep(0.2 * (attempt + 1))
    return False


def _run(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _hhmmss_to_seconds(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        hh, mm, ss = value.split(":")
        return int(hh) * 3600.0 + int(mm) * 60.0 + float(ss)
    except Exception:
        return None


def _resolve_downloaded_video_path(requested_path: Path) -> Optional[Path]:
    candidates = [requested_path]
    candidates.extend(
        sorted(
            requested_path.parent.glob(f"{requested_path.name}.*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    candidates.extend(
        sorted(
            requested_path.parent.glob(f"{requested_path.stem}.*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            candidate.exists()
            and candidate.is_file()
            and candidate.stat().st_size > 0
            and candidate.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
        ):
            return candidate
    return None


def _download_youtube_video(
    url: str,
    out_path: Path,
    timeout_s: int,
    cookies_file: Optional[Path],
    cookies_from_browser: str,
) -> Optional[Path]:
    if shutil.which("yt-dlp") is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]/bestvideo[height<=480]+bestaudio/best[height<=480]/best",
        "-o",
        str(out_path),
        "--no-playlist",
        "--retries",
        "1",
        "--fragment-retries",
        "1",
        "--extractor-retries",
        "1",
        "--socket-timeout",
        str(timeout_s),
        "--skip-unavailable-fragments",
    ]
    if cookies_file and cookies_file.exists():
        cmd += ["--cookies", str(cookies_file)]
    elif cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd.append(url)
    if not _run(cmd):
        return None
    return _resolve_downloaded_video_path(out_path)


def _clip_video_ffmpeg(src_video: Path, dst_clip: Path, start_ts: str, end_ts: str) -> bool:
    if not start_ts or not end_ts or shutil.which("ffmpeg") is None:
        return False
    dst_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        start_ts,
        "-to",
        end_ts,
        "-i",
        str(src_video),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(dst_clip),
    ]
    return _run(cmd) and dst_clip.exists() and dst_clip.stat().st_size > 0


def _read_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_rows(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str], summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _row_matches_modality(row: Dict[str, str], modality: str) -> bool:
    if modality == "all":
        return True
    return (row.get("modality") or "").strip().lower() == modality


def _video_partition_key(row: Dict[str, str]) -> str:
    extra_json = row.get("extra_json") or ""
    video_id = ""
    if extra_json:
        try:
            extra = json.loads(extra_json)
            video_id = str(extra.get("video_id", "") or "").strip()
        except json.JSONDecodeError:
            video_id = ""
    if video_id:
        return video_id
    source_id = (row.get("source_id") or "").strip()
    if source_id:
        return source_id.split(".")[0]
    source_url = (row.get("source_url") or "").strip()
    if source_url:
        return source_url
    return str(row.get("sample_idx") or "0")


def _assign_rows(rows: List[Dict[str, str]], modality: str, worker_id: int, num_workers: int) -> List[Dict[str, str]]:
    assigned = []
    for row in rows:
        if not _row_matches_modality(row, modality):
            continue
        if (row.get("modality") or "").strip().lower() == "video":
            key = _video_partition_key(row)
        else:
            key = str(row.get("sample_idx") or row.get("source_id") or "")
        if _stable_partition(key, num_workers) == worker_id:
            assigned.append(dict(row))
    return assigned


def materialize_images(
    rows: List[Dict[str, str]],
    dataset_root: Path,
    timeout_s: int,
    max_retries: int,
    log_every: int,
) -> Dict[str, int]:
    images_root = dataset_root / "raw" / "media" / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    existing = 0
    missing = 0
    for idx, row in enumerate(rows, start=1):
        source_url = (row.get("source_url") or "").strip()
        existing_path = (row.get("image_path") or row.get("media_path") or "").strip()
        if existing_path and Path(existing_path).exists():
            row["image_path"] = existing_path
            row["media_path"] = existing_path
            existing += 1
        else:
            source_id = _safe_name(row.get("source_id", ""), f"sample_{row.get('sample_idx', idx)}", _extension_from_url(source_url))
            out_path = images_root / source_id
            if not out_path.suffix:
                out_path = out_path.with_suffix(_extension_from_url(source_url))
            if out_path.exists() and out_path.stat().st_size > 0:
                row["image_path"] = str(out_path)
                row["media_path"] = str(out_path)
                existing += 1
            elif source_url and _download_url_to_path(source_url, out_path, timeout_s, max_retries):
                row["image_path"] = str(out_path)
                row["media_path"] = str(out_path)
                downloaded += 1
            else:
                missing += 1
        if log_every > 0 and idx % log_every == 0:
            LOGGER.info(
                "Image worker progress | processed=%d downloaded=%d existing=%d missing=%d",
                idx,
                downloaded,
                existing,
                missing,
            )
    return {"downloaded": downloaded, "existing": existing, "missing": missing}


def materialize_videos(
    rows: List[Dict[str, str]],
    dataset_root: Path,
    timeout_s: int,
    cookies_file: Optional[Path],
    cookies_from_browser: str,
    cleanup_full_video: bool,
    log_every: int,
) -> Dict[str, int]:
    videos_root = dataset_root / "raw" / "media" / "videos"
    cache_root = dataset_root / "raw" / "media" / "videos_full_cache"
    videos_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    by_video_id: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video_id[_video_partition_key(row)].append(row)

    downloaded = 0
    existing = 0
    missing = 0
    grouped_items = list(by_video_id.items())
    for group_idx, (video_id, group_rows) in enumerate(grouped_items, start=1):
        source_url = (group_rows[0].get("source_url") or "").strip()
        full_video = cache_root / _safe_name(video_id, f"video_{group_idx:08d}", ".mp4")
        existing_full_video = _resolve_downloaded_video_path(full_video)
        if existing_full_video is not None:
            full_video = existing_full_video
            full_ok = True
            downloaded_now = False
        else:
            downloaded_path = None
            if source_url:
                downloaded_path = _download_youtube_video(
                    source_url,
                    full_video,
                    timeout_s=timeout_s,
                    cookies_file=cookies_file,
                    cookies_from_browser=cookies_from_browser,
                )
            full_ok = bool(downloaded_path)
            if downloaded_path is not None:
                full_video = downloaded_path
            downloaded_now = full_ok
        if downloaded_now:
            downloaded += 1
        full_path_used = False

        for row in group_rows:
            current_path = (row.get("video_path") or row.get("media_path") or "").strip()
            if current_path and Path(current_path).exists():
                row["video_path"] = current_path
                row["media_path"] = current_path
                existing += 1
                continue
            if not full_ok or not full_video.exists():
                missing += 1
                continue

            clip_name = _safe_name(row.get("source_id", ""), f"clip_{row.get('sample_idx', '0')}", ".mp4")
            clip_path = videos_root / clip_name
            span_start = (row.get("span_start") or "").strip()
            span_end = (row.get("span_end") or "").strip()
            start_sec = _hhmmss_to_seconds(span_start)
            end_sec = _hhmmss_to_seconds(span_end)
            if (
                span_start
                and span_end
                and start_sec is not None
                and end_sec is not None
                and end_sec > start_sec
            ):
                if clip_path.exists() and clip_path.stat().st_size > 0:
                    row["video_path"] = str(clip_path)
                    row["media_path"] = str(clip_path)
                    existing += 1
                elif _clip_video_ffmpeg(full_video, clip_path, span_start, span_end):
                    row["video_path"] = str(clip_path)
                    row["media_path"] = str(clip_path)
                else:
                    row["video_path"] = str(full_video)
                    row["media_path"] = str(full_video)
                    full_path_used = True
            else:
                row["video_path"] = str(full_video)
                row["media_path"] = str(full_video)
                full_path_used = True

        if cleanup_full_video and full_video.exists() and not full_path_used:
            try:
                full_video.unlink()
            except OSError:
                pass

        if log_every > 0 and group_idx % log_every == 0:
            LOGGER.info(
                "Video worker progress | groups=%d/%d downloaded_full=%d existing=%d missing=%d",
                group_idx,
                len(grouped_items),
                downloaded,
                existing,
                missing,
            )

    return {"downloaded": downloaded, "existing": existing, "missing": missing}


def main() -> int:
    args = parse_args()
    setup_logging()

    if args.worker_id < 0 or args.worker_id >= max(1, args.num_workers):
        raise ValueError(f"worker_id must be in [0, {max(1, args.num_workers) - 1}], got {args.worker_id}")

    cookies_file = args.cookies_file
    if cookies_file is None:
        env_cookies = os.getenv("YTDLP_COOKIES_FILE", "").strip()
        cookies_file = Path(env_cookies) if env_cookies else None
    cookies_from_browser = args.cookies_from_browser or os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()

    rows, fieldnames = _read_rows(args.manifest_csv)
    assigned_rows = _assign_rows(rows, args.modality, args.worker_id, args.num_workers)
    if not assigned_rows:
        LOGGER.warning("No rows assigned to worker %s for modality=%s", args.worker_id, args.modality)

    stats = {"downloaded": 0, "existing": 0, "missing": 0}
    if args.modality in {"image", "all"}:
        image_rows = [r for r in assigned_rows if (r.get("modality") or "").strip().lower() == "image"]
        if image_rows:
            image_stats = materialize_images(
                rows=image_rows,
                dataset_root=args.dataset_root,
                timeout_s=args.url_timeout_s,
                max_retries=args.max_retries,
                log_every=args.log_every,
            )
            for key, value in image_stats.items():
                stats[key] += value

    if args.modality in {"video", "all"}:
        video_rows = [r for r in assigned_rows if (r.get("modality") or "").strip().lower() == "video"]
        if video_rows:
            video_stats = materialize_videos(
                rows=video_rows,
                dataset_root=args.dataset_root,
                timeout_s=args.url_timeout_s,
                cookies_file=cookies_file,
                cookies_from_browser=cookies_from_browser,
                cleanup_full_video=args.cleanup_full_video,
                log_every=args.log_every,
            )
            for key, value in video_stats.items():
                stats[key] += value

    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_csv": str(args.manifest_csv),
        "output_manifest": str(args.output_manifest),
        "dataset_root": str(args.dataset_root),
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "modality": args.modality,
        "assigned_rows": len(assigned_rows),
        "downloaded_media": int(stats["downloaded"]),
        "existing_media": int(stats["existing"]),
        "missing_media": int(stats["missing"]),
    }
    _write_rows(args.output_manifest, assigned_rows, fieldnames, summary)
    LOGGER.info(
        "Worker finished | output=%s assigned=%d downloaded=%d existing=%d missing=%d",
        args.output_manifest,
        len(assigned_rows),
        stats["downloaded"],
        stats["existing"],
        stats["missing"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
