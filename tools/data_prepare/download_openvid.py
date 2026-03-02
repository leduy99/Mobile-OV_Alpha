#!/usr/bin/env python3
"""
Script to download OpenVid-1M dataset from HuggingFace.

Usage:
    python tools/data_prepare/download_openvid.py --output_dir /path/to/output --num_parts 10
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENVID_DATASET_TREE_URL = "https://huggingface.co/api/datasets/nkp37/OpenVid-1M/tree/main?recursive=true"
OPENVID_CSV_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv"
OPENVID_ZIP_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{}.zip"
OPENVID_SPLIT_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{}_part{}"

ZIP_RE = re.compile(r"OpenVid_part(\d+)\.zip$")
SPLIT_RE = re.compile(r"OpenVid_part(\d+)_part([a-z]+)$")


def _is_valid_zip(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            _ = zf.infolist()
        return True
    except zipfile.BadZipFile:
        return False


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def download_file(url: str, output_path: str) -> bool:
    """Download a file using wget."""
    logger.info("Downloading %s -> %s", url, output_path)
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subprocess.run(["wget", "-c", url, "-O", output_path], check=True)
        logger.info("Downloaded %s", output_path)
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        return False
    except FileNotFoundError:
        logger.error("wget not found. Please install wget or use alternative download method.")
        return False


def _merge_split_files(split_paths: List[str], out_zip: str) -> bool:
    tmp_zip = out_zip + ".merge_tmp"
    try:
        with open(tmp_zip, "wb") as out_f:
            for p in split_paths:
                with open(p, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=8 * 1024 * 1024)
        os.replace(tmp_zip, out_zip)
        return _is_valid_zip(out_zip)
    finally:
        _safe_remove(tmp_zip)
        for p in split_paths:
            _safe_remove(p)


def _load_remote_index(cache_path: str, max_age_seconds: int = 6 * 3600) -> Dict[int, dict]:
    now = time.time()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            age = now - float(payload.get("fetched_at_epoch", 0.0))
            if age <= max_age_seconds:
                raw = payload.get("index", {})
                return {
                    int(k): {
                        "zip": (v.get("zip") if isinstance(v, dict) else None),
                        "split_suffixes": sorted(
                            set(v.get("split_suffixes", [])) if isinstance(v, dict) else set()
                        ),
                    }
                    for k, v in raw.items()
                }
        except Exception:
            logger.warning("Invalid OpenVid index cache, refreshing: %s", cache_path)

    with urllib.request.urlopen(OPENVID_DATASET_TREE_URL, timeout=60) as resp:
        entries = json.load(resp)
    if not isinstance(entries, list):
        raise RuntimeError("Unexpected HuggingFace API response for OpenVid index")

    index: Dict[int, dict] = {}
    for item in entries:
        path = item.get("path", "")
        if not isinstance(path, str):
            continue
        m_zip = ZIP_RE.fullmatch(path)
        if m_zip:
            pid = int(m_zip.group(1))
            index.setdefault(pid, {"zip": None, "split_suffixes": []})
            index[pid]["zip"] = path
            continue
        m_split = SPLIT_RE.fullmatch(path)
        if m_split:
            pid = int(m_split.group(1))
            suffix = m_split.group(2)
            index.setdefault(pid, {"zip": None, "split_suffixes": []})
            if suffix not in index[pid]["split_suffixes"]:
                index[pid]["split_suffixes"].append(suffix)

    for pid in list(index.keys()):
        index[pid]["split_suffixes"] = sorted(index[pid]["split_suffixes"])

    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fetched_at_epoch": now,
        "index": {str(k): v for k, v in index.items()},
    }
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return index


def download_openvid_csv(output_dir: str):
    """Download OpenVid-1M.csv file."""
    csv_path = os.path.join(output_dir, "OpenVid-1M.csv")
    if os.path.exists(csv_path):
        logger.info("CSV file already exists: %s", csv_path)
        return csv_path
    os.makedirs(output_dir, exist_ok=True)
    if download_file(OPENVID_CSV_URL, csv_path):
        return csv_path
    return None


def _download_part_with_fallback(output_dir: str, part_num: int, remote_index: Optional[Dict[int, dict]]) -> bool:
    zip_path = os.path.join(output_dir, f"OpenVid_part{part_num}.zip")

    if _is_valid_zip(zip_path):
        logger.info("Part %s already exists and valid: %s", part_num, zip_path)
        return True
    if os.path.exists(zip_path):
        logger.warning("Part %s had invalid zip, removing stale file: %s", part_num, zip_path)
        _safe_remove(zip_path)

    zip_url = OPENVID_ZIP_URL.format(part_num)
    if download_file(zip_url, zip_path) and _is_valid_zip(zip_path):
        logger.info("Part %s downloaded from ZIP", part_num)
        return True

    _safe_remove(zip_path)
    split_suffixes: List[str] = []
    if remote_index and part_num in remote_index:
        split_suffixes = list(remote_index[part_num].get("split_suffixes", []))
    if not split_suffixes:
        split_suffixes = ["aa", "ab"]

    split_paths: List[str] = []
    for suffix in split_suffixes:
        split_url = OPENVID_SPLIT_URL.format(part_num, suffix)
        split_path = os.path.join(output_dir, f"OpenVid_part{part_num}_part{suffix}")
        if not download_file(split_url, split_path) or not os.path.exists(split_path) or os.path.getsize(split_path) <= 0:
            logger.warning("Missing split chunk for part %s: suffix=%s", part_num, suffix)
            for p in split_paths:
                _safe_remove(p)
            _safe_remove(split_path)
            return False
        split_paths.append(split_path)

    if _merge_split_files(split_paths, zip_path):
        logger.info("Part %s downloaded via split fallback and merged to %s", part_num, zip_path)
        return True

    logger.warning("Part %s split merge produced invalid zip", part_num)
    _safe_remove(zip_path)
    return False


def download_openvid_videos(output_dir: str, num_parts: int = None, start_part: int = 0):
    """
    Download OpenVid-1M video parts.

    Args:
        output_dir: Directory to save downloaded files
        num_parts: Number of parts to download (None = all available from HF index)
        start_part: Starting part number
    """
    os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.join(output_dir, ".openvid_hf_index_cache.json")
    remote_index = None
    try:
        remote_index = _load_remote_index(cache_path)
    except Exception as exc:
        logger.warning("Failed to fetch OpenVid index from HF (%s). Using fallback probing.", exc)

    if num_parts is None:
        if remote_index is None:
            raise RuntimeError(
                "num_parts=None requires HuggingFace index lookup, but lookup failed. "
                "Please set --num_parts explicitly or retry with network."
            )
        target_parts = [p for p in sorted(remote_index.keys()) if p >= start_part]
    else:
        target_parts = list(range(start_part, start_part + num_parts))

    logger.info(
        "Starting OpenVid downloads: count=%d first=%s last=%s",
        len(target_parts),
        target_parts[0] if target_parts else None,
        target_parts[-1] if target_parts else None,
    )

    downloaded = 0
    failed = 0
    for part_num in target_parts:
        ok = _download_part_with_fallback(output_dir, part_num, remote_index)
        if ok:
            downloaded += 1
        else:
            failed += 1
            logger.warning("Part %s failed after ZIP + split fallback", part_num)

    logger.info("Downloaded=%d Failed=%d Total=%d", downloaded, failed, len(target_parts))
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download OpenVid-1M dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_parts", type=int, default=None, help="Number of parts to download (None = all)")
    parser.add_argument("--start_part", type=int, default=0, help="Starting part number")
    parser.add_argument("--csv_only", action="store_true", help="Only download CSV file")

    args = parser.parse_args()

    # Download CSV
    csv_path = download_openvid_csv(args.output_dir)
    if not csv_path:
        logger.error("Failed to download CSV file")
        return

    if args.csv_only:
        logger.info("CSV-only mode, skipping video downloads")
        return

    # Download videos
    if args.num_parts is None:
        logger.info("Starting video download from part %d to all available parts", args.start_part)
    else:
        logger.info("Starting video download (parts %d to %d)", args.start_part, args.start_part + args.num_parts - 1)
    download_openvid_videos(args.output_dir, args.num_parts, args.start_part)

    logger.info("Download completed!")
    logger.info("CSV file: %s", csv_path)
    logger.info("Video parts: %s/OpenVid_part*.zip", args.output_dir)


if __name__ == "__main__":
    main()
