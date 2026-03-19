import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen


LOGGER = logging.getLogger(__name__)

OPENVID_DATASET_TREE_URL = "https://huggingface.co/api/datasets/nkp37/OpenVid-1M/tree/main?recursive=true"

ZIP_RE = re.compile(r"OpenVid_part(\d+)\.zip$")
SPLIT_RE = re.compile(r"OpenVid_part(\d+)_part([a-z]+)$")


def _normalize_index(raw_index: dict) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for k, v in raw_index.items():
        try:
            part_id = int(k)
        except (TypeError, ValueError):
            continue
        zip_name = None
        split_suffixes: List[str] = []
        if isinstance(v, dict):
            zip_name = v.get("zip")
            split_suffixes = v.get("split_suffixes", [])
        if not isinstance(split_suffixes, list):
            split_suffixes = []
        split_suffixes = sorted({str(s) for s in split_suffixes if str(s)})
        out[part_id] = {
            "zip": str(zip_name) if zip_name else None,
            "split_suffixes": split_suffixes,
        }
    return out


def _build_index_from_tree(entries: List[dict]) -> Dict[int, dict]:
    index: Dict[int, dict] = {}

    def _ensure(part_id: int) -> dict:
        if part_id not in index:
            index[part_id] = {"zip": None, "split_suffixes": []}
        return index[part_id]

    for item in entries:
        path = item.get("path", "")
        if not isinstance(path, str):
            continue

        m_zip = ZIP_RE.fullmatch(path)
        if m_zip:
            pid = int(m_zip.group(1))
            slot = _ensure(pid)
            slot["zip"] = path
            continue

        m_split = SPLIT_RE.fullmatch(path)
        if m_split:
            pid = int(m_split.group(1))
            suffix = m_split.group(2)
            slot = _ensure(pid)
            if suffix not in slot["split_suffixes"]:
                slot["split_suffixes"].append(suffix)

    for pid in list(index.keys()):
        index[pid]["split_suffixes"] = sorted(index[pid]["split_suffixes"])
    return index


def fetch_remote_index(timeout: int = 60) -> Dict[int, dict]:
    with urlopen(OPENVID_DATASET_TREE_URL, timeout=timeout) as resp:
        entries = json.load(resp)
    if not isinstance(entries, list):
        raise RuntimeError("Unexpected HuggingFace API response for OpenVid dataset tree.")
    return _build_index_from_tree(entries)


def load_remote_index_with_cache(
    cache_path: Path,
    max_age_seconds: int = 6 * 3600,
    timeout: int = 60,
) -> Dict[int, dict]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()

    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            fetched_at_epoch = float(payload.get("fetched_at_epoch", 0.0))
            raw_index = payload.get("index", {})
            cached_index = _normalize_index(raw_index if isinstance(raw_index, dict) else {})
            if cached_index and now - fetched_at_epoch <= max_age_seconds:
                return cached_index
        except Exception:
            LOGGER.warning("Failed to parse OpenVid index cache, will refresh: %s", cache_path)

    try:
        fresh = fetch_remote_index(timeout=timeout)
        cache_payload = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "fetched_at_epoch": now,
            "index": {str(k): v for k, v in fresh.items()},
        }
        cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
        return fresh
    except (URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                raw_index = payload.get("index", {})
                cached_index = _normalize_index(raw_index if isinstance(raw_index, dict) else {})
                if cached_index:
                    LOGGER.warning(
                        "Failed to refresh OpenVid index from HF (%s). Using stale cache: %s",
                        exc,
                        cache_path,
                    )
                    return cached_index
            except Exception:
                pass
        raise RuntimeError(f"Failed to fetch OpenVid index from HuggingFace and no valid cache exists: {exc}") from exc


def list_remote_part_ids(index: Dict[int, dict]) -> List[int]:
    return sorted(index.keys())


def split_suffixes_for_part(index: Optional[Dict[int, dict]], remote_id: int) -> List[str]:
    if not index:
        return []
    item = index.get(int(remote_id))
    if not isinstance(item, dict):
        return []
    suffixes = item.get("split_suffixes", [])
    if not isinstance(suffixes, list):
        return []
    return sorted({str(s) for s in suffixes if str(s)})
