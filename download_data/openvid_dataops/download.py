import json
import logging
import shutil
import subprocess
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .index import list_remote_part_ids, load_remote_index_with_cache, split_suffixes_for_part
from .part_spec import parts_to_remote_ids
from .paths import DataLayout, ensure_layout

LOGGER = logging.getLogger(__name__)

OPENVID_CSV_URL = (
    "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv"
)
OPENVID_PART_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{remote_id}.zip"
OPENVID_PART_SPLIT_URL = (
    "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{remote_id}_part{suffix}"
)


@dataclass
class DownloadRecord:
    part_user: int
    part_remote: int
    zip_path: str
    extracted_dir: str
    downloaded: bool
    extracted: bool
    source_type: str = "zip"
    source_files: List[str] = field(default_factory=list)
    ts_utc: str = ""


def _run(cmd: List[str]) -> None:
    LOGGER.debug("Run: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _download_with_wget(url: str, out_path: Path) -> bool:
    if shutil.which("wget") is None:
        raise RuntimeError("wget not found. Please install wget.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run(["wget", "-c", url, "-O", str(out_path)])
        return True
    except subprocess.CalledProcessError as exc:
        LOGGER.warning("Download failed: %s (%s)", url, exc)
        return False


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    if shutil.which("unzip") is None:
        raise RuntimeError("unzip not found. Please install unzip.")
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(["unzip", "-q", "-n", str(zip_path), "-d", str(out_dir)])


def _is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # Read zip central directory to ensure archive is not obviously corrupted.
            _ = zf.infolist()
        return True
    except zipfile.BadZipFile:
        return False


def _clean_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _merge_split_files(split_files: List[Path], out_zip: Path) -> None:
    tmp_merged = out_zip.with_suffix(out_zip.suffix + ".merge_tmp")
    with tmp_merged.open("wb") as out_f:
        for fp in split_files:
            with fp.open("rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=8 * 1024 * 1024)
    tmp_merged.replace(out_zip)


def _download_split_and_merge(
    remote_id: int,
    split_suffixes: List[str],
    zip_path: Path,
) -> Tuple[bool, List[str]]:
    if not split_suffixes:
        return False, []

    split_files: List[Path] = []
    split_names: List[str] = []
    try:
        for suffix in split_suffixes:
            split_name = f"OpenVid_part{remote_id}_part{suffix}"
            split_url = OPENVID_PART_SPLIT_URL.format(remote_id=remote_id, suffix=suffix)
            split_path = zip_path.parent / split_name
            LOGGER.info("Downloading split chunk: %s", split_name)
            ok = _download_with_wget(split_url, split_path)
            if not ok or not split_path.exists() or split_path.stat().st_size <= 0:
                LOGGER.warning("Split chunk unavailable: %s", split_name)
                return False, []
            split_files.append(split_path)
            split_names.append(split_name)

        _merge_split_files(split_files, zip_path)
        if not _is_valid_zip(zip_path):
            LOGGER.warning("Merged split archive is invalid: %s", zip_path)
            _clean_file(zip_path)
            return False, []
        return True, split_names
    finally:
        # Always cleanup split chunks after merge attempt.
        for fp in split_files:
            _clean_file(fp)


def _download_part_with_fallback(
    remote_id: int,
    zip_path: Path,
    remote_index: Optional[Dict[int, dict]],
) -> Tuple[bool, bool, str, List[str]]:
    zip_name = zip_path.name
    if _is_valid_zip(zip_path):
        LOGGER.info("ZIP already exists and looks valid: %s", zip_path)
        return True, False, "zip", [zip_name]

    if zip_path.exists():
        LOGGER.warning("Existing ZIP is invalid/stale, removing: %s", zip_path)
        _clean_file(zip_path)

    zip_url = OPENVID_PART_URL.format(remote_id=remote_id)
    LOGGER.info("Downloading ZIP part remote=%s", remote_id)
    if _download_with_wget(zip_url, zip_path) and _is_valid_zip(zip_path):
        return True, True, "zip", [zip_name]

    LOGGER.warning("ZIP path not available for part %s, trying split fallback", remote_id)
    _clean_file(zip_path)

    split_suffixes = split_suffixes_for_part(remote_index, remote_id)
    if not split_suffixes:
        # HF index unavailable or missing entry; probe canonical first two split chunks.
        split_suffixes = ["aa", "ab"]
    ok_split, split_names = _download_split_and_merge(
        remote_id=remote_id,
        split_suffixes=split_suffixes,
        zip_path=zip_path,
    )
    if ok_split:
        return True, True, "split", split_names

    return False, False, "", []


def download_openvid(
    layout: DataLayout,
    parts_user: Optional[List[int]],
    part_index_base: int,
    extract: bool,
    keep_zip: bool,
    include_csv: bool,
) -> None:
    ensure_layout(layout)

    records: List[DownloadRecord] = []

    if include_csv:
        if layout.csv_path.exists():
            LOGGER.info("CSV exists, skip: %s", layout.csv_path)
        else:
            LOGGER.info("Downloading CSV -> %s", layout.csv_path)
            if not _download_with_wget(OPENVID_CSV_URL, layout.csv_path):
                raise RuntimeError(f"Failed to download OpenVid CSV: {OPENVID_CSV_URL}")

    index_cache_path = layout.state_root / "hf_openvid_index_cache.json"
    remote_index: Optional[Dict[int, dict]] = None
    try:
        remote_index = load_remote_index_with_cache(index_cache_path)
    except Exception as exc:
        LOGGER.warning("Could not load HF OpenVid index cache: %s", exc)

    if parts_user is None:
        if remote_index is None:
            raise RuntimeError(
                "parts='all' requires HuggingFace index lookup but index fetch failed. "
                "Please retry with network or provide explicit --parts list."
            )
        remote_ids = list_remote_part_ids(remote_index)
        parts_user = [rid + part_index_base for rid in remote_ids]
        LOGGER.info(
            "Resolved all parts from HF index: count=%d remote_range=%d..%d",
            len(remote_ids),
            remote_ids[0] if remote_ids else -1,
            remote_ids[-1] if remote_ids else -1,
        )
    else:
        remote_ids = parts_to_remote_ids(parts_user, part_index_base=part_index_base)

    for part_user, remote_id in zip(parts_user, remote_ids):
        zip_name = f"OpenVid_part{remote_id}.zip"
        zip_path = layout.zip_root / zip_name
        extracted_dir = layout.parts_root / f"part_{part_user:04d}"

        ok, downloaded, source_type, source_files = _download_part_with_fallback(
            remote_id=remote_id,
            zip_path=zip_path,
            remote_index=remote_index,
        )
        if not ok:
            raise RuntimeError(
                f"Failed to download part user={part_user} remote={remote_id}. "
                f"Tried ZIP + split fallback."
            )

        extracted_ok = False
        if extract:
            LOGGER.info("Extracting %s -> %s", zip_path.name, extracted_dir)
            _extract_zip(zip_path, extracted_dir)
            extracted_ok = True
            if not keep_zip:
                _clean_file(zip_path)

        records.append(
            DownloadRecord(
                part_user=part_user,
                part_remote=remote_id,
                zip_path=str(zip_path),
                extracted_dir=str(extracted_dir),
                downloaded=downloaded,
                extracted=extracted_ok,
                source_type=source_type,
                source_files=source_files,
                ts_utc=datetime.now(timezone.utc).isoformat(),
            )
        )

    state_path = layout.state_root / "download_state.json"
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "part_index_base": part_index_base,
        "resolved_parts_user": parts_user,
        "resolved_parts_remote": remote_ids,
        "records": [asdict(r) for r in records],
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved state: %s", state_path)
