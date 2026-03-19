import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .paths import DataLayout, ensure_layout

LOGGER = logging.getLogger(__name__)
PART_RE = re.compile(r"part_(\d+)")


def _list_videos(parts_root: Path) -> Dict[str, Dict[str, str]]:
    """Map video stem -> metadata (path + part_user)."""
    index: Dict[str, Dict[str, str]] = {}
    for video_path in sorted(parts_root.rglob("*.mp4")):
        stem = video_path.stem
        part_user = ""
        for p in video_path.parts:
            m = PART_RE.match(p)
            if m:
                part_user = m.group(1)
                break
        # Keep first path if duplicate stems exist
        if stem not in index:
            index[stem] = {
                "video_path": str(video_path),
                "part_user": part_user,
            }
    return index


def _normalize_video_key(video_value: str) -> str:
    p = Path(str(video_value))
    return p.stem if p.suffix else p.name


def build_manifest(
    layout: DataLayout,
    selected_parts_user: Optional[List[int]] = None,
    part_index_base: int = 0,
    output_name: str = "openvid_selected_parts.csv",
) -> Path:
    ensure_layout(layout)
    if not layout.csv_path.exists():
        raise FileNotFoundError(f"OpenVid CSV not found: {layout.csv_path}")

    LOGGER.info("Reading OpenVid CSV: %s", layout.csv_path)
    df = pd.read_csv(layout.csv_path)
    if "video" not in df.columns or "caption" not in df.columns:
        raise RuntimeError("CSV must contain columns: video, caption")

    LOGGER.info("Indexing extracted videos under: %s", layout.parts_root)
    video_index = _list_videos(layout.parts_root)
    LOGGER.info("Indexed %d extracted videos", len(video_index))

    parts_allow: Optional[set] = None
    if selected_parts_user:
        parts_allow = {f"{p:04d}" for p in selected_parts_user}

    rows = []
    missing = 0
    for idx, row in df.iterrows():
        key = _normalize_video_key(row["video"])
        info = video_index.get(key)
        if info is None:
            missing += 1
            continue

        if parts_allow is not None and info["part_user"] not in parts_allow:
            continue

        part_user_int = int(info["part_user"]) if info["part_user"] else -1
        part_remote_int = part_user_int - part_index_base if part_user_int >= 0 else -1

        rows.append(
            {
                "sample_idx": int(idx),
                "video": str(row["video"]),
                "caption": str(row["caption"]),
                "video_path": info["video_path"],
                "part_user": part_user_int,
                "part_remote": part_remote_int,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = layout.manifests_root / output_name
    out_df.to_csv(out_path, index=False)

    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_csv_rows": int(len(df)),
        "matched_rows": int(len(out_df)),
        "missing_rows": int(missing),
        "selected_parts_user": selected_parts_user or [],
        "part_index_base": part_index_base,
        "manifest_csv": str(out_path),
    }
    summary_path = layout.manifests_root / output_name.replace(".csv", ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info("Manifest saved: %s (%d rows)", out_path, len(out_df))
    LOGGER.info("Summary saved: %s", summary_path)
    return out_path
