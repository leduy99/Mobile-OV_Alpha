#!/usr/bin/env python3
"""
Recover/download LAION images with unique file names to avoid overwrite collisions.

Why:
- Existing LAION media paths can collide (same filename reused for different URLs),
  causing image-caption mismatch and downstream collapse.
- This script writes each LAION sample to a unique file name keyed by sample_idx.

Input:
- A manifest CSV containing at least:
  sample_idx, dataset, modality, source_url, caption

Output:
- A recovered LAION manifest with updated image_path/media_path/video_path
- Optional failures CSV and summary JSON
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image, UnidentifiedImageError


@dataclass
class RecoverResult:
    sample_idx: int
    ok: bool
    reason: str
    output_path: str
    width: int = -1
    height: int = -1
    bytes: int = -1


def _safe_int(v, default: int = -1) -> int:
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        return default


def _is_http_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def _valid_local_image(path: Path, min_side: int, min_bytes: int) -> Tuple[bool, str, int, int, int]:
    if not path.exists():
        return False, "missing_file", -1, -1, -1
    try:
        b = path.stat().st_size
        if b < min_bytes:
            return False, "too_small_bytes", -1, -1, b
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
        if min(w, h) < min_side:
            return False, "too_small_resolution", w, h, b
        return True, "ok", w, h, b
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"invalid_image:{type(exc).__name__}", -1, -1, -1


def _download_one(
    url: str,
    out_path: Path,
    timeout: float,
    retries: int,
    min_side: int,
    min_bytes: int,
) -> RecoverResult:
    sample_idx = int(out_path.stem.split("_")[-1]) if "_" in out_path.stem else -1

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Connection": "close",
    }

    for attempt in range(retries + 1):
        try:
            req = Request(url=url, headers=headers, method="GET")
            context = ssl.create_default_context()
            with urlopen(req, timeout=timeout, context=context) as resp:
                raw = resp.read()

            if len(raw) < min_bytes:
                raise ValueError(f"too_small_bytes:{len(raw)}")

            # Decode + normalize to RGB JPEG to ensure deterministic readable files.
            with Image.open(BytesIO(raw)) as im:
                im = im.convert("RGB")
                w, h = im.size
                if min(w, h) < min_side:
                    raise ValueError(f"too_small_resolution:{w}x{h}")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                im.save(out_path, format="JPEG", quality=95, optimize=True)

            ok, reason, w, h, b = _valid_local_image(out_path, min_side=min_side, min_bytes=min_bytes)
            if not ok:
                raise ValueError(reason)
            return RecoverResult(
                sample_idx=sample_idx,
                ok=True,
                reason="ok",
                output_path=str(out_path),
                width=w,
                height=h,
                bytes=b,
            )
        except (HTTPError, URLError, socket.timeout, TimeoutError, UnidentifiedImageError, OSError, ValueError) as exc:
            if attempt >= retries:
                return RecoverResult(
                    sample_idx=sample_idx,
                    ok=False,
                    reason=f"{type(exc).__name__}:{str(exc)[:200]}",
                    output_path=str(out_path),
                )
            continue
        except Exception as exc:  # pylint: disable=broad-except
            if attempt >= retries:
                return RecoverResult(
                    sample_idx=sample_idx,
                    ok=False,
                    reason=f"unexpected:{type(exc).__name__}:{str(exc)[:200]}",
                    output_path=str(out_path),
                )
            continue

    return RecoverResult(sample_idx=sample_idx, ok=False, reason="unreachable", output_path=str(out_path))


def recover_laion(
    input_manifest: Path,
    output_manifest: Path,
    output_image_dir: Path,
    failures_csv: Path,
    summary_json: Path,
    workers: int,
    timeout: float,
    retries: int,
    min_side: int,
    min_bytes: int,
    max_samples: Optional[int],
    overwrite: bool,
) -> None:
    df = pd.read_csv(input_manifest)
    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["modality"] = df["modality"].astype(str).str.strip().str.lower()
    laion = df[(df["dataset"] == "laion") & (df["modality"] == "image")].copy()
    laion = laion.reset_index(drop=True)
    if max_samples is not None and max_samples > 0:
        laion = laion.iloc[:max_samples].copy()

    if "source_url" not in laion.columns:
        raise ValueError("Input manifest missing required column: source_url")

    print(f"[info] laion rows={len(laion)}")
    output_image_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[int, str, Path]] = []
    kept_rows: List[Dict] = []
    failed_rows: List[Dict] = []

    # Pre-check existing files.
    for _, row in laion.iterrows():
        sample_idx = _safe_int(row.get("sample_idx"), default=-1)
        if sample_idx < 0:
            failed_rows.append(
                {
                    "sample_idx": sample_idx,
                    "reason": "invalid_sample_idx",
                    "source_url": str(row.get("source_url", "")),
                }
            )
            continue
        source_url = str(row.get("source_url", "")).strip()
        if not _is_http_url(source_url):
            failed_rows.append(
                {
                    "sample_idx": sample_idx,
                    "reason": "invalid_source_url",
                    "source_url": source_url,
                }
            )
            continue

        out_path = output_image_dir / f"laion_{sample_idx:08d}.jpg"
        if out_path.exists() and (not overwrite):
            ok, reason, w, h, b = _valid_local_image(out_path, min_side=min_side, min_bytes=min_bytes)
            if ok:
                out_row = row.to_dict()
                out_row["image_path"] = str(out_path)
                out_row["media_path"] = str(out_path)
                out_row["video_path"] = str(out_path)
                out_row["width"] = int(w)
                out_row["height"] = int(h)
                kept_rows.append(out_row)
                continue
            # invalid cached file -> redownload
        tasks.append((sample_idx, source_url, out_path))

    print(f"[info] cached_valid={len(kept_rows)} to_download={len(tasks)} pre_failed={len(failed_rows)}")

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        future_map = {
            ex.submit(
                _download_one,
                url=url,
                out_path=out_path,
                timeout=float(timeout),
                retries=int(retries),
                min_side=int(min_side),
                min_bytes=int(min_bytes),
            ): (sample_idx, url, out_path)
            for sample_idx, url, out_path in tasks
        }
        for fut in as_completed(future_map):
            sample_idx, source_url, out_path = future_map[fut]
            try:
                res = fut.result()
            except Exception as exc:  # pylint: disable=broad-except
                res = RecoverResult(
                    sample_idx=sample_idx,
                    ok=False,
                    reason=f"future_exception:{type(exc).__name__}:{str(exc)[:200]}",
                    output_path=str(out_path),
                )

            done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"[progress] downloaded={done}/{len(tasks)} ok_so_far={len(kept_rows)} failed_so_far={len(failed_rows)}")

            if res.ok:
                # Recover original row by sample_idx.
                row = laion[laion["sample_idx"].astype(int) == int(sample_idx)]
                if len(row) == 0:
                    failed_rows.append(
                        {
                            "sample_idx": sample_idx,
                            "reason": "row_not_found_after_download",
                            "source_url": source_url,
                            "output_path": str(out_path),
                        }
                    )
                    continue
                row = row.iloc[0].to_dict()
                row["image_path"] = str(out_path)
                row["media_path"] = str(out_path)
                row["video_path"] = str(out_path)
                row["width"] = int(res.width)
                row["height"] = int(res.height)
                kept_rows.append(row)
            else:
                failed_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "reason": res.reason,
                        "source_url": source_url,
                        "output_path": str(out_path),
                    }
                )

    out_df = pd.DataFrame(kept_rows)
    if len(out_df) > 0:
        out_df = out_df.sort_values("sample_idx").reset_index(drop=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_manifest, index=False)

    fail_df = pd.DataFrame(failed_rows)
    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    fail_df.to_csv(failures_csv, index=False)

    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "output_image_dir": str(output_image_dir),
        "workers": int(workers),
        "timeout": float(timeout),
        "retries": int(retries),
        "min_side": int(min_side),
        "min_bytes": int(min_bytes),
        "max_samples": int(max_samples) if max_samples is not None else None,
        "overwrite": bool(overwrite),
        "kept_rows": int(len(out_df)),
        "failed_rows": int(len(fail_df)),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] output_manifest={output_manifest}")
    print(f"[done] kept_rows={len(out_df)} failed_rows={len(fail_df)}")
    print(f"[done] failures_csv={failures_csv}")
    print(f"[done] summary_json={summary_json}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        default="data/laion_coyo/manifests/laion_coyo_selected_media_existing_58k.csv",
    )
    parser.add_argument(
        "--output-manifest",
        default="data/laion_coyo/manifests/laion_recovered_unique_by_sampleidx.csv",
    )
    parser.add_argument(
        "--output-image-dir",
        default="data/laion_coyo/raw/media/images_laion_recovered",
    )
    parser.add_argument(
        "--failures-csv",
        default="data/laion_coyo/manifests/laion_recovered_unique_failures.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="data/laion_coyo/manifests/laion_recovered_unique_summary.json",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-side", type=int, default=64)
    parser.add_argument("--min-bytes", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    recover_laion(
        input_manifest=Path(args.input_manifest),
        output_manifest=Path(args.output_manifest),
        output_image_dir=Path(args.output_image_dir),
        failures_csv=Path(args.failures_csv),
        summary_json=Path(args.summary_json),
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        min_side=args.min_side,
        min_bytes=args.min_bytes,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

