#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch


def _extract_shape(obj: Any) -> Optional[Tuple[int, ...]]:
    if torch.is_tensor(obj):
        return tuple(int(x) for x in obj.shape)
    if hasattr(obj, "shape"):
        try:
            return tuple(int(x) for x in obj.shape)
        except Exception:
            pass
    if isinstance(obj, dict):
        for key in ("latent_feature", "latents", "latent", "z", "vae_latents", "video_latents", "x"):
            if key in obj:
                s = _extract_shape(obj[key])
                if s is not None:
                    return s
        for v in obj.values():
            s = _extract_shape(v)
            if s is not None:
                return s
    if isinstance(obj, (list, tuple)) and obj:
        return _extract_shape(obj[0])
    return None


def _infer_t(shape: Tuple[int, ...]) -> Optional[int]:
    if len(shape) == 5:
        # likely B,C,T,H,W
        return int(shape[2])
    if len(shape) == 4:
        # likely C,T,H,W
        return int(shape[1])
    return None


def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan latent T distribution from manifest preprocessed pickles.")
    parser.add_argument("--csv-path", required=True, help="Path to joint manifest csv")
    parser.add_argument("--dataset", default="", help="Optional dataset filter, comma-separated (e.g. msrvtt,openvid_1m_existing)")
    parser.add_argument("--modality", default="video", help="Optional modality filter (video/image/all)")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means full scan after filtering")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--report-json", default="", help="Optional output json report path")
    args = parser.parse_args()

    rows = _read_rows(args.csv_path)
    dataset_filter = {x.strip().lower() for x in args.dataset.split(",") if x.strip()}
    modality_filter = args.modality.strip().lower()

    filtered: List[Dict[str, str]] = []
    for r in rows:
        ds = str(r.get("dataset", "")).strip().lower()
        md = str(r.get("modality", "")).strip().lower()
        if dataset_filter and ds not in dataset_filter:
            continue
        if modality_filter != "all" and md != modality_filter:
            continue
        filtered.append(r)

    rng = random.Random(args.seed)
    if args.max_rows and args.max_rows > 0 and len(filtered) > args.max_rows:
        filtered = rng.sample(filtered, args.max_rows)

    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "rows": 0,
        "exists": 0,
        "loaded": 0,
        "errors": 0,
        "missing_path": 0,
        "shape_counter": Counter(),
        "t_counter": Counter(),
    })

    for r in filtered:
        ds = str(r.get("dataset", "")).strip().lower()
        md = str(r.get("modality", "")).strip().lower()
        key = f"{ds}|{md}"
        st = stats[key]
        st["rows"] += 1

        p = str(r.get("preprocessed_path", "")).strip()
        if not p:
            st["missing_path"] += 1
            continue
        if not os.path.exists(p):
            st["missing_path"] += 1
            continue
        st["exists"] += 1
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            s = _extract_shape(obj)
            if s is None:
                st["errors"] += 1
                continue
            st["loaded"] += 1
            st["shape_counter"][str(s)] += 1
            t = _infer_t(s)
            if t is not None:
                st["t_counter"][int(t)] += 1
        except Exception:
            st["errors"] += 1

    report = {
        "csv_path": args.csv_path,
        "filters": {
            "dataset": sorted(dataset_filter),
            "modality": modality_filter,
            "max_rows": args.max_rows,
        },
        "groups": {},
    }

    for key in sorted(stats.keys()):
        st = stats[key]
        shape_counter: Counter = st["shape_counter"]
        t_counter: Counter = st["t_counter"]
        report["groups"][key] = {
            "rows": int(st["rows"]),
            "exists": int(st["exists"]),
            "loaded": int(st["loaded"]),
            "errors": int(st["errors"]),
            "missing_path": int(st["missing_path"]),
            "top_shapes": shape_counter.most_common(5),
            "t_counter": dict(sorted(t_counter.items(), key=lambda x: x[0])),
        }

    print(json.dumps(report, indent=2))
    if args.report_json:
        os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

