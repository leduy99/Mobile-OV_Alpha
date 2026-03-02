#!/usr/bin/env python3
"""
Preflight checks for joint image+video training pipeline.

Checks performed:
1) Manifest schema/integrity (required columns, modality counts, duplicates).
2) File-level checks (preprocessed pickle exists, tiny image files).
3) Latent-level checks on sampled rows (shape, finite values, modality consistency).
4) Dataloader dry-run (video/image dataloaders + collate behavior).

Exit code:
  0 -> pass
  2 -> failed preflight (fatal issues found)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pickle
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets.omni.datasets.openvid_dataset import OpenVidDataset, openvid_collate_fn  # noqa: E402


REQUIRED_COLUMNS = {
    "video",
    "caption",
    "preprocessed_path",
    "video_path",
    "dataset",
    "modality",
    "sample_idx",
}


@dataclass
class CheckIssue:
    level: str  # "fatal" | "warn" | "info"
    code: str
    message: str
    value: Optional[float] = None


def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _safe_get_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return -1


def _canonicalize_manifest(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Normalize minimal manifests (e.g. video,caption) into preflight schema.
    """
    dfx = df.copy()
    data_cfg = cfg.get("data", {}) or {}
    openvid_cfg = data_cfg.get("openvid", {}) or {}
    video_dir = str(openvid_cfg.get("video_dir", ".") or ".")
    pre_dir = openvid_cfg.get("preprocessed_dir", None)
    pre_dir = str(pre_dir) if pre_dir else None

    if "video_path" not in dfx.columns:
        if "video" in dfx.columns:
            dfx["video_path"] = dfx["video"].astype(str)
        else:
            dfx["video_path"] = ""
    else:
        dfx["video_path"] = dfx["video_path"].fillna("").astype(str)
        if "video" in dfx.columns:
            missing_video_path = dfx["video_path"].str.len() == 0
            if missing_video_path.any():
                dfx.loc[missing_video_path, "video_path"] = dfx.loc[missing_video_path, "video"].astype(str)

    if "preprocessed_path" not in dfx.columns:
        if pre_dir and "video_path" in dfx.columns:
            def _to_pkl(v: str) -> str:
                base = os.path.basename(str(v))
                stem = os.path.splitext(base)[0]
                return os.path.join(pre_dir, f"{stem}_features.pkl")

            dfx["preprocessed_path"] = dfx["video_path"].map(_to_pkl)
        else:
            dfx["preprocessed_path"] = ""
    else:
        dfx["preprocessed_path"] = dfx["preprocessed_path"].fillna("").astype(str)

    if "dataset" not in dfx.columns:
        dfx["dataset"] = "unknown"
    else:
        dfx["dataset"] = dfx["dataset"].fillna("unknown").astype(str)

    if "modality" not in dfx.columns:
        dfx["modality"] = "video"
    else:
        dfx["modality"] = dfx["modality"].fillna("video").astype(str)

    if "sample_idx" not in dfx.columns:
        dfx["sample_idx"] = dfx.index.astype(int)

    if "caption" not in dfx.columns:
        dfx["caption"] = ""
    else:
        dfx["caption"] = dfx["caption"].fillna("").astype(str)

    if "video" not in dfx.columns:
        dfx["video"] = dfx["video_path"].map(lambda p: os.path.basename(str(p)))

    # Keep paths non-empty for downstream checks.
    if "video_path" in dfx.columns and "video" in dfx.columns:
        empty_path = dfx["video_path"].str.len() == 0
        if empty_path.any():
            dfx.loc[empty_path, "video_path"] = dfx.loc[empty_path, "video"].astype(str)

    return dfx


def _check_manifest(
    df: pd.DataFrame,
    tiny_image_bytes: int,
    *,
    require_image: bool,
) -> Tuple[List[CheckIssue], Dict]:
    issues: List[CheckIssue] = []
    summary: Dict = {}

    required_cols = set(REQUIRED_COLUMNS)
    if not require_image and "modality" in required_cols:
        # Backward-compatible path for video-only manifests.
        required_cols.remove("modality")
    missing_cols = sorted(list(required_cols - set(df.columns)))
    if missing_cols:
        issues.append(CheckIssue("fatal", "missing_columns", f"Manifest missing columns: {missing_cols}"))
        return issues, summary

    dfx = df.copy()
    if "modality" not in dfx.columns:
        dfx["modality"] = "video"
        issues.append(
            CheckIssue(
                "warn",
                "missing_modality_default_video",
                "Manifest has no modality column; defaulting all rows to video for preflight",
            )
        )
    dfx["modality"] = dfx["modality"].astype(str).str.strip().str.lower()
    dfx["dataset"] = dfx["dataset"].astype(str).str.strip().str.lower()
    dfx["caption"] = dfx["caption"].fillna("").astype(str)
    dfx["video_path"] = dfx["video_path"].fillna("").astype(str)
    dfx["preprocessed_path"] = dfx["preprocessed_path"].fillna("").astype(str)

    modality_counts = dfx["modality"].value_counts(dropna=False).to_dict()
    dataset_counts = dfx["dataset"].value_counts(dropna=False).to_dict()
    summary["rows_total"] = int(len(dfx))
    summary["modality_counts"] = modality_counts
    summary["dataset_counts"] = dataset_counts

    if int(modality_counts.get("video", 0)) <= 0:
        issues.append(CheckIssue("fatal", "no_video_rows", "No video rows found in manifest"))
    if require_image and int(modality_counts.get("image", 0)) <= 0:
        issues.append(CheckIssue("fatal", "no_image_rows", "No image rows found in manifest"))

    # Missing preprocessed pkl.
    pre_exists = dfx["preprocessed_path"].map(lambda p: bool(p) and os.path.exists(p))
    missing_pre = int((~pre_exists).sum())
    summary["missing_preprocessed"] = missing_pre
    if missing_pre > 0:
        issues.append(CheckIssue("fatal", "missing_preprocessed", "Missing preprocessed pickle files", missing_pre))

    # Image path duplication and caption conflicts.
    img = dfx[dfx["modality"] == "image"].copy()
    if len(img) > 0:
        dup_image_path = int(img["video_path"].duplicated().sum())
        summary["image_duplicate_video_path"] = dup_image_path
        if dup_image_path > 0:
            issues.append(
                CheckIssue(
                    "fatal",
                    "image_path_duplicates",
                    "Image modality has duplicated video_path entries",
                    dup_image_path,
                )
            )
        caps_per_path = img.groupby("video_path")["caption"].nunique()
        conflict_paths = int((caps_per_path > 1).sum())
        summary["image_conflicting_caption_paths"] = conflict_paths
        if conflict_paths > 0:
            issues.append(
                CheckIssue(
                    "fatal",
                    "image_caption_conflicts",
                    "Same image path mapped to multiple captions",
                    conflict_paths,
                )
            )

        sizes = img["video_path"].map(_safe_get_size)
        tiny_count = int((sizes >= 0).sum() and (sizes < int(tiny_image_bytes)).sum())
        tiny_ratio = float(tiny_count) / max(1, len(img))
        summary["image_tiny_files_count"] = tiny_count
        summary["image_tiny_files_ratio"] = tiny_ratio
        if tiny_count > 0:
            level = "fatal" if tiny_ratio > 0.002 else "warn"
            issues.append(
                CheckIssue(
                    level,
                    "tiny_image_files",
                    f"Image files smaller than {tiny_image_bytes} bytes detected",
                    tiny_count,
                )
            )

    # sample_idx collisions are not always fatal, but useful to monitor.
    dup_sample_idx = int(dfx["sample_idx"].duplicated().sum())
    summary["duplicate_sample_idx"] = dup_sample_idx
    if dup_sample_idx > 0:
        issues.append(
            CheckIssue(
                "warn",
                "duplicate_sample_idx",
                "sample_idx is duplicated across rows (can affect teacher-cache indexing if enabled)",
                dup_sample_idx,
            )
        )

    return issues, summary


def _check_latents(
    df: pd.DataFrame,
    sample_video: int,
    sample_image: int,
    seed: int,
    *,
    require_image: bool,
) -> Tuple[List[CheckIssue], Dict]:
    issues: List[CheckIssue] = []
    summary: Dict = {}
    rng = random.Random(seed)

    dfx = df.copy()
    if "modality" not in dfx.columns:
        dfx["modality"] = "video"
    dfx["modality"] = dfx["modality"].astype(str).str.strip().str.lower()

    checks = [("video", sample_video)]
    if require_image:
        checks.append(("image", sample_image))
    for modality, n in checks:
        sub = dfx[dfx["modality"] == modality]
        idxs = list(sub.index)
        if not idxs:
            continue
        pick = rng.sample(idxs, min(n, len(idxs)))
        bad_shape = 0
        bad_t = 0
        bad_finite = 0
        bad_std = 0
        opened = 0
        t_values: List[int] = []
        std_values: List[float] = []
        for i in pick:
            pkl_path = str(sub.loc[i, "preprocessed_path"])
            try:
                with open(pkl_path, "rb") as f:
                    item = pickle.load(f)
                latent = item.get("latent_feature", None)
                if latent is None:
                    bad_shape += 1
                    continue
                if not torch.is_tensor(latent):
                    latent = torch.tensor(latent)
                opened += 1
                if latent.dim() != 4:
                    bad_shape += 1
                    continue
                t = int(latent.shape[1])
                t_values.append(t)
                stdv = float(latent.float().std().item())
                std_values.append(stdv)
                if modality == "video" and t < 2:
                    bad_t += 1
                if modality == "image" and t != 1:
                    bad_t += 1
                if not torch.isfinite(latent).all():
                    bad_finite += 1
                if stdv <= 1e-6:
                    bad_std += 1
            except Exception:
                bad_shape += 1

        key = f"{modality}_latent"
        summary[key] = {
            "sampled": len(pick),
            "opened": opened,
            "bad_shape": bad_shape,
            "bad_temporal": bad_t,
            "bad_finite": bad_finite,
            "bad_zero_std": bad_std,
            "t_min": min(t_values) if t_values else None,
            "t_max": max(t_values) if t_values else None,
            "std_median": float(pd.Series(std_values).median()) if std_values else None,
        }
        if bad_shape > 0:
            issues.append(CheckIssue("fatal", f"{modality}_bad_shape", f"{modality} latent bad shape/count", bad_shape))
        if bad_t > 0:
            issues.append(CheckIssue("fatal", f"{modality}_bad_temporal", f"{modality} latent temporal mismatch", bad_t))
        if bad_finite > 0:
            issues.append(CheckIssue("fatal", f"{modality}_non_finite", f"{modality} latent has NaN/Inf", bad_finite))
        if bad_std > 0:
            issues.append(CheckIssue("warn", f"{modality}_zero_std", f"{modality} latent near-zero std", bad_std))

    return issues, summary


def _check_dataloaders(
    cfg: Dict,
    csv_path: str,
    max_batches: int,
    *,
    require_image: bool,
) -> Tuple[List[CheckIssue], Dict]:
    issues: List[CheckIssue] = []
    summary: Dict = {}

    openvid_cfg = cfg.get("data", {}).get("openvid", {})
    batching_cfg = cfg.get("data", {}).get("batching", {})
    joint_cfg = cfg.get("data", {}).get("joint", {})

    batch_size_video = int(batching_cfg.get("batch_size", 1))
    batch_size_image = int(batching_cfg.get("batch_size_image", batch_size_video))
    video_modality = str(joint_cfg.get("video_modality", "video")).strip().lower()
    image_modality = str(joint_cfg.get("image_modality", "image")).strip().lower()

    video_ds = OpenVidDataset(
        csv_path=csv_path,
        video_dir=str(openvid_cfg.get("video_dir", ".")),
        preprocessed_dir=openvid_cfg.get("preprocessed_dir", None),
        use_preprocessed=bool(openvid_cfg.get("use_preprocessed", True)),
        max_samples=None,
        modality_filter=[video_modality],
    )
    summary["dataset_len_video"] = int(len(video_ds))
    if len(video_ds) == 0:
        issues.append(CheckIssue("fatal", "empty_video_dataset", "Video dataloader dataset is empty"))
    image_ds = None
    if require_image:
        image_ds = OpenVidDataset(
            csv_path=csv_path,
            video_dir=str(openvid_cfg.get("video_dir", ".")),
            preprocessed_dir=openvid_cfg.get("preprocessed_dir", None),
            use_preprocessed=bool(openvid_cfg.get("use_preprocessed", True)),
            max_samples=None,
            modality_filter=[image_modality],
        )
        summary["dataset_len_image"] = int(len(image_ds))
        if len(image_ds) == 0:
            issues.append(CheckIssue("fatal", "empty_image_dataset", "Image dataloader dataset is empty"))
    else:
        summary["dataset_len_image"] = 0

    video_loader = DataLoader(
        video_ds,
        batch_size=batch_size_video,
        shuffle=True,
        num_workers=0,
        collate_fn=openvid_collate_fn,
        drop_last=True,
    )
    def inspect_loader(name: str, loader: DataLoader, expect_image: bool) -> Dict:
        seen = 0
        bad_batch = 0
        temporal = []
        for batch in loader:
            lat = batch.get("latent_feature", None)
            if lat is None or (not torch.is_tensor(lat)) or lat.dim() != 5:
                bad_batch += 1
            else:
                t = int(lat.shape[2])
                temporal.append(t)
                if expect_image and t != 1:
                    bad_batch += 1
                if (not expect_image) and t < 2:
                    bad_batch += 1
            seen += 1
            if seen >= max_batches:
                break
        return {
            "seen_batches": seen,
            "bad_batches": bad_batch,
            "t_min": min(temporal) if temporal else None,
            "t_max": max(temporal) if temporal else None,
        }

    video_info = inspect_loader("video", video_loader, expect_image=False)
    if require_image and image_ds is not None:
        image_loader = DataLoader(
            image_ds,
            batch_size=batch_size_image,
            shuffle=True,
            num_workers=0,
            collate_fn=openvid_collate_fn,
            drop_last=True,
        )
        image_info = inspect_loader("image", image_loader, expect_image=True)
    else:
        image_info = {"seen_batches": 0, "bad_batches": 0, "t_min": None, "t_max": None}
    summary["loader_video"] = video_info
    summary["loader_image"] = image_info

    if video_info["bad_batches"] > 0:
        issues.append(CheckIssue("fatal", "video_loader_bad_batches", "Video loader produced invalid batches", video_info["bad_batches"]))
    if require_image and image_info["bad_batches"] > 0:
        issues.append(CheckIssue("fatal", "image_loader_bad_batches", "Image loader produced invalid batches", image_info["bad_batches"]))

    return issues, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_teacher_free_joint_msrvtt_laion_coyo_ivjoint_3gpu.yaml")
    parser.add_argument("--csv-path", default=None, help="Override csv path from config")
    parser.add_argument("--report-json", default="output/logs/preflight_joint_pipeline_report.json")
    parser.add_argument("--sample-video", type=int, default=400)
    parser.add_argument("--sample-image", type=int, default=1000)
    parser.add_argument("--max-loader-batches", type=int, default=20)
    parser.add_argument("--tiny-image-bytes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = _load_cfg(args.config)
    csv_path = args.csv_path or cfg.get("data", {}).get("openvid", {}).get("csv_path", "")
    joint_cfg = cfg.get("data", {}).get("joint", {}) or {}
    joint_enabled = bool(joint_cfg.get("enabled", False))
    image_per_video = joint_cfg.get("image_per_video", None)
    try:
        image_per_video = int(image_per_video) if image_per_video is not None else None
    except Exception:
        image_per_video = None
    joint_interval = int(joint_cfg.get("interval", 0) or 0)
    require_image = bool(joint_enabled and ((image_per_video is not None and image_per_video > 0) or joint_interval > 0))
    if not csv_path:
        print("[FATAL] csv_path is empty in config and --csv-path not provided")
        raise SystemExit(2)

    _print_header("Preflight: Manifest Load")
    df_raw = pd.read_csv(csv_path)
    df = _canonicalize_manifest(df_raw, cfg)
    print(f"csv_path={csv_path}")
    print(f"rows_raw={len(df_raw)}")
    print(f"rows_canonical={len(df)}")
    print(f"require_image={require_image}")
    if set(df.columns) != set(df_raw.columns):
        added = sorted(list(set(df.columns) - set(df_raw.columns)))
        print(f"canonicalize_added_columns={added}")

    issues: List[CheckIssue] = []
    report: Dict = {"config": args.config, "csv_path": csv_path}

    _print_header("Preflight: Manifest Integrity")
    manifest_issues, manifest_summary = _check_manifest(
        df,
        tiny_image_bytes=args.tiny_image_bytes,
        require_image=require_image,
    )
    issues.extend(manifest_issues)
    report["manifest"] = manifest_summary
    print(json.dumps(manifest_summary, indent=2))

    _print_header("Preflight: Latent Sanity (Sampled)")
    latent_issues, latent_summary = _check_latents(
        df,
        sample_video=int(args.sample_video),
        sample_image=int(args.sample_image),
        seed=int(args.seed),
        require_image=require_image,
    )
    issues.extend(latent_issues)
    report["latent"] = latent_summary
    print(json.dumps(latent_summary, indent=2))

    _print_header("Preflight: Dataloader Dry-Run")
    loader_issues, loader_summary = _check_dataloaders(
        cfg=cfg,
        csv_path=csv_path,
        max_batches=int(args.max_loader_batches),
        require_image=require_image,
    )
    issues.extend(loader_issues)
    report["loader"] = loader_summary
    print(json.dumps(loader_summary, indent=2))

    _print_header("Preflight: Issues")
    if not issues:
        print("No issues found.")
    else:
        for it in issues:
            print(f"[{it.level.upper()}] {it.code}: {it.message} value={it.value}")

    report["issues"] = [asdict(x) for x in issues]
    report_dir = os.path.dirname(args.report_json)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] report_json={args.report_json}")

    fatal_count = sum(1 for x in issues if x.level == "fatal")
    if fatal_count > 0:
        print(f"[result] FAILED preflight (fatal={fatal_count})")
        raise SystemExit(2)
    print("[result] PASSED preflight")


if __name__ == "__main__":
    main()
