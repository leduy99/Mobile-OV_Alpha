#!/usr/bin/env python3
"""
MSR-VTT data pipeline for MobileOV/SANA training.

Pipeline stages:
1) download raw assets
2) extract zip assets
3) build OpenVid-style CSV: columns [video, caption]
4) encode videos with WAN VAE to *_features.pkl

Supports distributed encoding via torchrun.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from PIL import Image

# Ensure repo root import path for nets.third_party.wan.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOG = logging.getLogger("msrvtt_data_prepare")

WAN_TASK_TO_VAE_CKPT = {
    "t2v-1.3B": "Wan2.1_VAE.pth",
    "t2i-1.3B": "Wan2.1_VAE.pth",
    "t2v-14B": "Wan2.1_VAE.pth",
    "i2v-14B": "Wan2.1_VAE.pth",
    "t2i-14B": "Wan2.1_VAE.pth",
}


@dataclass
class Paths:
    root: Path
    raw: Path
    videos: Path
    metadata: Path
    manifests: Path
    preprocessed: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MSR-VTT downloader + WAN VAE encoder")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("download", "build-csv", "encode", "all"):
        p = sub.add_parser(name)
        _add_common_args(p)

    return parser.parse_args()


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root-dir", type=Path, default=Path("data/msrvtt"))
    p.add_argument("--repo-id", type=str, default="AlexZigma/msr-vtt")
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--video-zip", type=str, default="test_videos.zip")
    p.add_argument("--meta-zip", type=str, default="test_videodatainfo.json.zip")
    p.add_argument("--train-parquet", type=str, default="train.parquet")
    p.add_argument("--val-parquet", type=str, default="val.parquet")
    p.add_argument("--manifest-name", type=str, default="OpenVid_extracted_subset_unique.csv")
    p.add_argument("--captions-per-video", type=int, default=1)
    p.add_argument("--caption-policy", type=str, default="longest", choices=["first", "longest", "random"])
    p.add_argument("--max-videos", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--extract", action="store_true", default=True)

    p.add_argument("--ckpt-dir", type=Path, default=Path("omni_ckpts/wan/wanxiang1_3b"))
    p.add_argument("--task", type=str, default="t2v-1.3B")
    p.add_argument("--frame-num", type=int, default=81)
    p.add_argument("--sampling-rate", type=int, default=1)
    p.add_argument("--skip-num", type=int, default=0)
    p.add_argument("--target-size", type=str, default="480,832")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--allow-pad-short",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, pad short videos by repeating the last sampled frame to reach frame-num.",
    )


def ensure_paths(root: Path) -> Paths:
    raw = root / "raw"
    videos = root / "videos"
    metadata = root / "metadata"
    manifests = root
    preprocessed = root / "preprocessed"
    for d in (raw, videos, metadata, manifests, preprocessed):
        d.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, raw=raw, videos=videos, metadata=metadata, manifests=manifests, preprocessed=preprocessed)


def _hf_download(repo_id: str, filename: str, out_dir: Path, token: Optional[str]) -> Path:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(out_dir),
            token=token,
            resume_download=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download '{filename}' from dataset '{repo_id}'. "
            "Pass correct --repo-id/--filename or authenticate via huggingface-cli login."
        ) from exc

    local_path = Path(local)
    target = out_dir / Path(filename).name
    if local_path.resolve() != target.resolve():
        shutil.copy2(local_path, target)
    return target


def _resolve_remote_file(repo_id: str, requested: str, token: Optional[str], kind: str) -> str:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    if requested in files:
        return requested

    req_base = Path(requested).name
    base_matches = [f for f in files if Path(f).name == req_base]
    if len(base_matches) == 1:
        return base_matches[0]

    if kind == "video_zip":
        cands = [f for f in files if f.endswith(".zip") and "video" in f.lower() and "test" in f.lower()]
    elif kind == "meta_zip":
        cands = [f for f in files if f.endswith(".zip") and "videodatainfo" in f.lower()]
    elif kind == "train_parquet":
        cands = [f for f in files if f.endswith(".parquet") and "/train" in f.lower()]
    elif kind == "val_parquet":
        cands = [f for f in files if f.endswith(".parquet") and "/val" in f.lower()]
    else:
        cands = []

    if cands:
        cands = sorted(cands)
        return cands[0]

    raise FileNotFoundError(
        f"Could not resolve remote file for kind={kind} requested={requested} in dataset {repo_id}. "
        f"Available files={len(files)}"
    )


def download_assets(args: argparse.Namespace, paths: Paths) -> Dict[str, Path]:
    LOG.info("Downloading MSR-VTT assets from HF dataset: %s", args.repo_id)
    def _resolve_or_keep(requested: str, kind: str) -> str:
        # Fast path: when caller passes explicit nested path (e.g. data/test_videos.zip),
        # try direct download first to avoid HF tree/list API (can be rate-limited).
        if "/" in requested:
            return requested
        try:
            return _resolve_remote_file(args.repo_id, requested, args.hf_token, kind=kind)
        except Exception:
            return requested

    video_zip = _resolve_or_keep(args.video_zip, "video_zip")
    meta_zip = _resolve_or_keep(args.meta_zip, "meta_zip")
    train_parquet = _resolve_or_keep(args.train_parquet, "train_parquet")
    val_parquet = _resolve_or_keep(args.val_parquet, "val_parquet")

    LOG.info(
        "Resolved remote files: video_zip=%s meta_zip=%s train_parquet=%s val_parquet=%s",
        video_zip,
        meta_zip,
        train_parquet,
        val_parquet,
    )
    files = {
        "video_zip": _hf_download(args.repo_id, video_zip, paths.raw, args.hf_token),
        "meta_zip": _hf_download(args.repo_id, meta_zip, paths.raw, args.hf_token),
        "train_parquet": _hf_download(args.repo_id, train_parquet, paths.raw, args.hf_token),
        "val_parquet": _hf_download(args.repo_id, val_parquet, paths.raw, args.hf_token),
    }
    for k, v in files.items():
        LOG.info("Downloaded %s -> %s", k, v)
    return files


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def extract_assets(files: Dict[str, Path], paths: Paths) -> Tuple[Path, Path]:
    LOG.info("Extracting zip assets...")
    _extract_zip(files["video_zip"], paths.videos)
    _extract_zip(files["meta_zip"], paths.metadata)

    # Normalize expected videos layout to data/msrvtt/videos/TestVideo
    test_video_dir = paths.videos / "TestVideo"
    if not test_video_dir.exists():
        mp4s = list(paths.videos.rglob("*.mp4"))
        if mp4s:
            test_video_dir.mkdir(parents=True, exist_ok=True)
            for src in mp4s:
                dst = test_video_dir / src.name
                if not dst.exists():
                    shutil.move(str(src), str(dst))

    # Locate json metadata file
    json_files = sorted(paths.metadata.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No metadata json found under {paths.metadata}")
    meta_json = json_files[0]
    if not test_video_dir.exists():
        raise FileNotFoundError(f"Expected extracted videos in {test_video_dir}")

    LOG.info("Extract done: videos=%s metadata=%s", test_video_dir, meta_json)
    return test_video_dir, meta_json


def _select_caption(captions: List[str], policy: str, rng: np.random.Generator) -> str:
    captions = [str(c).strip() for c in captions if str(c).strip()]
    if not captions:
        return ""
    if policy == "first":
        return captions[0]
    if policy == "longest":
        return max(captions, key=len)
    idx = int(rng.integers(0, len(captions)))
    return captions[idx]


def build_openvid_style_csv(
    args: argparse.Namespace,
    paths: Paths,
    test_video_dir: Optional[Path] = None,
    meta_json: Optional[Path] = None,
) -> Path:
    if test_video_dir is None:
        test_video_dir = paths.videos

    json_files = sorted(paths.metadata.rglob("*.json"))
    if not json_files:
        json_files = sorted(paths.raw.rglob("*.json"))
    if not json_files:
        json_files = sorted(paths.root.rglob("*.json"))
    if meta_json is not None:
        json_files = [meta_json]
    if not json_files:
        raise FileNotFoundError(f"No metadata json found under {paths.metadata} or {paths.raw}")

    if not test_video_dir.exists():
        raise FileNotFoundError(f"Video dir not found: {test_video_dir}")

    by_video: Dict[str, List[str]] = {}
    used_meta_files: List[str] = []
    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
            sentences = data.get("sentences", [])
            if not sentences:
                continue
            used_meta_files.append(str(jf))
            for row in sentences:
                vid = str(row.get("video_id", "")).strip()
                cap = str(row.get("caption", "")).strip()
                if not vid or not cap:
                    continue
                by_video.setdefault(vid, []).append(cap)
        except Exception:
            continue
    if not by_video:
        raise RuntimeError("No valid 'sentences' found in any metadata json file")

    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, str]] = []
    video_files = sorted(test_video_dir.rglob("*.mp4"))
    for vp in video_files:
        video_id = vp.stem
        caps = by_video.get(video_id, [])
        if not caps:
            continue
        if args.captions_per_video <= 1:
            rows.append({"video": vp.name, "caption": _select_caption(caps, args.caption_policy, rng)})
        else:
            unique_caps = list(dict.fromkeys([c.strip() for c in caps if c.strip()]))
            if args.caption_policy == "longest":
                unique_caps = sorted(unique_caps, key=len, reverse=True)
            elif args.caption_policy == "random":
                rng.shuffle(unique_caps)
            for cap in unique_caps[: args.captions_per_video]:
                rows.append({"video": vp.name, "caption": cap})

        if args.max_videos > 0 and len(rows) >= args.max_videos * max(1, args.captions_per_video):
            break

    if not rows:
        raise RuntimeError("No rows selected for CSV. Check metadata and extracted videos.")

    out_csv = paths.manifests / args.manifest_name
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    LOG.info("Built CSV: %s rows=%d (metadata_files=%d)", out_csv, len(rows), len(used_meta_files))
    return out_csv


def _import_wan_vae():
    try:
        from nets.third_party.wan.modules.vae import WanVAE  # type: ignore
        return WanVAE
    except Exception:
        # Fallback to lightweight bundled VAE copy used by download_data toolkit.
        from download_data.openvid_dataops.third_party.wan.modules.vae import WanVAE  # type: ignore
        return WanVAE


def _frame_count(cap: cv2.VideoCapture) -> int:
    c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if c > 0:
        return c
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cnt = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        cnt += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    return cnt


def _transform_frames(frames: List[np.ndarray], target_size: Tuple[int, int]) -> torch.Tensor:
    h, w = frames[0].shape[:2]
    ratio = float(target_size[1]) / float(target_size[0])
    if w < h * ratio:
        crop = (int(float(w) / ratio), w)
    else:
        crop = (h, int(float(h) * ratio))

    tfm = transforms.Compose(
        [
            transforms.CenterCrop(crop),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return torch.stack([tfm(Image.fromarray(x)) for x in frames], dim=0)


def read_video_frames(
    video_path: str,
    frame_num: int,
    sampling_rate: int,
    skip_num: int,
    target_size: Tuple[int, int],
    allow_pad_short: bool = True,
) -> Optional[torch.Tensor]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = _frame_count(cap)
    if total_frames <= skip_num:
        cap.release()
        return None

    sampling_rate = max(1, int(sampling_rate))
    frames = []
    cur = 0
    while cur < total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if cur >= skip_num and ((cur - skip_num) % sampling_rate == 0):
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cur += 1
    cap.release()

    if len(frames) == 0:
        return None

    if len(frames) >= frame_num:
        # Uniformly subsample to frame_num to keep temporal coverage.
        pick = np.linspace(0, len(frames) - 1, num=frame_num, dtype=np.int64)
        sampled = [frames[int(i)] for i in pick]
    else:
        if not allow_pad_short:
            return None
        sampled = list(frames)
        sampled.extend([sampled[-1]] * (frame_num - len(sampled)))

    return _transform_frames(sampled, target_size)


def _build_video_index(video_root: Path) -> Dict[str, Path]:
    """
    Build a filename -> full path index for all mp4 files under video_root.
    Supports CSV rows that contain either 'video123.mp4' or 'video123'.
    """
    index: Dict[str, Path] = {}
    video_files = sorted(video_root.rglob("*.mp4"))
    for vp in video_files:
        name = vp.name
        stem = vp.stem
        # Keep first occurrence deterministically.
        if name not in index:
            index[name] = vp
        if stem not in index:
            index[stem] = vp
    return index


def _setup_dist() -> Tuple[int, int, int, torch.device]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def encode_csv(args: argparse.Namespace, paths: Paths, csv_path: Optional[Path] = None) -> None:
    if csv_path is None:
        csv_path = paths.manifests / args.manifest_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if args.task not in WAN_TASK_TO_VAE_CKPT:
        raise ValueError(f"Unknown task={args.task}. choices={list(WAN_TASK_TO_VAE_CKPT.keys())}")

    rank, world_size, _, device = _setup_dist()
    target_size = tuple(int(x) for x in args.target_size.split(","))

    vae_ckpt = args.ckpt_dir / WAN_TASK_TO_VAE_CKPT[args.task]
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"WAN VAE checkpoint not found: {vae_ckpt}")

    WanVAE = _import_wan_vae()
    vae = WanVAE(vae_pth=str(vae_ckpt), device=device)

    df = pd.read_csv(csv_path)
    if args.max_samples > 0:
        df = df.iloc[: args.max_samples]

    video_index = _build_video_index(paths.videos)
    if rank == 0:
        LOG.info("Indexed %d video keys from %s", len(video_index), paths.videos)

    done = 0
    skipped = 0
    failed = 0
    failures = []

    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx % world_size != rank:
                continue
            video_name = str(row["video"])
            prompt = str(row.get("caption", "")).strip()
            vp = video_index.get(video_name) or video_index.get(Path(video_name).stem)
            out_name = f"{Path(video_name).stem}_features.pkl"
            out_path = paths.preprocessed / out_name

            if out_path.exists():
                skipped += 1
                continue
            if vp is None or (not vp.exists()):
                failed += 1
                failures.append({"video": video_name, "reason": "missing_video"})
                continue

            frames = read_video_frames(
                str(vp),
                frame_num=args.frame_num,
                sampling_rate=args.sampling_rate,
                skip_num=args.skip_num,
                target_size=target_size,
                allow_pad_short=bool(args.allow_pad_short),
            )
            if frames is None:
                failed += 1
                failures.append({"video": video_name, "reason": "invalid_frames"})
                continue

            try:
                frames = frames.to(device)
                latent = vae.encode(frames.transpose(0, 1).unsqueeze(0))[0].cpu()
                item = {
                    "latent_feature": latent,
                    "prompt": prompt,
                    "video_path": str(vp),
                    "frame_num": int(args.frame_num),
                }
                with out_path.open("wb") as f:
                    pickle.dump(item, f)
                done += 1
            except Exception as exc:  # pylint: disable=broad-except
                failed += 1
                failures.append({"video": video_name, "reason": str(exc)})

            if done > 0 and done % max(1, args.log_every) == 0:
                LOG.info("rank=%d progress done=%d skipped=%d failed=%d", rank, done, skipped, failed)

    summary = {
        "rank": rank,
        "world_size": world_size,
        "csv_path": str(csv_path),
        "preprocessed_dir": str(paths.preprocessed),
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "frame_num": int(args.frame_num),
        "sampling_rate": int(args.sampling_rate),
        "target_size": list(target_size),
        "allow_pad_short": bool(args.allow_pad_short),
    }
    summary_path = paths.preprocessed / f"encode_summary_rank{rank:02d}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        pd.DataFrame(failures).to_csv(paths.preprocessed / f"encode_failed_rank{rank:02d}.csv", index=False)

    LOG.info("rank=%d finished done=%d skipped=%d failed=%d", rank, done, skipped, failed)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        if rank == 0:
            LOG.info("All ranks finished encoding")
        dist.destroy_process_group()


def run_all(args: argparse.Namespace, paths: Paths) -> None:
    files = download_assets(args, paths)
    test_video_dir, meta_json = extract_assets(files, paths)
    csv_path = build_openvid_style_csv(args, paths, test_video_dir=test_video_dir, meta_json=meta_json)
    encode_csv(args, paths, csv_path=csv_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    paths = ensure_paths(args.root_dir)

    if args.command == "download":
        files = download_assets(args, paths)
        if args.extract:
            extract_assets(files, paths)
        return
    if args.command == "build-csv":
        build_openvid_style_csv(args, paths)
        return
    if args.command == "encode":
        encode_csv(args, paths)
        return
    if args.command == "all":
        run_all(args, paths)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
