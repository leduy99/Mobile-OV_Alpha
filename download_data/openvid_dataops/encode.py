import json
import logging
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from PIL import Image

LOGGER = logging.getLogger(__name__)

WAN_TASK_TO_VAE_CKPT = {
    "t2v-1.3B": "Wan2.1_VAE.pth",
    "t2i-1.3B": "Wan2.1_VAE.pth",
    "t2v-14B": "Wan2.1_VAE.pth",
    "i2v-14B": "Wan2.1_VAE.pth",
    "t2i-14B": "Wan2.1_VAE.pth",
}


def _import_wan_vae():
    from .third_party.wan.modules.vae import WanVAE  # pylint: disable=import-error
    return WanVAE


def _get_video_frame_count_fast(cap: cv2.VideoCapture) -> int:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        return frame_count
    # Fallback only when metadata is missing/corrupted.
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cnt = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        cnt += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    return cnt


def _transform_frames_to_tensor(frames, target_size=(480, 832)) -> torch.Tensor:
    first = frames[0]
    if isinstance(first, np.ndarray):
        h, w = first.shape[:2]
    elif isinstance(first, Image.Image):
        w, h = first.size
    else:
        first_arr = np.asarray(first)
        h, w = first_arr.shape[:2]

    ratio = float(target_size[1]) / float(target_size[0])
    if w < h * ratio:
        crop_size = (int(float(w) / ratio), w)
    else:
        crop_size = (h, int(float(h) * ratio))

    transform = transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    out = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        elif not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.asarray(frame))
        frame = frame.convert("RGB")
        out.append(transform(frame))
    return torch.stack(out)


def _build_temporal_sample_indices(
    total_frames: int,
    frame_num: int,
    sampling_rate: int,
    skip_num: int,
    source_fps: float,
    target_fps: float = 16.0,
) -> Optional[np.ndarray]:
    available_frames = int(total_frames) - int(skip_num)
    if frame_num <= 0 or available_frames <= 0:
        return None

    # Follow the SANA temporal contract: num_frames = N_seconds * 16 + 1.
    # For the common case frame_num=81 and sampling_rate=1, this maps to a 5s
    # clip sampled on a 16fps timeline.
    stride = max(1, int(sampling_rate))
    effective_target_fps = float(target_fps) / float(stride)
    if effective_target_fps <= 0:
        return None

    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = float(target_fps)

    target_span_sec = float(max(frame_num - 1, 0)) / effective_target_fps
    available_span_sec = float(max(available_frames - 1, 0)) / float(source_fps)
    sample_span_sec = min(target_span_sec, available_span_sec)

    timestamps = np.linspace(0.0, sample_span_sec, num=frame_num, dtype=np.float64)
    rel_indices = np.rint(timestamps * float(source_fps)).astype(np.int64)
    rel_indices = np.clip(rel_indices, 0, available_frames - 1)
    return rel_indices + int(skip_num)


def read_video_frames(
    video_path: str,
    frame_num: int,
    sampling_rate: int = 3,
    skip_num: int = 0,
    target_size: Tuple[int, int] = (480, 832),
) -> Optional[torch.Tensor]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = _get_video_frame_count_fast(cap)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= skip_num:
        cap.release()
        return None

    if (target_size[0] > target_size[1] and height < width) or (
        target_size[0] < target_size[1] and height > width
    ):
        cap.release()
        return None

    sample_indices = _build_temporal_sample_indices(
        total_frames=total_frames,
        frame_num=frame_num,
        sampling_rate=sampling_rate,
        skip_num=skip_num,
        source_fps=fps,
    )
    if sample_indices is None or len(sample_indices) == 0:
        cap.release()
        return None

    needed_counts: Dict[int, int] = {}
    for idx in sample_indices.tolist():
        needed_counts[int(idx)] = needed_counts.get(int(idx), 0) + 1

    frames = []
    current = 0
    while current < total_frames and needed_counts:
        ret, frame = cap.read()
        if not ret:
            break

        repeat = needed_counts.pop(current, 0)
        if repeat:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for _ in range(repeat):
                frames.append(frame_rgb.copy())
        current += 1
    cap.release()

    if len(frames) == 0:
        return None

    if len(frames) < frame_num:
        frames.extend([frames[-1].copy()] * (frame_num - len(frames)))
    elif len(frames) > frame_num:
        frames = frames[:frame_num]

    return _transform_frames_to_tensor(frames, target_size)


def _load_existing_fail_rows(fail_path: Path) -> List[Dict[str, object]]:
    if not fail_path.exists():
        return []
    try:
        df = pd.read_csv(fail_path)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Could not read existing failure CSV %s: %s", fail_path, exc)
        return []
    return df.to_dict(orient="records")


def _write_fail_csv(fail_rows: List[Dict[str, object]], fail_path: Path) -> None:
    if not fail_rows:
        return
    pd.DataFrame(fail_rows).to_csv(fail_path, index=False)


def _write_summary(
    summary_path: Path,
    *,
    rank: int,
    world_size: int,
    manifest_csv: Path,
    output_dir: Path,
    done: int,
    skipped: int,
    failed: int,
    fail_path: Path,
    has_fail_rows: bool,
    last_row_idx: Optional[int],
    last_sample_idx: Optional[int],
    status: str,
) -> None:
    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "rank": rank,
        "world_size": world_size,
        "manifest_csv": str(manifest_csv),
        "output_dir": str(output_dir),
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "failed_csv": str(fail_path) if has_fail_rows else None,
        "last_row_idx": last_row_idx,
        "last_sample_idx": last_sample_idx,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _atomic_pickle_dump(item: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=out_path.parent,
        prefix=f".{out_path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_f:
        tmp_path = Path(tmp_f.name)
        pickle.dump(item, tmp_f)
    tmp_path.replace(out_path)


def encode_manifest(
    ckpt_dir: Path,
    manifest_csv: Path,
    output_dir: Path,
    task: str,
    frame_num: int,
    sampling_rate: int,
    skip_num: int,
    target_size: Tuple[int, int],
    max_samples: Optional[int],
    log_every: int,
) -> None:
    if task not in WAN_TASK_TO_VAE_CKPT:
        raise ValueError(f"Unknown WAN task: {task}. Available: {list(WAN_TASK_TO_VAE_CKPT.keys())}")
    WanVAE = _import_wan_vae()

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

    vae_ckpt = ckpt_dir / WAN_TASK_TO_VAE_CKPT[task]
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"WAN VAE checkpoint not found: {vae_ckpt}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fail_path = output_dir / f"failed_rank{rank:02d}.csv"
    summary_path = output_dir / f"summary_rank{rank:02d}.json"

    LOGGER.info("Rank %d/%d loading manifest: %s", rank, world_size, manifest_csv)
    df = pd.read_csv(manifest_csv)
    if max_samples:
        df = df.iloc[:max_samples]

    LOGGER.info("Rank %d initializing WAN VAE: %s", rank, vae_ckpt)
    vae = WanVAE(vae_pth=str(vae_ckpt), device=device)

    done = 0
    skipped = 0
    failed = 0
    fail_rows = _load_existing_fail_rows(fail_path)
    last_row_idx: Optional[int] = None
    last_sample_idx: Optional[int] = None

    _write_summary(
        summary_path,
        rank=rank,
        world_size=world_size,
        manifest_csv=manifest_csv,
        output_dir=output_dir,
        done=done,
        skipped=skipped,
        failed=failed,
        fail_path=fail_path,
        has_fail_rows=bool(fail_rows),
        last_row_idx=last_row_idx,
        last_sample_idx=last_sample_idx,
        status="running",
    )

    with torch.no_grad():
        for row_idx, row in df.iterrows():
            if row_idx % world_size != rank:
                continue

            sample_idx = int(row.get("sample_idx", row_idx))
            last_row_idx = int(row_idx)
            last_sample_idx = int(sample_idx)
            video_path = Path(str(row["video_path"]))
            prompt = str(row.get("caption", ""))

            out_path = output_dir / f"sample_{sample_idx:08d}.pkl"
            if out_path.exists():
                skipped += 1
                processed = done + skipped + failed
                if processed % max(1, log_every) == 0:
                    LOGGER.info("Rank %d progress: done=%d skipped=%d failed=%d", rank, done, skipped, failed)
                    _write_summary(
                        summary_path,
                        rank=rank,
                        world_size=world_size,
                        manifest_csv=manifest_csv,
                        output_dir=output_dir,
                        done=done,
                        skipped=skipped,
                        failed=failed,
                        fail_path=fail_path,
                        has_fail_rows=bool(fail_rows),
                        last_row_idx=last_row_idx,
                        last_sample_idx=last_sample_idx,
                        status="running",
                    )
                continue

            if not video_path.exists():
                failed += 1
                fail_rows.append(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "sample_idx": sample_idx,
                        "video_path": str(video_path),
                        "reason": "missing_video",
                    }
                )
                _write_fail_csv(fail_rows, fail_path)
                _write_summary(
                    summary_path,
                    rank=rank,
                    world_size=world_size,
                    manifest_csv=manifest_csv,
                    output_dir=output_dir,
                    done=done,
                    skipped=skipped,
                    failed=failed,
                    fail_path=fail_path,
                    has_fail_rows=True,
                    last_row_idx=last_row_idx,
                    last_sample_idx=last_sample_idx,
                    status="running",
                )
                continue

            try:
                frames = read_video_frames(
                    str(video_path),
                    frame_num=frame_num,
                    sampling_rate=sampling_rate,
                    skip_num=skip_num,
                    target_size=target_size,
                )
                if frames is None:
                    raise RuntimeError("insufficient_or_invalid_frames")

                frames = frames.to(device)
                latent_feature = vae.encode(frames.transpose(0, 1).unsqueeze(0))[0].cpu()

                item = {
                    "sample_idx": sample_idx,
                    "video": str(row.get("video", video_path.name)),
                    "video_path": str(video_path),
                    "prompt": prompt,
                    "part_user": int(row.get("part_user", -1)),
                    "part_remote": int(row.get("part_remote", -1)),
                    "frame_num": frame_num,
                    "target_size": list(target_size),
                    "sampling_rate": sampling_rate,
                    "skip_num": skip_num,
                    "latent_feature": latent_feature,
                }
                _atomic_pickle_dump(item, out_path)
                done += 1

                processed = done + skipped + failed
                if processed % max(1, log_every) == 0:
                    LOGGER.info("Rank %d progress: done=%d skipped=%d failed=%d", rank, done, skipped, failed)
                    _write_summary(
                        summary_path,
                        rank=rank,
                        world_size=world_size,
                        manifest_csv=manifest_csv,
                        output_dir=output_dir,
                        done=done,
                        skipped=skipped,
                        failed=failed,
                        fail_path=fail_path,
                        has_fail_rows=bool(fail_rows),
                        last_row_idx=last_row_idx,
                        last_sample_idx=last_sample_idx,
                        status="running",
                    )

            except Exception as exc:  # pylint: disable=broad-except
                failed += 1
                fail_rows.append(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "sample_idx": sample_idx,
                        "video_path": str(video_path),
                        "reason": str(exc),
                    }
                )
                _write_fail_csv(fail_rows, fail_path)
                _write_summary(
                    summary_path,
                    rank=rank,
                    world_size=world_size,
                    manifest_csv=manifest_csv,
                    output_dir=output_dir,
                    done=done,
                    skipped=skipped,
                    failed=failed,
                    fail_path=fail_path,
                    has_fail_rows=True,
                    last_row_idx=last_row_idx,
                    last_sample_idx=last_sample_idx,
                    status="running",
                )

    _write_fail_csv(fail_rows, fail_path)
    _write_summary(
        summary_path,
        rank=rank,
        world_size=world_size,
        manifest_csv=manifest_csv,
        output_dir=output_dir,
        done=done,
        skipped=skipped,
        failed=failed,
        fail_path=fail_path,
        has_fail_rows=bool(fail_rows),
        last_row_idx=last_row_idx,
        last_sample_idx=last_sample_idx,
        status="finished",
    )
    LOGGER.info("Rank %d summary saved: %s", rank, summary_path)

    if world_size > 1 and dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            LOGGER.info("All ranks finished WAN VAE encoding")
        dist.destroy_process_group()
