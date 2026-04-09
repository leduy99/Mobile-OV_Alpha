#!/usr/bin/env python3
"""
Encode LAION/COYO image manifest into WAN VAE latents using SANA-video AR buckets.

This script:
- reads a unified manifest CSV (with image rows),
- picks closest size from ASPECT_RATIO_VIDEO_480_MS by original image ratio,
- center-crops + resizes image to bucket size,
- encodes with WanVAE,
- writes per-sample pickle compatible with OpenVidDataset direct_preprocessed_mode.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from PIL import Image

# Ensure repo root import path (for nets.third_party.wan).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets.third_party.wan.modules.vae import WanVAE


LOGGER = logging.getLogger(__name__)


# Match SANA-video 480 multi-scale buckets (diffusion/data/datasets/utils.py).
ASPECT_RATIO_VIDEO_480_MS: Dict[str, Tuple[int, int]] = {
    "0.5": (448, 896),
    "0.57": (480, 832),
    "0.68": (528, 768),
    "0.78": (560, 720),
    "1.0": (624, 624),
    "1.13": (672, 592),
    "1.29": (720, 560),
    "1.46": (768, 528),
    "1.67": (816, 496),
    "1.75": (832, 480),
    "2.0": (896, 448),
}


def _closest_bucket(height: int, width: int) -> Tuple[str, Tuple[int, int]]:
    ratio = float(height) / float(width)
    key = min(ASPECT_RATIO_VIDEO_480_MS.keys(), key=lambda k: abs(float(k) - ratio))
    return key, ASPECT_RATIO_VIDEO_480_MS[key]


def _to_int(value, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def _select_existing_path(*values: str) -> Optional[Path]:
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text or text.lower() == "nan":
            continue
        path = Path(text)
        if path.exists():
            return path
    return None


def _image_to_tensor(image: Image.Image, target_size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert one PIL image to normalized tensor [T=1, C, H, W].
    Uses center-crop with target aspect ratio, then resize.
    """
    target_h, target_w = target_size
    w, h = image.size
    ratio = float(target_w) / float(target_h)
    if w < h * ratio:
        crop_size = (int(float(w) / ratio), w)
    else:
        crop_size = (h, int(float(h) * ratio))

    transform = transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            transforms.Resize((target_h, target_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    tensor = transform(image.convert("RGB"))  # [C,H,W]
    return tensor.unsqueeze(0)  # [1,C,H,W]


def _setup_dist() -> Tuple[int, int, int, torch.device]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def encode_images(
    manifest_csv: Path,
    output_dir: Path,
    vae_ckpt: Path,
    max_samples: Optional[int],
    log_every: int,
) -> None:
    rank, world_size, local_rank, device = _setup_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    fail_csv = output_dir / f"failed_rank{rank:02d}.csv"
    summary_json = output_dir / f"summary_rank{rank:02d}.json"

    LOGGER.info("Rank %d/%d loading manifest: %s", rank, world_size, manifest_csv)
    df = pd.read_csv(manifest_csv)
    if "modality" in df.columns:
        df = df[df["modality"].astype(str).str.lower() == "image"]
    if max_samples is not None:
        df = df.iloc[: max_samples]
    df = df.reset_index(drop=True)

    if not vae_ckpt.exists():
        raise FileNotFoundError(f"WAN VAE checkpoint not found: {vae_ckpt}")
    LOGGER.info("Rank %d init WAN VAE: %s", rank, vae_ckpt)
    vae = WanVAE(vae_pth=str(vae_ckpt), device=device)

    done = 0
    skipped = 0
    failed = 0
    fail_rows = []

    with torch.no_grad():
        for row_idx, row in df.iterrows():
            if row_idx % world_size != rank:
                continue

            sample_idx = _to_int(row.get("sample_idx", row_idx), default=int(row_idx))
            out_path = output_dir / f"sample_{sample_idx:08d}.pkl"
            if out_path.exists():
                skipped += 1
                continue

            image_path = _select_existing_path(row.get("image_path"), row.get("media_path"), row.get("video_path"))
            if image_path is None:
                failed += 1
                fail_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "reason": "missing_image_path",
                        "image_path": "",
                    }
                )
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                orig_w, orig_h = image.size
                ar_key, closest_size = _closest_bucket(orig_h, orig_w)
                frames = _image_to_tensor(image, closest_size).to(device)  # [T=1,C,H,W]

                latent_feature = vae.encode(frames.transpose(0, 1).unsqueeze(0))[0].cpu()  # [C,T,H,W]

                item = {
                    "sample_idx": int(sample_idx),
                    "dataset": str(row.get("dataset", "laion_coyo")),
                    "modality": "image",
                    "video": str(row.get("source_id", f"sample_{sample_idx:08d}")),
                    "video_path": "",
                    "image_path": str(image_path),
                    "media_path": str(row.get("media_path", "")),
                    "prompt": str(row.get("caption", "")),
                    "part_user": _to_int(row.get("part_user", -1), default=-1),
                    "part_remote": _to_int(row.get("part_remote", -1), default=-1),
                    "frame_num": 1,
                    "target_size": [int(closest_size[0]), int(closest_size[1])],
                    "aspect_ratio": torch.tensor(float(ar_key), dtype=torch.float32),
                    "img_hw": torch.tensor([float(orig_h), float(orig_w)], dtype=torch.float32),
                    "closest_ratio": float(ar_key),
                    "latent_feature": latent_feature,
                }
                with out_path.open("wb") as f:
                    pickle.dump(item, f)
                done += 1

                if done % max(1, log_every) == 0:
                    LOGGER.info(
                        "Rank %d progress: done=%d skipped=%d failed=%d",
                        rank,
                        done,
                        skipped,
                        failed,
                    )
            except Exception as exc:  # pylint: disable=broad-except
                failed += 1
                fail_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "reason": str(exc),
                        "image_path": str(image_path),
                    }
                )

    if fail_rows:
        pd.DataFrame(fail_rows).to_csv(fail_csv, index=False)

    LOGGER.info(
        "Rank %d final counts: done=%d skipped=%d failed=%d",
        rank,
        done,
        skipped,
        failed,
    )

    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rank": rank,
        "world_size": world_size,
        "manifest_csv": str(manifest_csv),
        "output_dir": str(output_dir),
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "failed_csv": str(fail_csv) if fail_rows else None,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Rank %d summary saved: %s", rank, summary_json)

    if world_size > 1 and dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            LOGGER.info("All ranks finished image WAN VAE encoding")
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-csv",
        default="data/laion_coyo/manifests/laion_coyo_selected_media_existing_58k.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/laion_coyo/encoded/wan_vae_ivjoint_prep_58k_sana_ar",
    )
    parser.add_argument(
        "--vae-ckpt",
        default="omni_ckpts/sana_video_2b_480p/vae/Wan2.1_VAE.pth",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    torch.manual_seed(args.seed)
    np_seed = int(args.seed) % (2**32 - 1)
    try:
        import numpy as np  # local import to avoid hard dependency in non-runtime contexts

        np.random.seed(np_seed)
    except Exception:
        pass

    encode_images(
        manifest_csv=Path(args.manifest_csv).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        vae_ckpt=Path(args.vae_ckpt).resolve(),
        max_samples=args.max_samples,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
