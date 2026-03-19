#!/usr/bin/env python3
"""
Download WAN 2.1 VAE checkpoint file into local ckpt directory.

Example:
    python scripts/download_wan_vae_2_1.py --output-dir checkpoints/wan/wanxiang1_3b
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WAN 2.1 VAE checkpoint from Hugging Face.")
    parser.add_argument(
        "--repo-id",
        default="Wan-AI/Wan2.1-T2V-1.3B",
        help="HF repo id containing WAN VAE checkpoint.",
    )
    parser.add_argument(
        "--filename",
        default="Wan2.1_VAE.pth",
        help="Checkpoint filename in HF repo.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/wan/wanxiang1_3b",
        help="Local output directory to place checkpoint.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token for gated/private repo. If omitted, use cached login.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "huggingface_hub is required. Install with: pip install -U huggingface_hub"
        ) from exc

    print(f"[INFO] Downloading {args.filename} from {args.repo_id}")
    local_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=str(out_dir),
        token=args.token,
        resume_download=True,
    )

    final_path = out_dir / args.filename
    if not final_path.exists():
        # hf_hub_download may return cache path depending on version.
        # Ensure final target path is visible to users.
        shutil.copy2(local_path, final_path)

    print(f"[OK] Saved checkpoint: {final_path}")


if __name__ == "__main__":
    main()
