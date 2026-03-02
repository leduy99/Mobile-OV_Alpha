#!/usr/bin/env python3
"""
Runtime debugger for SANA-video inference stack.

This script helps answer:
- Which files/modules are actually used at runtime?
- Which sampler branch is active?
- Which flow_shift/cfg settings are resolved from config?

Usage:
  python tools/inference/debug_sana_video_runtime.py \
    --debug-config configs/sana_video_debug_runtime.yaml
"""

import argparse
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def _cfg_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def _resolve_debug_cfg(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "debug" in data and isinstance(data["debug"], dict):
        return data["debug"]
    return data


def _module_file(obj: Any) -> str:
    try:
        return str(Path(inspect.getfile(obj)).resolve())
    except Exception:
        return "unavailable"


def _build_runtime_report(debug_cfg: Dict[str, Any], svi_module: Any, sana_cfg: Any) -> Dict[str, Any]:
    # Import here so report shows exactly which resolved module file is used.
    from diffusion.model import builder as diffusion_builder
    from diffusion.model import utils as diffusion_utils

    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": str(_cfg_get(debug_cfg, "mode", "inspect")),
        "entry_script": str(Path(__file__).resolve()),
        "sana_infer_module_file": _module_file(svi_module),
        "diffusion_builder_file": _module_file(diffusion_builder),
        "diffusion_utils_file": _module_file(diffusion_utils),
        "sampler_available": {
            "sana_flow": bool(getattr(svi_module, "SANA_FLOW_AVAILABLE", False)),
            "sana_dpm": bool(getattr(svi_module, "SANA_DPM_AVAILABLE", False)),
            "diffusers_fallback": bool(getattr(svi_module, "DIFFUSERS_AVAILABLE", False)),
        },
        "resolved_inputs": {
            "config_file": str(Path(_cfg_get(debug_cfg, "config")).resolve()),
            "checkpoint_dir": str(Path(_cfg_get(debug_cfg, "checkpoint_dir")).resolve()),
            "device": str(_cfg_get(debug_cfg, "device", "cuda:0")),
            "prompt": str(_cfg_get(debug_cfg, "prompt", "")),
        },
        "resolved_sampling": {
            "sampling_algo": str(
                _cfg_get(debug_cfg, "sampling_algo", None)
                or getattr(sana_cfg.scheduler, "vis_sampler", "flow_euler")
            ),
            "cfg_scale": float(_cfg_get(debug_cfg, "cfg_scale", 7.0)),
            "inference_flow_shift_from_config": float(
                getattr(sana_cfg.scheduler, "inference_flow_shift", getattr(sana_cfg.scheduler, "flow_shift", 7.0))
            ),
            "num_inference_steps": int(_cfg_get(debug_cfg, "num_inference_steps", 8)),
        },
        "resolved_video_shape": {
            "num_frames": int(_cfg_get(debug_cfg, "num_frames", 81)),
            "height": int(_cfg_get(debug_cfg, "height", 480)),
            "width": int(_cfg_get(debug_cfg, "width", 832)),
            "vae_downsample_rate": int(getattr(sana_cfg.vae, "vae_downsample_rate", 8)),
            "vae_stride": getattr(sana_cfg.vae, "vae_stride", [1, 8, 8]),
            "vae_latent_dim": int(getattr(sana_cfg.vae, "vae_latent_dim", 16)),
        },
    }
    return report


def _save_report(report: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / f"sana_runtime_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug SANA-video runtime mapping")
    parser.add_argument(
        "--debug-config",
        type=str,
        default="configs/sana_video_debug_runtime.yaml",
        help="YAML debug config",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inspect", "smoke"],
        default=None,
        help="Optional override for debug mode from YAML.",
    )
    args = parser.parse_args()

    debug_cfg = _resolve_debug_cfg(args.debug_config)
    if args.mode is not None:
        debug_cfg["mode"] = args.mode

    # Import lazily so this script stays lightweight in inspect mode.
    from tools.inference import sana_video_inference_fixed as svi

    sana_cfg = svi.load_config_file(str(_cfg_get(debug_cfg, "config")))
    report = _build_runtime_report(debug_cfg, svi, sana_cfg)

    print("=" * 80)
    print("SANA Runtime Debug Report")
    print("=" * 80)
    print(f"sana_video_inference_fixed.py: {report['sana_infer_module_file']}")
    print(f"diffusion/model/builder.py  : {report['diffusion_builder_file']}")
    print(f"diffusion/model/utils.py    : {report['diffusion_utils_file']}")
    print(f"sampler availability         : {report['sampler_available']}")
    print(f"resolved sampling            : {report['resolved_sampling']}")
    print(f"resolved video shape         : {report['resolved_video_shape']}")

    out_dir = str(_cfg_get(debug_cfg, "output_dir", "output/sana_debug"))
    report_path = _save_report(report, out_dir)
    print(f"Saved debug report: {report_path}")

    mode = str(_cfg_get(debug_cfg, "mode", "inspect")).lower().strip()
    if mode != "smoke":
        print("Mode=inspect: done (no heavy model load).")
        return

    # smoke mode: do a tiny real run to validate pipeline end-to-end.
    print("Mode=smoke: loading models and running a tiny generation...")
    device = str(_cfg_get(debug_cfg, "device", "cuda:0"))
    model_dtype = svi.get_weight_dtype(sana_cfg.model.mixed_precision)
    vae_dtype = svi.get_weight_dtype(sana_cfg.vae.weight_dtype)
    height = int(_cfg_get(debug_cfg, "height", 480))
    width = int(_cfg_get(debug_cfg, "width", 832))
    latent_size = height // int(getattr(sana_cfg.vae, "vae_downsample_rate", 8))
    svi.set_env(int(_cfg_get(debug_cfg, "seed", 42)), latent_size)
    models = svi.load_sana_models(
        sana_cfg,
        checkpoint_dir=str(_cfg_get(debug_cfg, "checkpoint_dir")),
        device=device,
        model_dtype=model_dtype,
        vae_dtype=vae_dtype,
        latent_size=latent_size,
    )
    video = svi.generate_video(
        models=models,
        prompt=str(_cfg_get(debug_cfg, "prompt", "")),
        num_frames=int(_cfg_get(debug_cfg, "num_frames", 81)),
        height=height,
        width=width,
        num_inference_steps=int(_cfg_get(debug_cfg, "num_inference_steps", 8)),
        cfg_scale=float(_cfg_get(debug_cfg, "cfg_scale", 7.0)),
        seed=int(_cfg_get(debug_cfg, "seed", 42)),
        device=device,
        dtype=model_dtype,
        sampling_algo=str(_cfg_get(debug_cfg, "sampling_algo", "flow_euler")),
        negative_prompt=str(_cfg_get(debug_cfg, "negative_prompt", "")),
        motion_score=int(_cfg_get(debug_cfg, "motion_score", 10)),
        high_motion=bool(_cfg_get(debug_cfg, "high_motion", False)),
    )
    out_video = Path(out_dir) / f"sana_debug_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    svi.save_video(video, str(out_video), fps=16)
    print(f"Saved smoke video: {out_video}")


if __name__ == "__main__":
    main()
