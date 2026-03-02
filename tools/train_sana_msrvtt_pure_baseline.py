#!/usr/bin/env python3
"""
Pure SANA baseline training on MSR-VTT precomputed latents.

This script intentionally removes SmolVLM2/bridge from the training path:
prompt -> SANA native text encoder -> SANA DiT training loss.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

# Ensure repo root and third-party SANA are importable.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sana_root = project_root / "nets" / "third_party" / "sana"
if str(sana_root) not in sys.path:
    sys.path.insert(0, str(sana_root))

from nets.omni.datasets.openvid_dataset import OpenVidDataset, openvid_collate_fn
from nets.third_party.sana.diffusion.longsana.utils.model_wrapper import SanaTextEncoder

try:
    from diffusion import Scheduler as SanaScheduler
    from diffusion.model.respace import compute_density_for_timestep_sampling
except ModuleNotFoundError:
    from nets.third_party.sana.diffusion import Scheduler as SanaScheduler
    from nets.third_party.sana.diffusion.model.respace import compute_density_for_timestep_sampling

from tools.train_q1_sana_bridge import (
    AttrDict,
    build_data_info,
    load_sana_config,
    load_sana_diffusion_model,
    normalize_prompt,
    set_seed,
)
from tools.train_stage1_teacher_free import apply_lora_to_module, configure_dit_trainable


logger = logging.getLogger(__name__)


def to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attrdict(v) for v in obj]
    return obj


def load_config(path: str) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        return to_attrdict(yaml.safe_load(f))


def resolve_dtype(name: str) -> torch.dtype:
    name = str(name).lower().strip()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def setup_logging(log_path: str, is_main: bool = True) -> None:
    level = logging.INFO if is_main else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def save_trainable_ckpt(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    global_step: int,
    run_cfg: Dict[str, Any],
) -> None:
    state = {
        "global_step": int(global_step),
        "trainable_state": {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad},
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": (lr_scheduler.state_dict() if lr_scheduler is not None else None),
        "run_cfg": run_cfg,
    }
    torch.save(state, path)


def train(cfg: AttrDict, args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed = int(cfg.run.seed)
    set_seed(seed)

    dtype = resolve_dtype(args.precision or cfg.run.get("precision", "bf16"))
    device = torch.device(args.device or "cuda:0")

    output_dir = args.output_dir or cfg.run.output_dir
    run_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_path = args.log_file or os.path.join(
        "output",
        "logs",
        f"sana_msrvtt_pure_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logging(log_path, is_main=True)

    logger.info("Starting pure SANA baseline training")
    logger.info("Device=%s dtype=%s run_dir=%s", device, dtype, run_dir)

    sana_cfg = load_sana_config(cfg.sana.config)
    if getattr(cfg.sana, "ckpt_path", None):
        sana_cfg.model.load_from = str(cfg.sana.ckpt_path)

    diffusion_model = load_sana_diffusion_model(
        sana_cfg=sana_cfg,
        sana_ckpt_dir=str(cfg.sana.ckpt_dir),
        device=device,
        dtype=dtype,
        skip_ckpt_load=False,
    )
    diffusion_model.train()

    if bool(cfg.run.get("gradient_checkpointing", True)) and hasattr(diffusion_model, "enable_gradient_checkpointing"):
        diffusion_model.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing on DiT")

    # Freeze all first, then open trainables as requested.
    for p in diffusion_model.parameters():
        p.requires_grad = False

    dit_lora_cfg = cfg.model.get("dit_lora", AttrDict())
    if bool(dit_lora_cfg.get("enable", False)):
        replaced = apply_lora_to_module(
            diffusion_model,
            target_modules=dit_lora_cfg.get("target_modules", ["q_linear", "kv_linear", "proj"]),
            r=int(dit_lora_cfg.get("r", 8)),
            alpha=int(dit_lora_cfg.get("alpha", 16)),
            dropout=float(dit_lora_cfg.get("dropout", 0.05)),
            include_patterns=dit_lora_cfg.get("include_patterns", ["cross_attn"]),
            exclude_patterns=dit_lora_cfg.get("exclude_patterns", []),
        )
        logger.info("Applied DiT LoRA wrappers: replaced_linear_layers=%s", replaced)
    else:
        _ = configure_dit_trainable(diffusion_model, cfg.model.get("train_modules", ["all"]))

    trainable_params = [p for p in diffusion_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable DiT params in pure SANA baseline.")
    logger.info(
        "Trainable DiT params: tensors=%s numel=%.2fM",
        len(trainable_params),
        sum(p.numel() for p in trainable_params) / 1e6,
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.train.lr),
        betas=tuple(float(x) for x in cfg.train.get("betas", [0.9, 0.999])),
        eps=float(cfg.train.get("eps", 1e-8)),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    lr_warmup_steps = int(cfg.train.get("lr_warmup_steps", 0))
    if lr_warmup_steps > 0:
        def lr_lambda(current_step: int) -> float:
            if current_step < lr_warmup_steps:
                return float(current_step + 1) / float(max(lr_warmup_steps, 1))
            return 1.0
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        lr_scheduler = None

    scheduler_train_steps = int(getattr(cfg.sana, "train_sampling_steps", 1000))
    predict_flow_v = bool(getattr(sana_cfg.scheduler, "predict_flow_v", True))
    pred_sigma = bool(getattr(sana_cfg.scheduler, "pred_sigma", False))
    learn_sigma = bool(pred_sigma and getattr(sana_cfg.scheduler, "learn_sigma", False))
    sana_train_diffusion = SanaScheduler(
        str(scheduler_train_steps),
        noise_schedule=str(getattr(sana_cfg.scheduler, "noise_schedule", "linear_flow")),
        predict_flow_v=predict_flow_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        flow_shift=float(cfg.train.get("flow_shift", getattr(sana_cfg.scheduler, "flow_shift", 3.0))),
    )

    text_encoder = SanaTextEncoder(sana_cfg, device=device, dtype=dtype)
    text_encoder.eval()
    logger.info("Built SANA text encoder for pure baseline conditioning")

    openvid_cfg = cfg.data
    max_samples = args.max_samples if args.max_samples is not None else openvid_cfg.get("max_samples")
    dataset = OpenVidDataset(
        csv_path=str(openvid_cfg.csv_path),
        video_dir=str(openvid_cfg.video_dir),
        preprocessed_dir=str(openvid_cfg.preprocessed_dir),
        use_preprocessed=bool(openvid_cfg.get("use_preprocessed", True)),
        max_samples=int(max_samples) if max_samples is not None else None,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty for pure SANA baseline.")

    num_workers = int(cfg.run.get("num_workers", 4))
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.train.get("batch_size", 1)),
        shuffle=bool(cfg.train.get("shuffle", True)),
        drop_last=bool(cfg.train.get("drop_last", True)),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(cfg.run.get("persistent_workers", True)) and num_workers > 0,
        prefetch_factor=int(cfg.run.get("prefetch_factor", 2)) if num_workers > 0 else None,
        collate_fn=openvid_collate_fn,
    )
    logger.info("Dataloader ready: samples=%s batches=%s", len(dataset), len(dataloader))

    total_steps = int(args.total_steps or cfg.train.total_steps)
    log_every = int(args.log_every or cfg.run.log_every)
    save_every = int(args.save_every or cfg.run.save_every_steps)
    grad_accum = int(cfg.train.get("grad_accum_steps", 1))
    max_grad_norm = float(cfg.train.get("max_grad_norm", 1.0))
    latent_window_frames = int(cfg.train.get("latent_window_frames", 13))
    weighting_scheme = str(cfg.train.get("weighting_scheme", "logit_normal"))
    logit_mean = float(cfg.train.get("logit_mean", 0.0))
    logit_std = float(cfg.train.get("logit_std", 1.0))
    mode_scale = cfg.train.get("mode_scale", None)
    use_chi_prompt = bool(cfg.data.get("use_chi_prompt", False))
    logger.info(
        "Pure SANA settings: use_chi_prompt=%s latent_window_frames=%s lr=%.2e warmup_steps=%d grad_clip=%.3f",
        use_chi_prompt,
        latent_window_frames,
        float(cfg.train.lr),
        lr_warmup_steps,
        max_grad_norm,
    )

    global_step = 0
    micro_step = 0
    data_iter = iter(dataloader)
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prompts_raw: List[str] = [str(p) for p in batch["prompt"]]
        prompts = [
            normalize_prompt(
                p,
                normalize_whitespace=bool(cfg.data.get("normalize_whitespace", True)),
                strip=bool(cfg.data.get("strip", True)),
                remove_double_newlines=bool(cfg.data.get("remove_double_newlines", True)),
            )
            for p in prompts_raw
        ]

        with torch.no_grad():
            text_out = text_encoder.forward_chi(prompts, use_chi_prompt=use_chi_prompt)
            y = text_out["prompt_embeds"].to(device=device, dtype=dtype)
            mask = text_out["mask"].to(device=device, dtype=torch.long)

        latents = batch["latent_feature"].to(device=device, dtype=dtype)
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        if latent_window_frames > 0 and latents.shape[2] > latent_window_frames:
            latents = latents[:, :, :latent_window_frames, :, :]

        b, c, t, h, w = latents.shape
        timesteps = torch.randint(0, scheduler_train_steps, (b,), device=device).long()
        if weighting_scheme in {"logit_normal", "mode"}:
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=b,
                logit_mean=logit_mean,
                logit_std=logit_std,
                mode_scale=mode_scale,
            )
            timesteps = (u * scheduler_train_steps).long().to(device)

        model_kwargs = {
            "y": y.unsqueeze(1),  # [B, 1, L, C]
            "mask": mask,         # [B, L]
            "data_info": build_data_info(b, h, w, device=device),
        }
        loss_term = sana_train_diffusion.training_losses(
            diffusion_model,
            latents,
            timesteps,
            model_kwargs=model_kwargs,
        )
        diff_loss = loss_term["loss"].mean()
        loss = diff_loss / float(grad_accum)
        loss.backward()
        micro_step += 1

        if micro_step % grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % log_every == 0 or global_step == 1:
                dt = time.time() - t0
                logger.info(
                    "Step %d/%d loss=%.6f grad_norm=%.4f lr=%.2e dt=%.2fs prompt='%s'",
                    global_step,
                    total_steps,
                    float(diff_loss.detach().item()),
                    float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                    float(optimizer.param_groups[0]["lr"]),
                    dt,
                    prompts[0][:120],
                )
                t0 = time.time()

            if global_step % save_every == 0:
                ckpt_path = os.path.join(run_dir, f"checkpoint_step{global_step}.pt")
                save_trainable_ckpt(
                    ckpt_path,
                    diffusion_model,
                    optimizer,
                    lr_scheduler,
                    global_step,
                    run_cfg={"config": args.config, "pure_sana": True},
                )
                logger.info("Saved checkpoint: %s", ckpt_path)

    final_path = os.path.join(run_dir, "checkpoint_final.pt")
    save_trainable_ckpt(
        final_path,
        diffusion_model,
        optimizer,
        lr_scheduler,
        global_step,
        run_cfg={"config": args.config, "pure_sana": True},
    )
    logger.info("Training completed. Final checkpoint: %s", final_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure SANA baseline training on MSR-VTT")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sana_msrvtt_pure_baseline.yaml",
        help="Path to pure baseline yaml config",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.save_every is not None:
        cfg.run.save_every_steps = int(args.save_every)
    train(cfg, args)
