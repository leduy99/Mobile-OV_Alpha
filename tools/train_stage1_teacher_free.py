#!/usr/bin/env python3
"""
Teacher-free Stage 1 training for SANA-Video.

Key features:
- Uses real (prompt, video) pairs with diffusion/flow-matching loss on GT latents.
- No Gemma2 teacher required.
- Optional norm + variance regularizers to prevent collapse.
- Train student conditioner head + selected DiT modules (e.g., cross-attn).
"""

# Mini map (end-to-end):
# 1) Init distributed + seed + precision/device.
# 2) Build DiT (SANA) and student bridge, then wrap with FSDP/DDP depending on config.
# 3) Build OpenVid dataloader and optimizer/scheduler for bridge + selected DiT params.
# 4) For each micro-step: preprocess prompt -> student embeds -> noisy latent target -> DiT forward.
# 5) Compute losses, backward, gradient sync (auto via FSDP/DDP or manual all-reduce fallback).
# 6) On sync-step: clip grad, optimizer step, scheduler step, checkpoint save.
# 7) Save final checkpoint (student trainables + selected DiT trainables + step counters).

import argparse
import contextlib
import hashlib
import json
import logging
import math
import numpy as np
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import ShardingStrategy
try:
    import deepspeed
except Exception:  # pragma: no cover - optional runtime dependency
    deepspeed = None

import yaml
from pathlib import Path

# Ensure repo root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sana_root = project_root / "nets" / "third_party" / "sana"
if str(sana_root) not in sys.path:
    sys.path.insert(0, str(sana_root))

from nets.omni.datasets.openvid_dataset import OpenVidDataset, openvid_collate_fn
from nets.omni.modules.sana_prompt_bridge import SanaPromptBridge
try:
    from diffusion import Scheduler as SanaScheduler
    from diffusion.model.respace import (
        IncrementalTimesteps,
        compute_density_for_timestep_sampling,
        process_timesteps,
    )
except ModuleNotFoundError:
    from nets.third_party.sana.diffusion import Scheduler as SanaScheduler
    from nets.third_party.sana.diffusion.model.respace import (
        IncrementalTimesteps,
        compute_density_for_timestep_sampling,
        process_timesteps,
    )
try:
    from diffusion.utils.config import SanaVideoConfig
except ModuleNotFoundError:
    from nets.third_party.sana.diffusion.utils.config import SanaVideoConfig
try:
    from diffusion.longsana.utils.model_wrapper import SanaTextEncoder
except ModuleNotFoundError:
    from nets.third_party.sana.diffusion.longsana.utils.model_wrapper import SanaTextEncoder

# Reuse helpers from the distill script to avoid duplication.
from tools.train_q1_sana_bridge import (
    AttrDict,
    build_data_info,
    load_sana_config,
    load_sana_diffusion_model,
    normalize_prompt,
    pad_or_trim_teacher,
    set_seed,
    truncate_prompt,
    init_distributed,
)


logger = logging.getLogger(__name__)


def to_attrdict(obj: Any) -> Any:
    # Recursively convert nested dict/list/dataclass to AttrDict-style access.
    if isinstance(obj, dict):
        return AttrDict({k: to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attrdict(v) for v in obj]
    if is_dataclass(obj):
        return to_attrdict(obj.__dict__)
    return obj


def load_stage1_config(path: str) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return to_attrdict(cfg)


def parse_hf_uri(uri: str) -> Optional[Tuple[str, str]]:
    """Parse 'hf://org/repo/path/to/file' -> (repo_id, rel_path)."""
    if not isinstance(uri, str) or not uri.startswith("hf://"):
        return None
    raw = uri[len("hf://"):]
    parts = raw.split("/")
    if len(parts) < 3:
        return None
    repo_id = "/".join(parts[:2])
    rel_path = "/".join(parts[2:])
    return repo_id, rel_path


def download_hf_files(repo_id: str, rel_paths: List[str], local_dir: str) -> None:
    """Download selected files from HF repo into local_dir."""
    rel_paths = sorted(set([p for p in rel_paths if p]))
    if not rel_paths:
        return
    os.makedirs(local_dir, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=rel_paths,
            resume_download=True,
        )
        return
    except Exception as exc:
        logger.warning("huggingface_hub snapshot_download failed (%s), fallback to CLI.", exc)

    hf_cli = shutil.which("huggingface-cli")
    if hf_cli is None:
        raise RuntimeError(
            "Missing dependency to download checkpoints: install huggingface_hub "
            "(pip install huggingface_hub) or huggingface-cli."
        )
    cmd = [
        hf_cli,
        "download",
        repo_id,
        "--local-dir",
        local_dir,
        "--local-dir-use-symlinks",
        "False",
    ]
    for rel in rel_paths:
        cmd += ["--include", rel]
    subprocess.run(cmd, check=True)


def ensure_sana_assets_available(
    sana_cfg: SanaVideoConfig,
    sana_ckpt_dir: str,
    is_main: bool,
    local_rank: int,
) -> None:
    """
    Ensure SANA DiT/VAE files referenced by hf:// config paths exist locally.
    """
    uri_list = [
        getattr(getattr(sana_cfg, "model", AttrDict()), "load_from", ""),
        getattr(getattr(sana_cfg, "vae", AttrDict()), "vae_pretrained", ""),
    ]
    missing_by_repo: Dict[str, List[str]] = {}
    required_local: List[str] = []
    for uri in uri_list:
        parsed = parse_hf_uri(str(uri))
        if parsed is None:
            continue
        repo_id, rel_path = parsed
        local_path = os.path.join(sana_ckpt_dir, rel_path)
        required_local.append(local_path)
        if not os.path.exists(local_path):
            missing_by_repo.setdefault(repo_id, []).append(rel_path)

    needs_download_local = bool(missing_by_repo)

    # Fast path on shared local storage: if all required assets already exist,
    # avoid any early distributed collectives here. This bootstrap stage happens
    # immediately after process-group init, and we've seen NCCL occasionally
    # wedge on the all_reduce below even when there is nothing to download.
    if not needs_download_local:
        for local_path in required_local:
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Required SANA asset missing before auto-download check: {local_path}"
                )
        return

    if missing_by_repo and is_main:
        logger.info("SANA checkpoints missing, auto-downloading to %s ...", sana_ckpt_dir)
        for repo_id, rels in missing_by_repo.items():
            logger.info("Downloading from %s: %s", repo_id, rels)
            download_hf_files(repo_id=repo_id, rel_paths=rels, local_dir=sana_ckpt_dir)

    # Avoid early NCCL barrier when nothing needs to be downloaded.
    # Some clusters can intermittently hang on very-early barriers.
    needs_download_global = needs_download_local
    if dist.is_initialized():
        flag = torch.tensor([1 if needs_download_local else 0], device=f"cuda:{local_rank}", dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        needs_download_global = bool(int(flag.item()))
    if needs_download_global:
        safe_barrier(local_rank)

    for local_path in required_local:
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Required SANA asset missing after auto-download attempt: {local_path}"
            )


def ensure_smolvlm2_checkpoint_available(
    ckpt_path: str,
    is_main: bool,
    local_rank: int,
) -> None:
    """
    Ensure converted SmolVLM2 checkpoint exists; auto-convert from HF when missing.
    """
    if os.path.exists(ckpt_path):
        return

    if is_main:
        ckpt_dir = os.path.dirname(ckpt_path) or "."
        os.makedirs(ckpt_dir, exist_ok=True)
        model_id = os.environ.get("SMOLVLM2_MODEL_ID", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        convert_device = os.environ.get("SMOLVLM2_CONVERT_DEVICE", "cpu")
        converter = os.path.join(project_root, "tools", "convert_weights", "convert_smolvlm2_weight.py")
        if not os.path.exists(converter):
            raise FileNotFoundError(f"SmolVLM2 converter script not found: {converter}")
        logger.info("SmolVLM2 checkpoint missing, auto-converting from HF model: %s", model_id)
        cmd = [
            sys.executable,
            converter,
            "--model-id",
            model_id,
            "--output-path",
            ckpt_path,
            "--device",
            convert_device,
        ]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if dist.is_initialized():
        safe_barrier(local_rank)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"SmolVLM2 checkpoint missing after auto-convert attempt: {ckpt_path}"
        )


def apply_template(prompt: str, templates: List[str], motion_score: int, rng: random.Random) -> str:
    if not templates:
        return prompt
    template = rng.choice(templates)
    return template.format(prompt=prompt, motion_score=int(motion_score))


def masked_mean_pool(embeds: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(embeds.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(eps)
    return (embeds * mask_f).sum(dim=1) / denom.squeeze(1)


class PrecomputedTeacherStore:
    """Load teacher embedding shards and provide fast by-index lookup."""

    def __init__(self, root_dir: str, preload: bool = True, max_cached_shards: int = 2):
        manifest_path = os.path.join(root_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Teacher manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        shards = manifest.get("shards", [])
        if not shards:
            raise RuntimeError(f"No shards found in manifest: {manifest_path}")

        self.root_dir = root_dir
        self.preload = bool(preload)
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.shard_files = [os.path.join(root_dir, s["file"]) for s in shards]
        self.shard_ranges = [
            (
                sid,
                int(s.get("sample_idx_min", -1)),
                int(s.get("sample_idx_max", -1)),
            )
            for sid, s in enumerate(shards)
        ]
        self._shards: Dict[int, Dict[str, torch.Tensor]] = {}
        self._shard_lru: List[int] = []
        self._index: Dict[int, Tuple[int, int]] = {}

        if self.preload:
            for shard_id in range(len(self.shard_files)):
                self._load_shard(shard_id)
        else:
            # Build global sample_idx -> (shard,row) index once to avoid expensive
            # per-batch shard scans that can desync ranks and stall NCCL collectives.
            self._build_index()

    def _build_index(self) -> None:
        for shard_id, path in enumerate(self.shard_files):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Teacher shard not found: {path}")
            shard = torch.load(path, map_location="cpu")
            sample_idx = shard["sample_idx"].tolist()
            for row, idx in enumerate(sample_idx):
                self._index[int(idx)] = (shard_id, row)
            # Do not keep shard tensors in memory during index build.
            del shard

    def _touch_lru(self, shard_id: int) -> None:
        if shard_id in self._shard_lru:
            self._shard_lru.remove(shard_id)
        self._shard_lru.append(shard_id)

    def _evict_if_needed(self) -> None:
        if self.preload:
            return
        while len(self._shards) > self.max_cached_shards:
            victim = self._shard_lru.pop(0)
            self._shards.pop(victim, None)

    def _load_shard(self, shard_id: int) -> None:
        if shard_id in self._shards:
            self._touch_lru(shard_id)
            return
        path = self.shard_files[shard_id]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Teacher shard not found: {path}")
        shard = torch.load(path, map_location="cpu")
        sample_idx = shard["sample_idx"].tolist()
        for row, idx in enumerate(sample_idx):
            self._index[int(idx)] = (shard_id, row)
        self._shards[shard_id] = shard
        self._touch_lru(shard_id)
        self._evict_if_needed()

    def fetch(self, sample_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (teacher_embeds, teacher_mask) for sample indices [B]."""
        if sample_idx.dim() != 1:
            sample_idx = sample_idx.view(-1)
        idx_list = [int(v) for v in sample_idx.tolist()]
        shard_rows: List[Tuple[int, int]] = []
        for idx in idx_list:
            pos = self._index.get(idx)
            if pos is None:
                # Fallback scan (should be rare when _build_index is available).
                for sid in range(len(self.shard_files)):
                    self._load_shard(sid)
                    pos = self._index.get(idx)
                    if pos is not None:
                        break
            if pos is None:
                raise KeyError(f"sample_idx={idx} not found in precomputed teacher shards")
            shard_rows.append(pos)

        embeds_out = []
        masks_out = []
        for shard_id, row in shard_rows:
            self._load_shard(shard_id)
            shard = self._shards[shard_id]
            embeds_out.append(shard["prompt_embeds"][row])
            masks_out.append(shard["mask"][row])
        return torch.stack(embeds_out, dim=0), torch.stack(masks_out, dim=0)


def compute_masked_mse(student: torch.Tensor, teacher: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(student, teacher)
    mask_f = mask.unsqueeze(-1).to(student.dtype)
    denom = mask_f.sum().clamp_min(1.0) * student.shape[-1]
    return (((student - teacher) ** 2) * mask_f).sum() / denom


def compute_masked_token_cos(student: torch.Tensor, teacher: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    cos = F.cosine_similarity(student, teacher, dim=-1)
    if mask is None:
        return 1.0 - cos.mean()
    mask_f = mask.to(cos.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return 1.0 - ((cos * mask_f).sum() / denom)


def _unwrap_module_for_attrs(module: nn.Module) -> nn.Module:
    base = module
    while hasattr(base, "module"):
        base = base.module
    return base


@contextlib.contextmanager
def freeze_module_params(modules: List[Optional[nn.Module]]):
    params_with_state: List[Tuple[nn.Parameter, bool]] = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            params_with_state.append((param, bool(param.requires_grad)))
            param.requires_grad_(False)
    try:
        yield
    finally:
        for param, old_requires_grad in params_with_state:
            param.requires_grad_(old_requires_grad)


def canonicalize_distill_target_space(name: str) -> str:
    key = str(name or "sana_post_ynorm").strip().lower()
    aliases = {
        "raw_ln": "raw_layernorm",
        "raw_layernorm": "raw_layernorm",
        "layernorm_raw": "raw_layernorm",
        "raw": "raw",
        "teacher_raw": "raw",
        "post_yproj": "sana_post_yproj",
        "sana_post_yproj": "sana_post_yproj",
        "yproj": "sana_post_yproj",
        "post_ynorm": "sana_post_ynorm",
        "sana_post_ynorm": "sana_post_ynorm",
        "ynorm": "sana_post_ynorm",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported distill.target_space={name!r}")
    return aliases[key]


def project_sana_conditioning_space(
    diffusion_model: nn.Module,
    embeds: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    target_space: str,
    freeze_modules_for_grad: bool,
) -> torch.Tensor:
    target_space = canonicalize_distill_target_space(target_space)
    if target_space == "raw":
        return embeds.float()
    if target_space == "raw_layernorm":
        return F.layer_norm(embeds.float(), (embeds.shape[-1],))

    diffusion_module = _unwrap_module_for_attrs(diffusion_model)
    y_embedder = getattr(diffusion_module, "y_embedder", None)
    if y_embedder is None:
        raise RuntimeError("distill.target_space requires diffusion_model.y_embedder, but it is unavailable")

    y = embeds
    if y.dim() == 2:
        y = y.unsqueeze(0)
    if y.dim() == 3:
        y = y.unsqueeze(1)
    elif y.dim() != 4:
        raise ValueError(f"Expected embeds with 3D/4D shape, got {tuple(embeds.shape)}")

    mask_4d = None
    if mask is not None:
        mask_4d = mask
        if mask_4d.dim() == 1:
            mask_4d = mask_4d.unsqueeze(0)
        if mask_4d.dim() == 2:
            mask_4d = mask_4d.unsqueeze(1).unsqueeze(1)

    condition_modules: List[Optional[nn.Module]] = [y_embedder]
    if target_space == "sana_post_ynorm" and bool(getattr(diffusion_module, "y_norm", False)):
        condition_modules.append(getattr(diffusion_module, "attention_y_norm", None))
    ctx = freeze_module_params(condition_modules) if freeze_modules_for_grad else contextlib.nullcontext()
    with ctx:
        y = y_embedder(y, False, mask=mask_4d)
        if target_space == "sana_post_ynorm" and bool(getattr(diffusion_module, "y_norm", False)):
            attention_y_norm = getattr(diffusion_module, "attention_y_norm", None)
            if attention_y_norm is None:
                raise RuntimeError("distill.target_space=sana_post_ynorm but diffusion_model.attention_y_norm is unavailable")
            y = attention_y_norm(y)

    if y.dim() == 4 and y.shape[1] == 1:
        y = y.squeeze(1)
    return y.float()


def extract_diffusion_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        if isinstance(output.get("x"), torch.Tensor):
            return output["x"]
        if isinstance(output.get("sample"), torch.Tensor):
            return output["sample"]
    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(f"Unsupported diffusion output type: {type(output)}")


def compute_prompt_anticollapse_losses(
    pooled: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    VICReg-style anti-collapse regularizer on pooled prompt embeddings.
    Returns (variance_loss, covariance_loss).
    """
    if pooled.dim() != 2:
        pooled = pooled.view(pooled.shape[0], -1)

    centered = pooled - pooled.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(float(target_std) - std).mean()

    if centered.shape[0] <= 1:
        cov_loss = torch.zeros_like(var_loss)
    else:
        cov = (centered.T @ centered) / float(centered.shape[0] - 1)
        cov = cov - torch.diag(torch.diag(cov))
        cov_loss = cov.pow(2).mean()
    return var_loss, cov_loss


def pad_or_trim_tokens(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad/trim token embeddings [B, L, C] to a target token length."""
    if tokens.dim() != 3:
        raise ValueError(f"Expected 3D tokens [B, L, C], got shape={tuple(tokens.shape)}")
    cur_len = int(tokens.shape[1])
    if cur_len == target_len:
        return tokens
    if cur_len > target_len:
        return tokens[:, :target_len, :]
    pad_len = target_len - cur_len
    pad = torch.zeros(
        (tokens.shape[0], pad_len, tokens.shape[2]),
        device=tokens.device,
        dtype=tokens.dtype,
    )
    return torch.cat([tokens, pad], dim=1)


def pad_or_trim_token_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad/trim token mask [B, L] to a target token length."""
    if mask.dim() != 2:
        raise ValueError(f"Expected 2D mask [B, L], got shape={tuple(mask.shape)}")
    cur_len = int(mask.shape[1])
    if cur_len == target_len:
        return mask
    if cur_len > target_len:
        return mask[:, :target_len]
    pad_len = target_len - cur_len
    pad = torch.zeros((mask.shape[0], pad_len), device=mask.device, dtype=mask.dtype)
    return torch.cat([mask, pad], dim=1)


def offdiag_similarity_vector(pooled: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Return off-diagonal cosine similarities for pooled vectors [B, D].
    With B<2, returns an empty vector.
    """
    if pooled.dim() != 2:
        pooled = pooled.view(pooled.shape[0], -1)
    pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + eps)
    bsz = int(pooled.shape[0])
    if bsz < 2:
        return pooled.new_zeros((0,))
    sim = pooled @ pooled.T
    offdiag = sim[~torch.eye(bsz, dtype=torch.bool, device=sim.device)]
    return offdiag


def compute_geometry_preservation_loss(
    pooled_student: torch.Tensor,
    pooled_source: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Preserve in-batch geometry by matching off-diagonal similarity structure:
      MSE(offdiag(sim(student)), offdiag(sim(source))).
    """
    off_student = offdiag_similarity_vector(pooled_student, eps=eps)
    off_source = offdiag_similarity_vector(pooled_source, eps=eps)
    if off_student.numel() == 0 or off_source.numel() == 0:
        return pooled_student.new_zeros(())
    if off_student.shape != off_source.shape:
        n = min(off_student.numel(), off_source.numel())
        off_student = off_student[:n]
        off_source = off_source[:n]
    return F.mse_loss(off_student, off_source)


def compute_inbatch_contrastive_distill_loss(
    pooled_student: torch.Tensor,
    pooled_teacher: torch.Tensor,
    *,
    temperature: float = 0.07,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Match each student prompt to its corresponding teacher prompt while pushing
    away other prompts in the batch.
    """
    if pooled_student.dim() != 2:
        pooled_student = pooled_student.view(pooled_student.shape[0], -1)
    if pooled_teacher.dim() != 2:
        pooled_teacher = pooled_teacher.view(pooled_teacher.shape[0], -1)
    if pooled_student.shape[0] <= 1 or pooled_teacher.shape[0] <= 1:
        return pooled_student.new_zeros(())
    temp = max(float(temperature), 1e-4)
    student_n = pooled_student / pooled_student.norm(dim=-1, keepdim=True).clamp_min(eps)
    teacher_n = pooled_teacher / pooled_teacher.norm(dim=-1, keepdim=True).clamp_min(eps)
    logits = (student_n @ teacher_n.T) / temp
    targets = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)


def configure_dit_trainable(model: torch.nn.Module, train_modules: List[str]) -> List[torch.nn.Parameter]:
    trainable = []
    # Special tokens to train full DiT without listing module name substrings.
    train_all = any(str(key).lower() in {"all", "*"} for key in (train_modules or []))
    if not train_modules:
        for p in model.parameters():
            p.requires_grad = False
        return trainable

    for name, param in model.named_parameters():
        if train_all or any(key in name for key in train_modules):
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False
    return trainable


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for nn.Linear used in DiT/Smol fine-tuning."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        self.r = int(max(0, r))
        self.alpha = int(max(1, alpha))
        self.scaling = float(self.alpha) / float(self.r) if self.r > 0 else 1.0
        if self.r > 0:
            self.lora_A = nn.Parameter(
                torch.zeros(self.r, self.in_features, device=base.weight.device, dtype=base.weight.dtype)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(self.out_features, self.r, device=base.weight.device, dtype=base.weight.dtype)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_input = F.dropout(x, p=self.dropout, training=self.training)
            update = F.linear(lora_input, self.lora_A)
            update = F.linear(update, self.lora_B)
            result = result + update * self.scaling
        return result


def apply_lora_to_module(
    module: torch.nn.Module,
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> int:
    """Replace matched nn.Linear with LoRA wrappers and freeze base params."""
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []
    target_set = set(target_modules or [])

    named_modules = list(module.named_modules())
    replaced = 0
    for name, submodule in named_modules:
        if not isinstance(submodule, nn.Linear):
            continue
        if include_patterns and not any(pat in name for pat in include_patterns):
            continue
        if exclude_patterns and any(pat in name for pat in exclude_patterns):
            continue
        if target_set and not any(name.endswith(t) or t in name for t in target_set):
            continue

        parent_path = name.split(".")[:-1]
        leaf = name.split(".")[-1]
        parent = module
        for p in parent_path:
            parent = getattr(parent, p)
        wrapped = LoRALinear(submodule, r=r, alpha=alpha, dropout=dropout)
        wrapped.to(submodule.weight.device, dtype=submodule.weight.dtype)
        if hasattr(parent, "_modules") and leaf in parent._modules:
            parent._modules[leaf] = wrapped
            replaced += 1

    # Freeze all params then re-enable LoRA params only.
    for p in module.parameters():
        p.requires_grad = False
    for m in module.modules():
        if hasattr(m, "lora_A"):
            m.lora_A.requires_grad = True
        if hasattr(m, "lora_B"):
            m.lora_B.requires_grad = True
    return replaced


def get_sorted_trainable_named_params(module: torch.nn.Module) -> List[tuple[str, torch.nn.Parameter]]:
    return sorted([(n, p) for n, p in module.named_parameters() if p.requires_grad], key=lambda x: x[0])


def log_trainable_signature(
    named_params: List[tuple[str, torch.nn.Parameter]],
    tag: str,
    rank: int,
    world_size: int,
) -> None:
    rows: List[str] = []
    total_numel = 0
    for name, param in named_params:
        rows.append(f"{name}|{tuple(param.shape)}|{param.dtype}")
        total_numel += int(param.numel())
    signature = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    local_summary = {
        "rank": int(rank),
        "count": len(named_params),
        "numel": int(total_numel),
        "sig": signature,
        "first": [n for n, _ in named_params[:3]],
        "last": [n for n, _ in named_params[-3:]],
    }

    # Intentionally avoid distributed collectives here; this probe should never
    # introduce a new sync point while debugging hangs.
    logger.info(
        "param_signature[%s] rank=%s/%s tensors=%s numel=%s sig=%s first=%s last=%s",
        tag,
        local_summary["rank"],
        world_size,
        local_summary["count"],
        local_summary["numel"],
        local_summary["sig"][:16],
        local_summary["first"],
        local_summary["last"],
    )


def build_student(
    cfg: AttrDict,
    device: torch.device,
    dtype: torch.dtype,
    *,
    strict_sana_parity_text_path: bool = False,
    strict_sana_use_full_text_window: bool = False,
    strict_sana_token_select_strategy: str = "tail",
    strict_sana_head_tokens: int = 96,
    strict_sana_tail_tokens: int = 96,
    strict_fail_fast_mask: bool = False,
    sana_model_max_length: int = 300,
    sana_chi_prompt_text: str = "",
) -> SanaPromptBridge:
    # Build prompt bridge on top of the requested frozen VLM backbone.
    student_cfg = cfg.model.student
    bridge_cfg = student_cfg.conditioner_bridge
    text_encoder_cfg = student_cfg.text_encoder
    backbone_type = str(text_encoder_cfg.get("backbone_type", "smolvlm2")).lower()
    logger.info(
        "build_student: backbone_type=%s ckpt_path=%s projector_type=%s",
        backbone_type,
        text_encoder_cfg.ckpt_path,
        student_cfg.get("projector", {}).get("type", "legacy"),
    )
    common_kwargs = dict(
        adapter_ckpt_dir=student_cfg.get("adapter_ckpt_dir", "omni_ckpts/wan/wanxiang1_3b/adapter"),
        adapter_in_channels=student_cfg.get("adapter_in_channels", 1152),
        adapter_out_channels=student_cfg.get("adapter_out_channels", 4096),
        adapter_query_length=student_cfg.get("adapter_query_length", 64),
        adapter_num_encoder_layers=student_cfg.get("adapter_num_encoder_layers", 4),
        adapter_num_decoder_layers=student_cfg.get("adapter_num_decoder_layers", 4),
        adapter_ff_mult=student_cfg.get("adapter_ff_mult", 4),
        num_prompt_queries=bridge_cfg.out_seq_len,
        caption_channels=bridge_cfg.out_dim,
        precision_dtype=dtype,
        device=device,
        tokenizer_model_id=text_encoder_cfg.get("tokenizer_model_id", "HuggingFaceTB/SmolVLM-Instruct"),
        force_adapter_query_length=student_cfg.get("force_adapter_query_length"),
        max_length=text_encoder_cfg.max_length,
        use_vision_head=student_cfg.get("use_vision_head", True),
        resampler_num_heads=student_cfg.get("resampler_num_heads", 16),
        resampler_mlp_mult=student_cfg.get("resampler_mlp_mult", 4),
        lora_enable=student_cfg.get("lora", {}).get("enable", False),
        gate_min_value=student_cfg.get("gate_min_value", 0.0),
        projector_type=student_cfg.get("projector", {}).get("type", "legacy"),
        mcp_hidden_dim=student_cfg.get("projector", {}).get("mcp_hidden_dim", 512),
        mcp_num_fuse_layers=student_cfg.get("projector", {}).get("mcp_num_fuse_layers", 2),
        mcp_use_refine=student_cfg.get("projector", {}).get("mcp_use_refine", True),
        mcp_refine_kernel_size=student_cfg.get("projector", {}).get("mcp_refine_kernel_size", 3),
        mcp_fusion_temperature=student_cfg.get("projector", {}).get("mcp_fusion_temperature", 1.0),
        mcp_lexical_bottleneck_dim=student_cfg.get("projector", {}).get("mcp_lexical_bottleneck_dim", 256),
        mcp_lexical_gate_init=student_cfg.get("projector", {}).get("mcp_lexical_gate_init", 0.05),
        strict_sana_parity_text_path=bool(strict_sana_parity_text_path),
        strict_sana_use_full_text_window=bool(strict_sana_use_full_text_window),
        strict_sana_token_select_strategy=str(strict_sana_token_select_strategy or "tail"),
        strict_sana_head_tokens=int(strict_sana_head_tokens),
        strict_sana_tail_tokens=int(strict_sana_tail_tokens),
        fail_fast_mask=bool(strict_fail_fast_mask),
        sana_model_max_length=int(sana_model_max_length),
        sana_chi_prompt=sana_chi_prompt_text,
    )
    if backbone_type in {"qwen3_vl", "qwen3vl", "qwen"}:
        from nets.omni.modules.sana_prompt_bridge_qwen3vl import Qwen3VLSanaPromptBridge

        student = Qwen3VLSanaPromptBridge(
            qwen_ckpt_path=text_encoder_cfg.ckpt_path,
            **common_kwargs,
        )
    else:
        student = SanaPromptBridge(
            smolvlm2_ckpt_path=text_encoder_cfg.ckpt_path,
            smol_vh_num_queries=student_cfg.get("smol_vh_num_queries", 1),
            lora_r=student_cfg.get("lora", {}).get("r", 8),
            lora_alpha=student_cfg.get("lora", {}).get("alpha", 16),
            lora_dropout=student_cfg.get("lora", {}).get("dropout", 0.05),
            lora_target_modules=student_cfg.get("lora", {}).get("target_modules"),
            lora_include_patterns=student_cfg.get("lora", {}).get("include_patterns"),
            lora_exclude_patterns=student_cfg.get("lora", {}).get("exclude_patterns"),
            **common_kwargs,
        )

    # Ensure consistent trainable flags across ranks, then apply the text-tower
    # training policy (frozen / LoRA / top-N full layers).
    student.requires_grad_(True)
    text_enc_cfg = student_cfg.get("text_encoder", AttrDict())
    lora_enabled = bool(student_cfg.get("lora", {}).get("enable", False))
    student.smolvlm2_model.eval().requires_grad_(False)
    if lora_enabled:
        # Important: SmolVLMModel.forward internally switches to no_grad() when
        # module.training == False. Keep train-mode so LoRA can receive gradients,
        # while base params remain frozen via requires_grad=False.
        student.smolvlm2_model.train()
        # Re-enable LoRA params after freezing the text tower
        for name, p in student.smolvlm2_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = True

    def _get_text_model_root(module: torch.nn.Module) -> Optional[torch.nn.Module]:
        root = getattr(module, "_model", module)
        if hasattr(root, "text_model"):
            return root.text_model
        if hasattr(root, "model") and hasattr(root.model, "text_model"):
            return root.model.text_model
        if hasattr(root, "language_model"):
            lm = root.language_model
            if hasattr(lm, "model"):
                return lm.model
            return lm
        return None

    def _find_text_layers(text_model: torch.nn.Module):
        candidate_chains = [
            ("model", "layers"),
            ("layers",),
            ("encoder", "layers"),
            ("decoder", "layers"),
            ("transformer", "h"),
        ]
        for chain in candidate_chains:
            cur = text_model
            ok = True
            for attr in chain:
                if not hasattr(cur, attr):
                    ok = False
                    break
                cur = getattr(cur, attr)
            if ok and isinstance(cur, (torch.nn.ModuleList, list, tuple)) and len(cur) > 0:
                return list(cur)
        return None

    train_top_layers = int(getattr(text_enc_cfg, "train_top_layers", 0) or 0)
    train_final_norm = bool(getattr(text_enc_cfg, "train_final_norm", train_top_layers > 0))
    train_embeddings = bool(getattr(text_enc_cfg, "train_embeddings", False))
    text_model = _get_text_model_root(student.smolvlm2_model)
    if train_top_layers > 0 and text_model is not None:
        layers = _find_text_layers(text_model)
        if layers:
            top_layers = layers[-min(train_top_layers, len(layers)) :]
            for layer in top_layers:
                layer.requires_grad_(True)
        if train_final_norm:
            for norm_name in ("norm", "final_layernorm", "final_norm", "post_attention_layernorm"):
                if hasattr(text_model, norm_name):
                    getattr(text_model, norm_name).requires_grad_(True)
        if train_embeddings and hasattr(text_model, "get_input_embeddings"):
            emb = text_model.get_input_embeddings()
            if emb is not None:
                emb.requires_grad_(True)

    if student_cfg.get("vision_head_frozen", False) and getattr(student, "smolvlm2_vision_head", None) is not None:
        student.smolvlm2_vision_head.eval().requires_grad_(False)

    # MCP path does not consume adapter_output_gate; keep checkpoint key but do not optimize it.
    proj_type = str(student_cfg.get("projector", {}).get("type", "legacy")).lower()
    if proj_type in {"mcp_tiny", "mcp_full", "mcp_lexical_gated", "mcp_lexical_bottleneck"} and hasattr(student, "adapter_output_gate"):
        student.adapter_output_gate.requires_grad_(False)

    return student


def extract_lora_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    lora_state: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.detach().cpu()
    return lora_state


def extract_trainable_smolvlm2_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    trainable_names = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and ("lora_A" not in name and "lora_B" not in name)
    }
    if not trainable_names:
        return {}
    state_dict = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if name in trainable_names
    }


def load_lora_state_dict(model: torch.nn.Module, lora_state: Dict[str, torch.Tensor], is_main: bool) -> None:
    if not lora_state:
        return
    named = dict(model.named_parameters())
    loaded = 0
    missing = 0
    for name, tensor in lora_state.items():
        target = named.get(name)
        if target is None:
            missing += 1
            continue
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
        loaded += 1
    if is_main:
        logger.info("Loaded LoRA params: loaded=%d missing=%d", loaded, missing)


def load_trainable_smolvlm2_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor], is_main: bool) -> None:
    if not state:
        return
    named = dict(model.named_parameters())
    loaded = 0
    missing = 0
    for name, tensor in state.items():
        target = named.get(name)
        if target is None:
            missing += 1
            continue
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
        loaded += 1
    if is_main:
        logger.info("Loaded trainable SmolVLM2 params: loaded=%d missing=%d", loaded, missing)


def get_trainable_dit_state_dict(diffusion_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    # Save only trainable DiT params to keep checkpoint small and focused.
    base_model = diffusion_model.module if hasattr(diffusion_model, "module") else diffusion_model
    trainable_names = {name for name, p in base_model.named_parameters() if p.requires_grad}
    if not trainable_names:
        return {}

    if isinstance(diffusion_model, FSDP):
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(diffusion_model, StateDictType.FULL_STATE_DICT, full_cfg):
            full_state = diffusion_model.state_dict()
    else:
        full_state = base_model.state_dict()

    return {k: v for k, v in full_state.items() if k in trainable_names}


def allreduce_gradients(
    params: List[torch.nn.Parameter],
    world_size: int,
    rank: int = 0,
    tag: str = "grad",
    debug_log_every: int = 0,
    trace_each: bool = False,
    bucket_max_elems: int = 25_000_000,
    group=None,
    backend: str = "nccl",
) -> None:
    # Manual gradient synchronization path (used when neither DDP nor FSDP wraps a module).
    if world_size <= 1 or not dist.is_initialized():
        return
    total = len(params)
    t0 = time.time()
    bucket_max_elems = max(int(bucket_max_elems), 1)

    # Flatten multiple grad tensors into a single all-reduce buffer per bucket.
    bucket: List[tuple[int, torch.nn.Parameter, torch.Tensor]] = []
    bucket_elems = 0

    backend = (backend or "nccl").lower()

    def flush_bucket() -> None:
        nonlocal bucket, bucket_elems
        if not bucket:
            return
        flat = torch.cat([g.reshape(-1) for _, _, g in bucket], dim=0)
        reduce_buf = flat if backend != "gloo" else flat.cpu()
        dist.all_reduce(reduce_buf, op=dist.ReduceOp.SUM, group=group)
        reduce_buf /= float(world_size)

        offset = 0
        for _, p_inner, g_inner in bucket:
            numel = g_inner.numel()
            reduced = reduce_buf[offset : offset + numel].view_as(g_inner)
            if p_inner.grad is None:
                p_inner.grad = reduced.to(device=p_inner.device, dtype=p_inner.dtype)
            else:
                p_inner.grad.copy_(reduced.to(device=p_inner.grad.device, dtype=p_inner.grad.dtype))
            offset += numel
        bucket = []
        bucket_elems = 0

    for idx, p in enumerate(params):
        grad = p.grad
        if rank == 0 and trace_each:
            logger.info(
                "%s reduce idx=%d/%d shape=%s device=%s grad_none=%s",
                tag,
                idx + 1,
                total,
                tuple(p.shape),
                str(p.device),
                grad is None,
            )

        grad_f32 = grad.detach().float() if grad is not None else torch.zeros_like(
            p, dtype=torch.float32, memory_format=torch.preserve_format
        )
        numel = grad_f32.numel()
        if bucket and (bucket_elems + numel > bucket_max_elems):
            flush_bucket()
        bucket.append((idx, p, grad_f32))
        bucket_elems += numel

        if rank == 0 and debug_log_every > 0 and ((idx + 1) % debug_log_every == 0 or idx == total - 1):
            logger.info("%s allreduce queue progress %d/%d elapsed=%.2fs", tag, idx + 1, total, time.time() - t0)

    flush_bucket()


def preprocess_prompts(
    prompts: List[str],
    cfg: AttrDict,
    rng: random.Random,
    tokenizer=None,
    chi_prompt: Optional[str] = None,
) -> List[str]:
    # Prompt normalization/truncation/template logic shared by train and debug runs.
    prep_cfg = cfg.data.get("preprocessing", AttrDict())
    normalize_whitespace = bool(getattr(prep_cfg, "normalize_whitespace", True))
    strip = bool(getattr(prep_cfg, "strip", True))
    remove_double_newlines = bool(getattr(prep_cfg, "remove_double_newlines", True))
    use_chi_prompt = bool(getattr(prep_cfg, "use_chi_prompt", False))
    use_prompt_templates_cfg = getattr(prep_cfg, "use_prompt_templates", None)
    # Backward compatible: if not explicitly set, keep old behavior (follow use_chi_prompt).
    use_prompt_templates = bool(use_chi_prompt if use_prompt_templates_cfg is None else use_prompt_templates_cfg)
    max_prompt_tokens = getattr(prep_cfg, "max_prompt_tokens", None)

    templates = cfg.data.get("prompt_templates", []) if use_prompt_templates else []
    motion_score = int(cfg.data.get("motion_score", 10))

    processed = []
    for prompt in prompts:
        text = normalize_prompt(prompt, normalize_whitespace, strip, remove_double_newlines)
        if max_prompt_tokens and tokenizer is not None:
            text = truncate_prompt(text, tokenizer, max_prompt_tokens)
        if templates:
            text = apply_template(text, templates, motion_score, rng)
        if use_chi_prompt and chi_prompt:
            text = f"{chi_prompt}{text}"
        processed.append(text)
    return processed


def safe_barrier(local_rank: int) -> None:
    if not dist.is_initialized():
        return
    try:
        dist.barrier(device_ids=[local_rank])
    except TypeError:
        dist.barrier()


def deepspeed_zero_grad_compat(engine: torch.nn.Module) -> None:
    # DeepSpeed API differs by version; newer versions may not accept set_to_none.
    try:
        engine.zero_grad(set_to_none=True)
    except TypeError:
        engine.zero_grad()


def build_student_checkpoint_state(student: torch.nn.Module, student_module: SanaPromptBridge) -> Dict[str, Any]:
    # Collect trainable student-side states (bridge blocks, gate, optional LoRA).
    lora_state = extract_lora_state_dict(student_module.smolvlm2_model)
    trainable_text_state = extract_trainable_smolvlm2_state_dict(student_module.smolvlm2_model)
    if isinstance(student, FSDP):
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, full_cfg):
            full_state = student.state_dict()
        student_state: Dict[str, Any] = {}
        for prefix in ["adapter", "adapter_output_norm", "resampler", "projector", "smolvlm2_vision_head"]:
            key_prefix = f"{prefix}."
            sub_state = {
                k[len(key_prefix):]: v for k, v in full_state.items() if k.startswith(key_prefix)
            }
            if sub_state:
                student_state[prefix] = sub_state
        gate = full_state.get("adapter_output_gate")
        if gate is not None:
            student_state["adapter_output_gate"] = gate.detach().cpu()
        if trainable_text_state:
            student_state["smolvlm2_text_trainable"] = trainable_text_state
        if lora_state:
            student_state["smolvlm2_lora"] = lora_state
        return student_state

    student_state: Dict[str, Any] = {
        "adapter_output_gate": student_module.adapter_output_gate.detach().cpu(),
    }
    if hasattr(student_module, "adapter"):
        student_state["adapter"] = student_module.adapter.state_dict()
    if hasattr(student_module, "adapter_output_norm"):
        student_state["adapter_output_norm"] = student_module.adapter_output_norm.state_dict()
    if hasattr(student_module, "resampler"):
        student_state["resampler"] = student_module.resampler.state_dict()
    if getattr(student_module, "projector", None) is not None:
        student_state["projector"] = student_module.projector.state_dict()
    if getattr(student_module, "smolvlm2_vision_head", None) is not None:
        student_state["smolvlm2_vision_head"] = student_module.smolvlm2_vision_head.state_dict()
    if trainable_text_state:
        student_state["smolvlm2_text_trainable"] = trainable_text_state
    if lora_state:
        student_state["smolvlm2_lora"] = lora_state
    return student_state


def train_teacher_free(cfg: AttrDict, args: argparse.Namespace):
    # --------------------------
    # 1) Distributed/bootstrap
    # --------------------------
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rank, world_size, local_rank = init_distributed(args.max_gpus)
    is_main = rank == 0

    set_seed(cfg.run.seed + rank)
    if is_main:
        logger.info("Distributed init: world_size=%s local_rank=%s", world_size, local_rank)

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device or "cuda")

    precision = (args.precision or cfg.run.precision or "bf16").lower()
    dtype = torch.bfloat16 if precision in ("bf16", "bfloat16") else torch.float32

    output_dir = args.output_dir or cfg.run.output_dir
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        run_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = output_dir

    sana_cfg: SanaVideoConfig = load_sana_config(cfg.sana.config)
    sana_ckpt_dir = cfg.sana.dit_ckpt or cfg.sana.get("ckpt_dir", "omni_ckpts/sana_video_2b_480p")
    auto_download_pretrained = bool(getattr(cfg.run, "auto_download_pretrained", True))
    auto_download_sana = bool(getattr(cfg.run, "auto_download_sana", auto_download_pretrained))
    auto_download_smol = bool(getattr(cfg.run, "auto_download_smolvlm2", auto_download_pretrained))
    text_encoder_cfg = cfg.model.student.text_encoder
    backbone_type = str(text_encoder_cfg.get("backbone_type", "smolvlm2")).lower()
    smol_ckpt_path = str(text_encoder_cfg.ckpt_path)
    if is_main:
        logger.info(
            "Bootstrap stage: loaded config, auto_download_sana=%s auto_download_smol=%s",
            auto_download_sana,
            auto_download_smol,
        )

    if auto_download_sana:
        if is_main:
            logger.info("Bootstrap stage: ensure_sana_assets_available (start)")
        ensure_sana_assets_available(
            sana_cfg=sana_cfg,
            sana_ckpt_dir=sana_ckpt_dir,
            is_main=is_main,
            local_rank=local_rank,
        )
        if is_main:
            logger.info("Bootstrap stage: ensure_sana_assets_available (done)")
    if backbone_type in {"qwen3_vl", "qwen3vl", "qwen"}:
        if not os.path.exists(smol_ckpt_path):
            raise FileNotFoundError(
                f"Qwen3-VL checkpoint path not found: {smol_ckpt_path}. "
                "Download the local model assets before launching training."
            )
        if is_main:
            logger.info("Bootstrap stage: using local Qwen3-VL checkpoint at %s", smol_ckpt_path)
    elif auto_download_smol:
        if is_main:
            logger.info("Bootstrap stage: ensure_smolvlm2_checkpoint_available (start)")
        ensure_smolvlm2_checkpoint_available(
            ckpt_path=smol_ckpt_path,
            is_main=is_main,
            local_rank=local_rank,
        )
        if is_main:
            logger.info("Bootstrap stage: ensure_smolvlm2_checkpoint_available (done)")

    chi_prompt_text = ""
    if bool(getattr(cfg.data.get("preprocessing", AttrDict()), "use_chi_prompt", False)):
        chi_list = getattr(getattr(sana_cfg, "text_encoder", AttrDict()), "chi_prompt", None)
        if isinstance(chi_list, (list, tuple)) and len(chi_list) > 0:
            chi_prompt_text = "\n".join(str(x) for x in chi_list)
            if is_main:
                logger.info("Using SANA chi_prompt prefix (len=%d chars)", len(chi_prompt_text))

    strict_sana_parity_text_path = bool(getattr(cfg.run, "strict_sana_parity_text_path", False))
    strict_sana_use_full_text_window = bool(getattr(cfg.run, "strict_sana_use_full_text_window", False))
    strict_sana_token_select_strategy = str(getattr(cfg.run, "strict_sana_token_select_strategy", "tail") or "tail")
    strict_sana_head_tokens = int(getattr(cfg.run, "strict_sana_head_tokens", 96) or 96)
    strict_sana_tail_tokens = int(getattr(cfg.run, "strict_sana_tail_tokens", 96) or 96)
    strict_fail_fast_mask = bool(getattr(cfg.run, "strict_fail_fast_mask", strict_sana_parity_text_path))
    sana_model_max_length = int(getattr(getattr(sana_cfg, "text_encoder", AttrDict()), "model_max_length", 300) or 300)
    checkpoint_load_path = args.resume_from or getattr(args, "init_from", None)
    checkpoint_load_mode = "resume" if args.resume_from else ("init" if getattr(args, "init_from", None) else None)
    load_log_prefix = "Resume" if checkpoint_load_mode == "resume" else "Init"
    resume_ckpt = None
    resume_student_state = None
    resume_dit_state = None
    resume_step = 0
    resume_micro_step = 0
    preloaded_dit_resume = False
    if checkpoint_load_path:
        if is_main:
            logger.info("%s prewrap: torch.load checkpoint (start) path=%s", load_log_prefix, checkpoint_load_path)
        resume_ckpt = torch.load(checkpoint_load_path, map_location="cpu")
        resume_student_state = resume_ckpt.get("student_state", resume_ckpt)
        resume_dit_state = resume_ckpt.get("dit_trainable_state")
        resume_step = int(resume_ckpt.get("step", 0))
        resume_micro_step = int(resume_ckpt.get("micro_step", 0))
        if is_main:
            logger.info(
                "%s prewrap: torch.load checkpoint (done) step=%d micro_step=%d keys=%s",
                load_log_prefix,
                resume_step,
                resume_micro_step,
                sorted(list(resume_ckpt.keys())),
            )
    if is_main and strict_sana_parity_text_path:
        logger.info(
            "Strict SANA-parity text path enabled: model_max_length=%d fail_fast_mask=%s full_text_window=%s select_strategy=%s head_tokens=%d tail_tokens=%d",
            sana_model_max_length,
            strict_fail_fast_mask,
            strict_sana_use_full_text_window,
            strict_sana_token_select_strategy,
            strict_sana_head_tokens,
            strict_sana_tail_tokens,
        )

    dit_fsdp_cfg = cfg.model.dit.get("fsdp")
    use_fsdp = bool(getattr(sana_cfg, "use_fsdp", False)) if dit_fsdp_cfg is None else bool(dit_fsdp_cfg)
    use_dit_ddp = bool(cfg.model.dit.get("ddp", False))
    use_dit_deepspeed = bool(cfg.model.dit.get("deepspeed", False))
    use_student_ddp = bool(getattr(cfg.run, "student_ddp", False))
    use_student_fsdp = bool(getattr(cfg.run, "student_fsdp", False))
    # FSDP init is more stable if all ranks load identical weights locally.
    # We therefore avoid rank0-only checkpoint loading and disable state sync.
    skip_ckpt_load = False
    sana_train_cfg = getattr(sana_cfg, "train", AttrDict())
    # Match upstream SANA: grad-checkpointing must be wired at model-build time.
    # Fallback to run.gradient_checkpointing for backward compatibility.
    use_sana_model_grad_ckpt = bool(
        getattr(sana_train_cfg, "grad_checkpointing", getattr(cfg.run, "gradient_checkpointing", False))
    )
    sana_gc_step = int(getattr(sana_train_cfg, "gc_step", 1) or 1)
    diffusion_model = load_sana_diffusion_model(
        sana_cfg=sana_cfg,
        sana_ckpt_dir=sana_ckpt_dir,
        device=device,
        dtype=dtype,
        skip_ckpt_load=skip_ckpt_load,
        use_grad_checkpoint=use_sana_model_grad_ckpt,
        grad_checkpoint_step=sana_gc_step,
    )
    diffusion_model.to(device)
    if is_main:
        logger.info("Loaded SANA DiT checkpoint")
        y_embedder = getattr(diffusion_model, "y_embedder", None)
        uncond_prob = getattr(y_embedder, "uncond_prob", None) if y_embedder is not None else None
        if uncond_prob is not None:
            logger.info("SANA caption dropout: y_embedder.uncond_prob=%.4f", float(uncond_prob))
        else:
            logger.warning("Could not read SANA caption dropout: diffusion_model.y_embedder.uncond_prob is unavailable")
    # FSDP with this DiT stack is unstable with flash SDP + gradient checkpointing.
    # Prefer stable defaults for multi-GPU runs unless explicitly disabled in config.
    fsdp_disable_flash_sdp = bool(getattr(cfg.run, "fsdp_disable_flash_sdp", True))
    fsdp_disable_grad_ckpt = bool(getattr(cfg.run, "fsdp_disable_grad_checkpointing", True))
    if use_fsdp and fsdp_disable_flash_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        if is_main:
            logger.info("FSDP safety: disabled flash SDP kernel")
    if getattr(cfg.run, "gradient_checkpointing", False):
        if use_fsdp and fsdp_disable_grad_ckpt:
            if is_main:
                logger.info("FSDP safety: gradient_checkpointing disabled")
        else:
            diffusion_model.gradient_checkpointing = True
    diffusion_model.train()

    dit_lora_cfg = cfg.model.dit.get("lora", AttrDict())
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
        if is_main:
            lora_trainable = sum(p.numel() for n, p in diffusion_model.named_parameters() if ("lora_A" in n or "lora_B" in n))
            logger.info(
                "Applied DiT LoRA: replaced_linear_layers=%d trainable_lora_params=%d",
                replaced,
                lora_trainable,
            )

    dit_train_modules = cfg.model.dit.get("train_modules", [])
    if bool(dit_lora_cfg.get("enable", False)) and (not dit_train_modules or any(m in {"cross_attn", "all", "*"} for m in dit_train_modules)):
        # Keep only LoRA tensors trainable when DiT LoRA mode is enabled.
        dit_train_modules = ["lora_A", "lora_B"]
    dit_params = configure_dit_trainable(diffusion_model, dit_train_modules)
    dit_has_trainable = any(p.requires_grad for p in diffusion_model.parameters())
    if not dit_has_trainable:
        # No DiT updates in this stage: skip distributed wrappers to avoid
        # unnecessary collectives/hangs and keep forward purely local.
        use_fsdp = False
        use_dit_ddp = False
        use_dit_deepspeed = False
    if is_main:
        logger.info(
            "Configured DiT trainable modules: %s (has_trainable=%s, wrap_fsdp=%s, wrap_ddp=%s)",
            dit_train_modules,
            dit_has_trainable,
            use_fsdp,
            use_dit_ddp,
        )
    if bool(getattr(cfg.run, "debug_param_signature", False)):
        log_trainable_signature(
            get_sorted_trainable_named_params(diffusion_model),
            tag="dit_prewrap",
            rank=rank,
            world_size=world_size,
        )
    if resume_dit_state:
        if is_main:
            logger.info(
                "%s prewrap: loading dit_trainable_state into unwrapped DiT (start) keys=%d",
                load_log_prefix,
                len(resume_dit_state),
            )
        missing, unexpected = diffusion_model.load_state_dict(resume_dit_state, strict=False)
        preloaded_dit_resume = True
        if is_main:
            logger.info(
                "%s prewrap: loading dit_trainable_state into unwrapped DiT (done) keys=%d missing=%d unexpected=%d",
                load_log_prefix,
                len(resume_dit_state),
                len(missing),
                len(unexpected),
            )
    if use_dit_deepspeed and world_size > 1:
        if deepspeed is None:
            raise RuntimeError("model.dit.deepspeed=true but deepspeed is not installed in current environment.")
        # Defer wrapping until optimizer init (deepspeed.initialize).
        use_fsdp = False
        use_dit_ddp = False
        if is_main:
            logger.info("Using DeepSpeed for DiT (deferred initialize after optimizer setup)")
    dit_ddp_find_unused = bool(getattr(cfg.run, "dit_ddp_find_unused_parameters", False))
    student_ddp_find_unused = bool(getattr(cfg.run, "student_ddp_find_unused_parameters", False))

    if use_fsdp and world_size > 1:
        sharding_name = str(cfg.model.dit.get("fsdp_sharding_strategy", "shard_grad_op") or "shard_grad_op").lower()
        sharding_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
            "_hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        }
        if sharding_name not in sharding_map:
            raise RuntimeError(
                "Invalid model.dit.fsdp_sharding_strategy=%r; valid values are: %s"
                % (sharding_name, ", ".join(sorted(sharding_map.keys())))
            )
        dit_sharding_strategy = sharding_map[sharding_name]
        # FSDP for DiT: shard trainable params/gradients across ranks.
        if is_main:
            logger.info(
                "Wrapping DiT with FSDP (use_orig_params=True, sync_module_states=False, sharding=%s)",
                sharding_name,
            )
        diffusion_model = FSDP(
            diffusion_model,
            use_orig_params=True,
            device_id=device,
            sync_module_states=False,
            sharding_strategy=dit_sharding_strategy,
            limit_all_gathers=True,
        )
        # After FSDP wrapping, rebuild the param list for the optimizer.
        dit_params = [p for p in diffusion_model.parameters() if p.requires_grad]
        if is_main:
            logger.info("FSDP wrap done")
    elif world_size > 1 and use_dit_ddp:
        logger.info("Rank %s entering DDP(DiT) wrap", rank)
        diffusion_model = DDP(
            diffusion_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=dit_ddp_find_unused,
            broadcast_buffers=False,
        )
        dit_params = [p for p in diffusion_model.parameters() if p.requires_grad]
        logger.info("Rank %s finished DDP(DiT) wrap", rank)
    elif world_size > 1:
        logger.info("Rank %s using manual gradient all-reduce for DiT (FSDP/DDP disabled)", rank)

    logger.info("Rank %s entering build_student", rank)
    student = build_student(
        cfg,
        device,
        dtype,
        strict_sana_parity_text_path=strict_sana_parity_text_path,
        strict_sana_use_full_text_window=strict_sana_use_full_text_window,
        strict_sana_token_select_strategy=strict_sana_token_select_strategy,
        strict_sana_head_tokens=strict_sana_head_tokens,
        strict_sana_tail_tokens=strict_sana_tail_tokens,
        strict_fail_fast_mask=strict_fail_fast_mask,
        sana_model_max_length=sana_model_max_length,
        sana_chi_prompt_text=chi_prompt_text,
    )
    student.to(device)
    if is_main:
        logger.info("Built student model")
    lora_enabled = bool(cfg.model.student.get("lora", {}).get("enable", False))
    if world_size > 1 and use_student_fsdp and lora_enabled:
        # smolvlm2_model is ignored by student FSDP wrap. Keeping LoRA trainable here
        # would make it unsynchronized across ranks, so disable it for correctness.
        disabled = 0
        for name, p in student.smolvlm2_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = False
                disabled += 1
        if is_main:
            logger.warning(
                "LoRA was enabled but student_fsdp=true with ignored smolvlm2_model. "
                "Disabled %d LoRA tensors to keep FSDP training synchronized.",
                disabled,
            )
    # Optional parameter stats (can be slow on some setups).
    if bool(getattr(cfg.run, "log_param_stats", False)):
        total_params = sum(p.numel() for p in student.parameters())
        trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
        tensor_params = sum(1 for _ in student.parameters())
        logging.info(
            "Rank %s student params: tensors=%s total=%s trainable=%s",
            rank,
            tensor_params,
            total_params,
            trainable_params,
        )
        if tensor_params == 0:
            raise RuntimeError(f"Rank {rank} built student with zero parameters; check checkpoint paths or init.")
    if world_size > 1 and use_student_fsdp:
        # FSDP for student bridge while ignoring frozen heavy modules.
        logger.info("Rank %s entering FSDP(student) wrap", rank)
        ignored_modules = []
        if hasattr(student, "smolvlm2_model"):
            ignored_modules.append(student.smolvlm2_model)
        if getattr(student, "smolvlm2_vision_head", None) is not None:
            vision_head_has_trainable = any(p.requires_grad for p in student.smolvlm2_vision_head.parameters())
            if not vision_head_has_trainable:
                ignored_modules.append(student.smolvlm2_vision_head)
        student = FSDP(
            student,
            use_orig_params=True,
            device_id=device,
            sync_module_states=False,
            ignored_modules=ignored_modules,
        )
        logger.info("Rank %s finished FSDP(student) wrap", rank)
    elif world_size > 1 and use_student_ddp:
        logger.info("Rank %s entering DDP(student) wrap", rank)
        student = DDP(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=student_ddp_find_unused,
            broadcast_buffers=False,
        )
        logger.info("Rank %s finished DDP(student) wrap", rank)
    elif world_size > 1:
        logger.info("Rank %s using manual gradient all-reduce for student (DDP disabled)", rank)
    student_module = student.module if isinstance(student, (DDP, FSDP)) else student
    student_tokenizer = student_module._get_tokenizer()
    student_tokenizer_cls = type(student_tokenizer).__name__
    student_tokenizer_mod = type(student_tokenizer).__module__
    student_tokenizer_name = getattr(student_tokenizer, "name_or_path", None)
    if is_main:
        logger.info(
            "Bridge tokenizer: class=%s module=%s name_or_path=%s",
            student_tokenizer_cls,
            student_tokenizer_mod,
            student_tokenizer_name,
        )
    if student_tokenizer_cls == "SimpleTokenizer":
        raise RuntimeError(
            "SimpleTokenizer fallback detected in training. "
            "Please ensure HuggingFace tokenizer cache is available for SmolVLM2 tokenizer_model_id."
        )

    # --------------------------
    # 2) Data + optimizer setup
    # --------------------------
    # Dataset / dataloaders
    logger.info("Rank %s building dataset", rank)
    openvid_cfg = cfg.data.openvid
    expected_latent_t = getattr(openvid_cfg, "expected_latent_t", None)
    if expected_latent_t is not None:
        try:
            expected_latent_t = int(expected_latent_t)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"data.openvid.expected_latent_t must be int, got {expected_latent_t!r}") from exc
        if expected_latent_t < 1:
            raise RuntimeError(f"data.openvid.expected_latent_t must be >=1, got {expected_latent_t}")
    expected_frame_num = getattr(openvid_cfg, "expected_frame_num", None)
    if expected_frame_num is not None:
        try:
            expected_frame_num = int(expected_frame_num)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"data.openvid.expected_frame_num must be int, got {expected_frame_num!r}") from exc
        if expected_frame_num < 1:
            raise RuntimeError(f"data.openvid.expected_frame_num must be >=1, got {expected_frame_num}")
    joint_cfg = cfg.data.get("joint", AttrDict())
    sana_train_cfg_for_joint = getattr(sana_cfg, "train", AttrDict())
    joint_interval = int(
        getattr(joint_cfg, "interval", getattr(sana_train_cfg_for_joint, "joint_training_interval", 0)) or 0
    )
    image_per_video_raw = getattr(joint_cfg, "image_per_video", None)
    image_per_video = None
    if image_per_video_raw is not None:
        try:
            parsed_ratio = int(image_per_video_raw)
        except (TypeError, ValueError):
            parsed_ratio = -1
        if parsed_ratio > 0:
            image_per_video = parsed_ratio
        elif is_main:
            logger.warning(
                "Ignoring invalid joint.image_per_video=%r (must be positive integer)",
                image_per_video_raw,
            )
    joint_use_image_ratio = image_per_video is not None
    joint_enabled = bool(getattr(joint_cfg, "enabled", False)) and (joint_use_image_ratio or (joint_interval > 0))
    video_modality = str(getattr(joint_cfg, "video_modality", "video") or "video").strip().lower()
    image_modality = str(getattr(joint_cfg, "image_modality", "image") or "image").strip().lower()
    video_csv_path = str(getattr(openvid_cfg, "csv_path_video", getattr(openvid_cfg, "csv_path", "")) or "")
    image_csv_path = str(getattr(openvid_cfg, "csv_path_image", video_csv_path) or video_csv_path)
    if not video_csv_path:
        raise RuntimeError("data.openvid.csv_path is empty")

    dataset_video = OpenVidDataset(
        csv_path=video_csv_path,
        video_dir=openvid_cfg.video_dir,
        preprocessed_dir=openvid_cfg.preprocessed_dir,
        use_preprocessed=openvid_cfg.use_preprocessed,
        max_samples=openvid_cfg.get("max_samples"),
        modality_filter=[video_modality] if joint_enabled else None,
    )
    dataset_image = None
    if joint_enabled:
        image_max_samples = getattr(joint_cfg, "image_max_samples", None)
        dataset_image = OpenVidDataset(
            csv_path=image_csv_path,
            video_dir=openvid_cfg.video_dir,
            preprocessed_dir=openvid_cfg.preprocessed_dir,
            use_preprocessed=openvid_cfg.use_preprocessed,
            max_samples=image_max_samples,
            modality_filter=[image_modality],
        )
        if len(dataset_video) == 0 or len(dataset_image) == 0:
            if is_main:
                logger.warning(
                    "Joint 2-dataloader requested but dataset is empty "
                    "(video=%d image=%d). Fallback to single dataloader.",
                    len(dataset_video),
                    len(dataset_image),
                )
            joint_enabled = False
            dataset_image = None

    if is_main:
        if joint_enabled and dataset_image is not None:
            if joint_use_image_ratio:
                logger.info(
                    "Datasets initialized: video=%d (%s), image=%d (%s), image_per_video=%d",
                    len(dataset_video),
                    video_modality,
                    len(dataset_image),
                    image_modality,
                    image_per_video,
                )
            else:
                logger.info(
                    "Datasets initialized: video=%d (%s), image=%d (%s), interval=%d",
                    len(dataset_video),
                    video_modality,
                    len(dataset_image),
                    image_modality,
                    joint_interval,
                )
        else:
            logger.info("Dataset initialized: %s samples", len(dataset_video))
        if expected_latent_t is not None or expected_frame_num is not None:
            logger.info(
                "Dataset temporal contract: expected_latent_t=%s expected_frame_num=%s",
                expected_latent_t,
                expected_frame_num,
            )
    else:
        if joint_enabled and dataset_image is not None:
            logger.info(
                "Rank %s datasets initialized: video=%d image=%d",
                rank,
                len(dataset_video),
                len(dataset_image),
            )
        else:
            logger.info("Rank %s dataset initialized: %s samples", rank, len(dataset_video))

    sampler_video = None
    sampler_image = None
    if world_size > 1:
        sampler_video = DistributedSampler(dataset_video, num_replicas=world_size, rank=rank, shuffle=True)
        if joint_enabled and dataset_image is not None:
            sampler_image = DistributedSampler(dataset_image, num_replicas=world_size, rank=rank, shuffle=True)

    num_workers = args.num_workers if args.num_workers is not None else cfg.run.num_workers
    loader_prefetch_factor = int(getattr(cfg.run, "prefetch_factor", 2) or 2)
    loader_persistent_workers = bool(getattr(cfg.run, "persistent_workers", True))
    dataloader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = max(1, loader_prefetch_factor)
        dataloader_kwargs["persistent_workers"] = loader_persistent_workers

    batch_size_video = int(args.batch_size or cfg.data.batching.batch_size)
    batch_size_image = int(getattr(cfg.data.batching, "batch_size_image", batch_size_video) or batch_size_video)
    shuffle_data = bool(cfg.data.batching.get("shuffle", True))
    drop_last_video = bool(cfg.data.batching.drop_last)
    drop_last_image = bool(getattr(cfg.data.batching, "drop_last_image", drop_last_video))

    dataloader_video = DataLoader(
        dataset_video,
        batch_size=batch_size_video,
        shuffle=(sampler_video is None and shuffle_data),
        sampler=sampler_video,
        num_workers=num_workers,
        collate_fn=openvid_collate_fn,
        drop_last=drop_last_video,
        pin_memory=True,
        timeout=60 if num_workers > 0 else 0,
        **dataloader_kwargs,
    )
    dataloader_image = None
    if joint_enabled and dataset_image is not None:
        dataloader_image = DataLoader(
            dataset_image,
            batch_size=batch_size_image,
            shuffle=(sampler_image is None and shuffle_data),
            sampler=sampler_image,
            num_workers=num_workers,
            collate_fn=openvid_collate_fn,
            drop_last=drop_last_image,
            pin_memory=True,
            timeout=60 if num_workers > 0 else 0,
            **dataloader_kwargs,
        )
    if is_main:
        logger.info(
            "Dataloader ready (num_workers=%s, joint=%s, video_batches=%d, image_batches=%d)",
            num_workers,
            joint_enabled,
            len(dataloader_video),
            (len(dataloader_image) if dataloader_image is not None else 0),
        )

    # Optimizer
    if isinstance(student, FSDP):
        bridge_named_params = []
        bridge_params = [p for p in student.parameters() if p.requires_grad]
    else:
        bridge_named_params = get_sorted_trainable_named_params(student_module)
        bridge_params = [p for _, p in bridge_named_params]
    if isinstance(diffusion_model, FSDP):
        dit_named_params = []
        dit_params = [p for p in diffusion_model.parameters() if p.requires_grad]
    else:
        dit_named_params = get_sorted_trainable_named_params(diffusion_model)
        dit_params = [p for _, p in dit_named_params]
    logger.info(
        "Rank %s trainable bridge params: tensors=%s numel=%s",
        rank,
        len(bridge_params),
        sum(p.numel() for p in bridge_params),
    )
    logger.info(
        "Rank %s trainable DiT params: tensors=%s numel=%s",
        rank,
        len(dit_params),
        sum(p.numel() for p in dit_params),
    )
    for idx, p in enumerate(dit_params):
        if not p.is_cuda:
            raise RuntimeError(
                f"Rank {rank} DiT trainable param idx={idx} is not CUDA: "
                f"device={p.device}, shape={tuple(p.shape)}"
            )
    preview_batch_size = args.batch_size or cfg.data.batching.batch_size
    preview_grad_accum = args.grad_accum_steps or cfg.data.batching.grad_accum_steps

    optimizer = None
    scheduler = None
    bridge_optimizer = None
    bridge_scheduler = None
    dit_optimizer = None
    dit_scheduler = None
    deepspeed_engine = None
    ds_cfg = None
    ds_cfg_dict = None
    ds_zero_stage = None
    ds_dist_init_required = None

    if use_dit_deepspeed and world_size > 1:
        if not dit_params:
            raise RuntimeError("DeepSpeed requested for DiT, but no trainable DiT params found.")
        if bridge_params:
            bridge_optimizer = torch.optim.AdamW(
                [{"params": bridge_params, "lr": cfg.train.lr.bridge}],
                betas=tuple(cfg.train.optimizer.betas),
                eps=cfg.train.optimizer.eps,
                weight_decay=cfg.train.optimizer.weight_decay,
            )
        dit_optimizer = torch.optim.AdamW(
            [{"params": dit_params, "lr": cfg.train.lr.dit}],
            betas=tuple(cfg.train.optimizer.betas),
            eps=cfg.train.optimizer.eps,
            weight_decay=cfg.train.optimizer.weight_decay,
        )
        ds_cfg = str(cfg.model.dit.get("deepspeed_config", "configs/train_config/zero1_config.json"))
        with open(ds_cfg, "r", encoding="utf-8") as f:
            ds_cfg_dict = json.load(f)
        ds_zero_stage = int(cfg.model.dit.get("deepspeed_zero_stage", ds_cfg_dict.get("zero_optimization", {}).get("stage", 1)))
        if "zero_optimization" not in ds_cfg_dict:
            ds_cfg_dict["zero_optimization"] = {}
        ds_cfg_dict["zero_optimization"]["stage"] = ds_zero_stage
        ds_cfg_dict["train_micro_batch_size_per_gpu"] = int(preview_batch_size)
        ds_cfg_dict["gradient_accumulation_steps"] = int(preview_grad_accum)
        ds_cfg_dict["train_batch_size"] = int(preview_batch_size) * int(preview_grad_accum) * int(world_size)
        if str(getattr(cfg.run, "precision", "")).lower() == "bf16":
            ds_cfg_dict["bf16"] = {"enabled": True}
            ds_cfg_dict["fp16"] = {"enabled": False}
        ds_dist_init_required = bool(cfg.model.dit.get("deepspeed_dist_init_required", True))
    else:
        param_groups = []
        if bridge_params:
            param_groups.append({"params": bridge_params, "lr": cfg.train.lr.bridge})
        if dit_params:
            param_groups.append({"params": dit_params, "lr": cfg.train.lr.dit})
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=tuple(cfg.train.optimizer.betas),
            eps=cfg.train.optimizer.eps,
            weight_decay=cfg.train.optimizer.weight_decay,
        )
        if not param_groups:
            raise RuntimeError("No trainable parameters found for optimizer. Check student/DiT trainable config.")

    sana_train_cfg = getattr(sana_cfg, "train", AttrDict())
    lr_schedule = str(getattr(cfg.train, "lr_schedule", getattr(sana_train_cfg, "lr_schedule", "legacy_cosine")) or "legacy_cosine").lower()
    lr_schedule_args = getattr(cfg.train, "lr_schedule_args", getattr(sana_train_cfg, "lr_schedule_args", AttrDict()))
    if isinstance(lr_schedule_args, dict):
        lr_schedule_args = to_attrdict(lr_schedule_args)
    if lr_schedule_args is None:
        lr_schedule_args = AttrDict()
    total_steps_for_sched = int(args.total_steps or cfg.train.total_steps)
    warmup_steps = int(getattr(lr_schedule_args, "num_warmup_steps", getattr(cfg.train.lr, "warmup_steps", 0)) or 0)
    warmup_steps = max(0, warmup_steps)
    min_lr_ratio = float(getattr(lr_schedule_args, "final_lr", getattr(cfg.train.lr, "min_lr_ratio", 0.0)) or 0.0)
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
    cosine_cycles = float(getattr(lr_schedule_args, "num_cycles", 0.5) or 0.5)
    cosine_decay_ratio = float(getattr(lr_schedule_args, "num_decay", 1.0) or 1.0)
    cosine_decay_ratio = max(0.0, min(1.0, cosine_decay_ratio))
    if cosine_decay_ratio <= 0.0:
        cosine_decay_ratio = 1.0
    decay_steps = max(1, int(total_steps_for_sched * cosine_decay_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if lr_schedule == "constant":
            return 1.0
        if lr_schedule == "cosine":
            progress = float(step - warmup_steps) / float(max(1, total_steps_for_sched - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * cosine_cycles * progress)))
        if lr_schedule in {"cosine_decay_to_constant", "legacy_cosine"}:
            if step > decay_steps:
                return min_lr_ratio
            progress = float(step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * cosine_cycles * progress)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        raise RuntimeError(f"Unsupported lr_schedule={lr_schedule}")

    if bridge_optimizer is not None:
        bridge_scheduler = torch.optim.lr_scheduler.LambdaLR(bridge_optimizer, lr_lambda)
    if optimizer is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if dit_optimizer is not None:
        dit_scheduler = torch.optim.lr_scheduler.LambdaLR(dit_optimizer, lr_lambda)

    if use_dit_deepspeed and world_size > 1:
        logger.info(
            "Rank %s entering DeepSpeed initialize (zero_stage=%d mb=%d accum=%d)",
            rank,
            int(ds_zero_stage),
            int(preview_batch_size),
            int(preview_grad_accum),
        )
        deepspeed_engine, _, _, _ = deepspeed.initialize(
            model=diffusion_model,
            model_parameters=dit_params,
            optimizer=dit_optimizer,
            lr_scheduler=dit_scheduler,
            config=ds_cfg_dict,
            dist_init_required=ds_dist_init_required,
        )
        logger.info("Rank %s finished DeepSpeed initialize", rank)
        diffusion_model = deepspeed_engine
        if is_main:
            logger.info(
                "DeepSpeed initialized for DiT with config: %s (zero_stage=%d mb=%d grad_accum=%d global_batch=%d)",
                ds_cfg,
                int(ds_zero_stage),
                int(preview_batch_size),
                int(preview_grad_accum),
                int(ds_cfg_dict["train_batch_size"]),
            )
    if is_main:
        logger.info(
            "LR schedule: mode=%s warmup=%d total_steps=%d min_ratio=%.4f cycles=%.3f decay_ratio=%.3f",
            lr_schedule,
            warmup_steps,
            total_steps_for_sched,
            min_lr_ratio,
            cosine_cycles,
            cosine_decay_ratio,
        )

    # Use the same training objective path as upstream SANA train.py.
    sana_sched_cfg = getattr(sana_cfg, "scheduler", AttrDict())
    train_sampling_steps = int(
        getattr(sana_sched_cfg, "train_sampling_steps", cfg.sana.timesteps.num_train_timesteps)
    )
    noise_schedule = str(getattr(sana_sched_cfg, "noise_schedule", "linear_flow"))
    predict_flow_v = bool(getattr(sana_sched_cfg, "predict_flow_v", True))
    pred_sigma = bool(getattr(sana_sched_cfg, "pred_sigma", False))
    learn_sigma = bool(getattr(sana_sched_cfg, "learn_sigma", True)) and pred_sigma
    flow_shift = float(getattr(sana_sched_cfg, "flow_shift", getattr(cfg.train, "flow_shift", 3.0)))
    weighting_scheme = str(getattr(sana_sched_cfg, "weighting_scheme", "none"))
    weighting_logit_mean = float(getattr(sana_sched_cfg, "logit_mean", 0.0))
    weighting_logit_std = float(getattr(sana_sched_cfg, "logit_std", 1.0))
    weighting_mode_scale = float(getattr(sana_sched_cfg, "mode_scale", 1.29))
    weighting_p_low = getattr(sana_sched_cfg, "p_low", None)
    weighting_p_high = getattr(sana_sched_cfg, "p_high", None)
    if weighting_p_low is not None:
        weighting_p_low = float(weighting_p_low)
    if weighting_p_high is not None:
        weighting_p_high = float(weighting_p_high)
    use_sana_process_timesteps = bool(getattr(cfg.train, "use_sana_process_timesteps", False))
    force_ivjoint_timestep = bool(getattr(cfg.train, "force_ivjoint_timestep", False))
    same_timestep_prob = float(getattr(cfg.train, "same_timestep_prob", 0.0) or 0.0)
    same_timestep_prob = max(0.0, min(1.0, same_timestep_prob))
    chunk_sampling_strategy = str(getattr(cfg.train, "chunk_sampling_strategy", "uniform") or "uniform")
    timestep_weight = bool(getattr(cfg.train, "timestep_weight", False))
    chunk_index = getattr(getattr(sana_cfg, "model", AttrDict()), "chunk_index", None)
    if chunk_index is None:
        chunk_index = getattr(cfg.model.dit, "chunk_index", None)
    if chunk_index is not None:
        if isinstance(chunk_index, (int, float)):
            chunk_index = [0, int(chunk_index)]
        elif isinstance(chunk_index, str):
            text = chunk_index.strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            values = [v.strip() for v in text.split(",") if v.strip()]
            chunk_index = [int(v) for v in values]
        else:
            chunk_index = [int(v) for v in list(chunk_index)]
        if len(chunk_index) == 1:
            chunk_index = [0, int(chunk_index[0])]
        chunk_index = sorted(set(int(v) for v in chunk_index))
        if not chunk_index or chunk_index[0] != 0:
            chunk_index = [0] + chunk_index
        # Chunk-causal training expects frame-aware timestep tensors.
        use_sana_process_timesteps = True
    if force_ivjoint_timestep:
        # Upstream ivjoint baseline: global timestep sampling without chunk/frame-aware process_timesteps path.
        use_sana_process_timesteps = False
        chunk_index = None
    time_sampler = None
    if chunk_index is not None and use_sana_process_timesteps:
        time_sampler = IncrementalTimesteps(
            F=chunk_index,
            T=train_sampling_steps,
            device=device,
            dtype=torch.float64,
        )
    sana_train_diffusion = SanaScheduler(
        str(train_sampling_steps),
        noise_schedule=noise_schedule,
        predict_flow_v=predict_flow_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=bool(getattr(cfg.train, "snr_loss", False)),
        flow_shift=flow_shift,
    )
    if is_main:
        logger.info(
            "SANA objective: train_sampling_steps=%d noise_schedule=%s predict_flow_v=%s flow_shift=%.3f "
            "weighting_scheme=%s mode_scale=%.4f use_process_timesteps=%s force_ivjoint_timestep=%s "
            "chunk_index=%s chunk_sampling=%s time_sampler=%s timestep_weight=%s",
            train_sampling_steps,
            noise_schedule,
            predict_flow_v,
            flow_shift,
            weighting_scheme,
            weighting_mode_scale,
            use_sana_process_timesteps,
            force_ivjoint_timestep,
            chunk_index,
            chunk_sampling_strategy,
            time_sampler is not None,
            timestep_weight,
        )

    rng = random.Random(cfg.run.seed + rank)
    tokenizer = None
    max_prompt_tokens = getattr(cfg.data.get("preprocessing", AttrDict()), "max_prompt_tokens", None)
    if max_prompt_tokens:
        tokenizer = student_module._get_tokenizer()

    grad_accum_steps = args.grad_accum_steps or cfg.data.batching.grad_accum_steps
    batch_size = args.batch_size or cfg.data.batching.batch_size
    log_every = args.log_every or cfg.run.log_every
    save_every = args.save_every or cfg.run.save_every_steps
    total_steps = args.total_steps or cfg.train.total_steps
    sync_debug_every_cfg = getattr(cfg.run, "sync_debug_log_every_params", 0)
    sync_debug_every = int(0 if sync_debug_every_cfg is None else sync_debug_every_cfg)
    trace_each_dit_sync = bool(getattr(cfg.run, "sync_trace_each_dit_param", False))
    sync_bucket_max_elems = int(getattr(cfg.run, "sync_bucket_max_elems", 25_000_000) or 25_000_000)
    manual_sync_backend = str(getattr(cfg.run, "manual_sync_backend", "nccl") or "nccl").lower()
    student_sync = bool(getattr(cfg.run, "student_sync", True))
    dit_sync = bool(getattr(cfg.run, "dit_sync", True))
    require_dit_sharding = bool(getattr(cfg.run, "require_dit_sharding", False))
    use_barrier = bool(getattr(cfg.run, "use_barrier", False))
    gate_min_value = float(cfg.model.student.get("gate_min_value", 0.0))
    student_projector_cfg = cfg.model.student.get("projector", AttrDict())
    student_pre_dit_layernorm = bool(getattr(student_projector_cfg, "pre_dit_layernorm", True))
    gate_loss_cfg = cfg.loss.get("gate", AttrDict())
    gate_loss_enabled = bool(getattr(gate_loss_cfg, "enabled", False))
    gate_loss_weight = float(getattr(gate_loss_cfg, "weight", 0.0))
    gate_loss_target = float(getattr(gate_loss_cfg, "target", 1.0))
    aux_start_step = int(getattr(cfg.loss, "aux_start_step", 0) or 0)
    log_lora_grad = bool(getattr(cfg.run, "log_lora_grad", False))
    debug_probe_every = int(getattr(cfg.run, "debug_probe_every_steps", 0) or 0)
    debug_probe_prompts = list(getattr(cfg.run, "debug_probe_prompts", []) or [])
    cfg_dropout_prob = float(getattr(cfg.run, "cfg_dropout_prob", 0.0) or 0.0)
    cfg_dropout_prob = max(0.0, min(1.0, cfg_dropout_prob))
    cfg_delta_every = int(getattr(cfg.run, "cfg_delta_every_steps", 0) or 0)
    cfg_delta_uncond_prompt = str(getattr(cfg.run, "cfg_delta_uncond_prompt", "") or "")
    cfg_uncond_detach_bridge = bool(getattr(cfg.run, "cfg_uncond_detach_bridge", True))
    cfg_uncond_fixed_eval = bool(getattr(cfg.run, "cfg_uncond_fixed_eval", True))
    # TODO(cleanup): temporary conditioning-collapse diagnostics.
    # Keep while triaging prompt-following collapse; remove once training is stable.
    conditioning_diag_every = int(getattr(cfg.run, "conditioning_diag_every_steps", 0) or 0)
    conditioning_diag_shuffle = bool(getattr(cfg.run, "conditioning_diag_shuffle", True))
    conditioning_diag_uncond = bool(getattr(cfg.run, "conditioning_diag_uncond", True))
    conditioning_diag_grad = bool(getattr(cfg.run, "conditioning_diag_grad", True))
    conditioning_shape_log = bool(getattr(cfg.run, "conditioning_shape_log", True))
    conditioning_pred_probe_every = int(getattr(cfg.run, "conditioning_pred_probe_every_steps", 0) or 0)
    conditioning_pred_probe_prompts = list(getattr(cfg.run, "conditioning_pred_probe_prompts", []) or [])
    if len(conditioning_pred_probe_prompts) < 2:
        conditioning_pred_probe_prompts = list(debug_probe_prompts[:2]) if len(debug_probe_prompts) >= 2 else []
    probe_only = bool(getattr(cfg.run, "probe_only", False))
    probe_only_steps = int(getattr(cfg.run, "probe_only_steps", 1) or 1)
    distill_cfg = cfg.loss.get("distill", AttrDict())
    distill_enabled = bool(getattr(distill_cfg, "enabled", False))
    distill_teacher_store = None
    distill_precomputed = distill_cfg.get("precomputed", AttrDict())
    distill_precomputed_dir = str(getattr(distill_precomputed, "dir", "") or "")
    distill_has_precomputed = bool(distill_precomputed_dir)
    distill_preload = bool(getattr(distill_precomputed, "preload", True))
    distill_max_cached_shards = int(getattr(distill_precomputed, "max_cached_shards", 2) or 2)
    distill_w_mse = float(getattr(distill_cfg, "token_mse_weight", 0.0) or 0.0)
    distill_w_cos = float(getattr(distill_cfg, "token_cos_weight", 0.0) or 0.0)
    distill_w_pool = float(getattr(distill_cfg, "pooled_cos_weight", 0.0) or 0.0)
    distill_w_contrastive = float(getattr(distill_cfg, "contrastive_weight", 0.0) or 0.0)
    distill_contrastive_temp = float(getattr(distill_cfg, "contrastive_temperature", 0.07) or 0.07)
    distill_hidden0_geom_weight = float(getattr(distill_cfg, "hidden0_geom_weight", 0.0) or 0.0)
    distill_hidden0_geom_layernorm = bool(getattr(distill_cfg, "hidden0_geom_layernorm", True))
    distill_use_mask = bool(getattr(distill_cfg, "use_attention_mask", True))
    distill_target_space = canonicalize_distill_target_space(
        str(getattr(distill_cfg, "target_space", "sana_post_ynorm") or "sana_post_ynorm")
    )
    distill_freeze_sana_conditioner = bool(getattr(distill_cfg, "freeze_sana_conditioner", True))
    distill_every_steps = int(getattr(distill_cfg, "every_steps", 1) or 1)
    if distill_every_steps < 1:
        distill_every_steps = 1
    distill_skip_missing = bool(getattr(distill_cfg, "skip_missing", False))
    distill_missing_warn_every = int(getattr(distill_cfg, "missing_warn_every", 200) or 200)
    distill_online_fallback = bool(getattr(distill_cfg, "online_fallback_on_missing", False))
    distill_online_use_chi_prompt_cfg = getattr(distill_cfg, "online_use_chi_prompt", None)
    if distill_online_use_chi_prompt_cfg is None:
        distill_online_use_chi_prompt = False
    else:
        distill_online_use_chi_prompt = bool(distill_online_use_chi_prompt_cfg)
    distill_online_teacher: Optional[SanaTextEncoder] = None
    distill_online_fallback_count = 0
    distill_missing_count = 0
    functional_cfg = cfg.loss.get("functional", AttrDict())
    functional_enabled = bool(getattr(functional_cfg, "enabled", False))
    functional_pred_mse_weight = float(getattr(functional_cfg, "pred_mse_weight", 0.0) or 0.0)
    functional_pred_cos_weight = float(getattr(functional_cfg, "pred_cos_weight", 0.0) or 0.0)
    functional_every_steps = int(getattr(functional_cfg, "every_steps", 1) or 1)
    if functional_every_steps < 1:
        functional_every_steps = 1
    if functional_enabled and functional_pred_mse_weight <= 0.0 and functional_pred_cos_weight <= 0.0:
        if is_main:
            logger.warning(
                "functional distill disabled because pred_mse_weight and pred_cos_weight are both zero."
            )
        functional_enabled = False
    semantic_cfg = cfg.loss.get("semantic_probe", AttrDict())
    semantic_enabled = bool(getattr(semantic_cfg, "enabled", False))
    semantic_weight = float(getattr(semantic_cfg, "weight", 0.0) or 0.0)
    semantic_var_weight = float(getattr(semantic_cfg, "var_weight", 1.0) or 0.0)
    semantic_cov_weight = float(getattr(semantic_cfg, "cov_weight", 1.0) or 0.0)
    semantic_geom_weight = float(getattr(semantic_cfg, "geom_weight", 0.0) or 0.0)
    semantic_geom_source = str(getattr(semantic_cfg, "geom_source", "raw") or "raw").strip().lower()
    semantic_target_std = float(getattr(semantic_cfg, "target_std", 1.0) or 1.0)
    semantic_every_steps = int(getattr(semantic_cfg, "every_steps", 1) or 1)
    semantic_max_prompts = int(getattr(semantic_cfg, "max_prompts", 0) or 0)
    semantic_prompts = list(getattr(semantic_cfg, "prompts", []) or [])
    if not semantic_prompts:
        semantic_prompts = list(debug_probe_prompts)
    semantic_prompts = preprocess_prompts(
        semantic_prompts,
        cfg,
        random.Random(cfg.run.seed),
        tokenizer=tokenizer,
        chi_prompt=chi_prompt_text,
    )
    if semantic_max_prompts > 0:
        semantic_prompts = semantic_prompts[:semantic_max_prompts]
    if semantic_enabled and (semantic_weight <= 0.0 or len(semantic_prompts) < 2):
        if is_main:
            logger.warning(
                "semantic_probe disabled due to invalid settings (weight=%.4f prompts=%d)",
                semantic_weight,
                len(semantic_prompts),
            )
        semantic_enabled = False
    if semantic_geom_source not in {"raw", "teacher"}:
        if is_main:
            logger.warning(
                "semantic_probe.geom_source=%r is invalid; falling back to 'raw'.",
                semantic_geom_source,
            )
        semantic_geom_source = "raw"
    if semantic_geom_weight > 0.0 and (not hasattr(student_module, "encode_prompts")):
        if is_main:
            logger.warning("semantic_probe.geom_weight>0 but student_module has no encode_prompts(); geom term disabled.")
        semantic_geom_weight = 0.0
    if semantic_geom_weight > 0.0 and semantic_geom_source == "teacher" and not distill_enabled:
        if is_main:
            logger.warning("semantic_probe.geom_source=teacher requires distill.enabled=true; geom term disabled.")
        semantic_geom_weight = 0.0
    if distill_enabled:
        if distill_has_precomputed:
            distill_teacher_store = PrecomputedTeacherStore(
                distill_precomputed_dir,
                preload=distill_preload,
                max_cached_shards=distill_max_cached_shards,
            )
        elif not distill_online_fallback:
            raise RuntimeError(
                "distill.enabled=true but distill.precomputed.dir is empty and "
                "distill.online_fallback_on_missing=false. Provide a precomputed dir or enable online fallback."
            )
        if distill_online_fallback:
            distill_online_teacher = SanaTextEncoder(sana_cfg, device=device, dtype=dtype)
            distill_online_teacher.eval().requires_grad_(False)
        if is_main:
            distill_mode = "precomputed+online_fallback" if distill_has_precomputed and distill_online_fallback else (
                "precomputed_only" if distill_has_precomputed else "online_only"
            )
            logger.info(
                "Distill enabled: mode=%s precomputed_dir=%s target_space=%s freeze_sana_conditioner=%s "
                "token_mse=%.3f token_cos=%.3f pooled_cos=%.3f every_steps=%d skip_missing=%s "
                "preload=%s cache=%d online_fallback=%s online_use_chi=%s",
                distill_mode,
                distill_precomputed_dir or "<none>",
                distill_target_space,
                distill_freeze_sana_conditioner,
                distill_w_mse,
                distill_w_cos,
                distill_w_pool,
                distill_every_steps,
                distill_skip_missing,
                distill_preload,
                distill_max_cached_shards,
                distill_online_fallback,
                distill_online_use_chi_prompt,
            )
    if functional_enabled and is_main:
        logger.info(
            "Functional distill enabled: pred_mse=%.3f pred_cos=%.3f every_steps=%d",
            functional_pred_mse_weight,
            functional_pred_cos_weight,
            functional_every_steps,
        )
    if is_main and aux_start_step > 0:
        logger.info("Aux losses will be enabled from update_step >= %d", aux_start_step)

    fixed_uncond_embed = None
    fixed_uncond_mask = None
    if cfg_uncond_detach_bridge:
        was_training = student_module.training
        try:
            if cfg_uncond_fixed_eval:
                student_module.eval()
            with torch.no_grad():
                fixed_uncond_embed, fixed_uncond_mask = student_module([cfg_delta_uncond_prompt], return_mask=True)
            fixed_uncond_embed = fixed_uncond_embed.detach().float()
            if fixed_uncond_mask is not None:
                fixed_uncond_mask = fixed_uncond_mask.detach().long()
            if is_main:
                fixed_len = int(fixed_uncond_embed.shape[1]) if fixed_uncond_embed is not None else 0
                fixed_tok = (
                    float(fixed_uncond_mask.sum().item())
                    if fixed_uncond_mask is not None
                    else float(fixed_len)
                )
                logger.info(
                    "CFG uncond bridge bypass enabled: fixed_eval=%s prompt=%r fixed_len=%d fixed_tok=%.1f",
                    cfg_uncond_fixed_eval,
                    cfg_delta_uncond_prompt,
                    fixed_len,
                    fixed_tok,
                )
        except Exception as exc:
            if is_main:
                logger.warning("Failed to build fixed uncond context; fallback to trainable uncond path. err=%s", exc)
            cfg_uncond_detach_bridge = False
            fixed_uncond_embed = None
            fixed_uncond_mask = None
        finally:
            if was_training:
                student_module.train()

    if world_size > 1:
        manual_sync_group = None
        if manual_sync_backend == "gloo":
            manual_sync_group = dist.new_group(backend="gloo")
            if is_main:
                logger.info("Manual gradient sync backend: gloo")
        elif is_main:
            logger.info("Manual gradient sync backend: nccl")

        dit_is_wrapped = bool(use_fsdp or use_dit_ddp or use_dit_deepspeed)
        allow_manual_dit_sync = bool(getattr(cfg.run, "allow_manual_dit_sync", False))
        if not dit_is_wrapped:
            msg = (
                "DiT is not wrapped by FSDP/DDP/DeepSpeed in multi-GPU mode "
                "(model.dit.fsdp/model.dit.ddp/model.dit.deepspeed are effectively disabled). "
                "This would fallback to manual DiT gradient all-reduce, which is memory-heavy and easy to OOM."
            )
            # Default behavior is fail-fast: require explicit override for manual DiT sync fallback.
            if require_dit_sharding or (not allow_manual_dit_sync):
                raise RuntimeError(
                    f"{msg} Set model.dit.fsdp=true (recommended), or explicitly set "
                    "run.allow_manual_dit_sync=true only for debug."
                )
            if is_main:
                logger.warning(
                    "%s Manual DiT sync fallback is explicitly allowed by run.allow_manual_dit_sync=true.",
                    msg,
                )

        manual_sync_mode = (
            (not use_fsdp)
            and (not use_dit_ddp)
            and (not use_dit_deepspeed)
            and (not use_student_ddp)
            and (not use_student_fsdp)
        )
        if manual_sync_mode and (not student_sync) and (not dit_sync):
            raise RuntimeError(
                "Invalid multi-GPU config: manual sync mode is active but both "
                "run.student_sync and run.dit_sync are false. Enable DDP/FSDP, "
                "or turn on gradient sync to avoid divergent per-rank training."
            )
        if (
            is_main
            and use_dit_deepspeed
            and (not use_student_ddp)
            and (not use_student_fsdp)
            and student_sync
            and manual_sync_backend == "gloo"
        ):
            logger.warning(
                "Student gradient sync is using GLOO with DeepSpeed DiT; this is often slow. "
                "For better throughput, prefer run.student_ddp=true or run.manual_sync_backend=nccl."
            )

    if is_main:
        logger.info("Trainable params bridge: %.2fM", sum(p.numel() for p in bridge_params) / 1e6)
        logger.info("Trainable params DiT: %.2fM", sum(p.numel() for p in dit_params) / 1e6)
        logger.info(
            "CFG settings: dropout_prob=%.3f delta_every_steps=%d uncond_prompt=%r",
            cfg_dropout_prob,
            cfg_delta_every,
            cfg_delta_uncond_prompt,
        )
        logger.info(
            "CFG uncond detach settings: enabled=%s fixed_eval=%s",
            cfg_uncond_detach_bridge,
            cfg_uncond_fixed_eval,
        )
        if conditioning_diag_every > 0:
            logger.info(
                "Conditioning diagnostics: every=%d shuffle=%s uncond=%s grad=%s",
                conditioning_diag_every,
                conditioning_diag_shuffle,
                conditioning_diag_uncond,
                conditioning_diag_grad,
            )
        if semantic_enabled:
            logger.info(
                "Semantic anti-collapse enabled: weight=%.4f var=%.3f cov=%.3f geom=%.3f source=%s every=%d prompts=%d target_std=%.2f",
                semantic_weight,
                semantic_var_weight,
                semantic_cov_weight,
                semantic_geom_weight,
                semantic_geom_source,
                semantic_every_steps,
                len(semantic_prompts),
                semantic_target_std,
            )
        if distill_enabled:
            logger.info(
                "Distill settings: target_space=%s freeze_sana_conditioner=%s pre_dit_layernorm=%s weights(mse=%.3f cos=%.3f pool=%.3f nce=%.3f h0geom=%.3f)",
                distill_target_space,
                distill_freeze_sana_conditioner,
                student_pre_dit_layernorm,
                distill_w_mse,
                distill_w_cos,
                distill_w_pool,
                distill_w_contrastive,
                distill_hidden0_geom_weight,
            )
        if functional_enabled:
            logger.info(
                "Functional distill enabled: pred_mse=%.4f pred_cos=%.4f every=%d",
                functional_pred_mse_weight,
                functional_pred_cos_weight,
                functional_every_steps,
            )

    student_lora_cfg = cfg.model.student.get("lora", AttrDict())
    dit_lora_cfg = cfg.model.dit.get("lora", AttrDict())
    projector_cfg = student_projector_cfg
    prep_cfg = cfg.data.get("preprocessing", AttrDict())
    use_chi_prompt_cfg = bool(getattr(prep_cfg, "use_chi_prompt", False))
    use_prompt_templates_cfg = getattr(prep_cfg, "use_prompt_templates", None)
    if use_prompt_templates_cfg is None:
        use_prompt_templates_cfg = use_chi_prompt_cfg
    prompt_templates_cfg = cfg.data.get("prompt_templates", []) or []
    # Persist the exact train-time conditioning knobs into checkpoint so inference
    # can reconstruct projector/LoRA/flow-shift without manual guesswork.
    infer_hints = {
        "projector_type": str(getattr(projector_cfg, "type", "legacy")),
        "mcp_hidden_dim": int(getattr(projector_cfg, "mcp_hidden_dim", 512) or 512),
        "mcp_num_fuse_layers": int(getattr(projector_cfg, "mcp_num_fuse_layers", 2) or 2),
        "mcp_use_refine": bool(getattr(projector_cfg, "mcp_use_refine", False)),
        "mcp_refine_kernel_size": int(getattr(projector_cfg, "mcp_refine_kernel_size", 3) or 3),
        "mcp_lexical_bottleneck_dim": int(getattr(projector_cfg, "mcp_lexical_bottleneck_dim", 256) or 256),
        "mcp_lexical_gate_init": float(getattr(projector_cfg, "mcp_lexical_gate_init", 0.05) or 0.05),
        "mcp_pre_dit_layernorm": bool(getattr(projector_cfg, "pre_dit_layernorm", True)),
        "student_lora_enable": bool(getattr(student_lora_cfg, "enable", False)),
        "student_lora_r": int(getattr(student_lora_cfg, "r", 0) or 0),
        "student_lora_alpha": int(getattr(student_lora_cfg, "alpha", 0) or 0),
        "student_text_train_top_layers": int(getattr(cfg.model.student.text_encoder, "train_top_layers", 0) or 0),
        "student_text_train_final_norm": bool(getattr(cfg.model.student.text_encoder, "train_final_norm", False)),
        "dit_lora_enable": bool(getattr(dit_lora_cfg, "enable", False)),
        "dit_lora_r": int(getattr(dit_lora_cfg, "r", 0) or 0),
        "dit_lora_alpha": int(getattr(dit_lora_cfg, "alpha", 0) or 0),
        "train_flow_shift": float(flow_shift),
        "inference_flow_shift": float(getattr(sana_sched_cfg, "inference_flow_shift", flow_shift)),
        "train_vis_sampler": str(getattr(sana_sched_cfg, "vis_sampler", "")),
        "train_use_chi_prompt": use_chi_prompt_cfg,
        "train_use_prompt_templates": bool(use_prompt_templates_cfg),
        "train_motion_score": int(cfg.data.get("motion_score", 10)),
        "train_prompt_templates": [str(t) for t in list(prompt_templates_cfg)],
        "strict_sana_parity_text_path": bool(strict_sana_parity_text_path),
        "strict_sana_use_full_text_window": bool(getattr(cfg.run, "strict_sana_use_full_text_window", False)),
        "strict_sana_token_select_strategy": str(getattr(cfg.run, "strict_sana_token_select_strategy", "tail") or "tail"),
        "strict_sana_head_tokens": int(getattr(cfg.run, "strict_sana_head_tokens", 96) or 96),
        "strict_sana_tail_tokens": int(getattr(cfg.run, "strict_sana_tail_tokens", 96) or 96),
        "strict_fail_fast_mask": bool(strict_fail_fast_mask),
        "sana_model_max_length": int(sana_model_max_length),
        "train_student_max_length": int(getattr(cfg.model.student.text_encoder, "max_length", 0) or 0),
        "train_chunk_index": (list(chunk_index) if chunk_index is not None else None),
        "train_chunk_sampling_strategy": str(chunk_sampling_strategy),
        "train_same_timestep_prob": float(same_timestep_prob),
        "train_use_process_timesteps": bool(use_sana_process_timesteps),
        "train_sana_config": str(cfg.sana.config),
        "train_expected_latent_t": (int(expected_latent_t) if expected_latent_t is not None else None),
        "train_expected_frame_num": (int(expected_frame_num) if expected_frame_num is not None else None),
        "train_latent_t": None,
        "train_effective_latent_t": None,
        "train_frame_num": None,
        "train_joint_enabled": bool(joint_enabled),
        "train_joint_interval": int(joint_interval) if joint_enabled else 0,
        "train_joint_image_per_video": int(image_per_video) if (joint_enabled and joint_use_image_ratio) else 0,
        "train_joint_video_modality": str(video_modality),
        "train_joint_image_modality": str(image_modality),
    }

    if world_size > 1 and use_barrier:
        safe_barrier(local_rank)
    if checkpoint_load_path:
        # Initialize model weights from checkpoint; true resume additionally restores step counters.
        ckpt = resume_ckpt
        state = resume_student_state if resume_student_state is not None else ckpt.get("student_state", ckpt)
        if is_main:
            logger.info("%s stage: loading student_state (start)", load_log_prefix)
        if isinstance(student, FSDP):
            with FSDP.summon_full_params(student, writeback=True, recurse=True):
                if "adapter" in state and hasattr(student_module, "adapter"):
                    student_module.adapter.load_state_dict(state["adapter"], strict=False)
                if "adapter_output_norm" in state and hasattr(student_module, "adapter_output_norm"):
                    student_module.adapter_output_norm.load_state_dict(state["adapter_output_norm"], strict=False)
                if "adapter_output_gate" in state:
                    student_module.adapter_output_gate.data.copy_(
                        state["adapter_output_gate"].to(student_module.adapter_output_gate.device)
                    )
                if "resampler" in state and hasattr(student_module, "resampler"):
                    student_module.resampler.load_state_dict(state["resampler"], strict=False)
                if "projector" in state and getattr(student_module, "projector", None) is not None:
                    student_module.projector.load_state_dict(state["projector"], strict=False)
                if "smolvlm2_vision_head" in state and getattr(student_module, "smolvlm2_vision_head", None) is not None:
                    student_module.smolvlm2_vision_head.load_state_dict(state["smolvlm2_vision_head"], strict=False)
                if "smolvlm2_text_trainable" in state:
                    load_trainable_smolvlm2_state_dict(
                        student_module.smolvlm2_model,
                        state["smolvlm2_text_trainable"],
                        is_main=is_main,
                    )
                if "smolvlm2_lora" in state:
                    load_lora_state_dict(student_module.smolvlm2_model, state["smolvlm2_lora"], is_main=is_main)
        else:
            if "adapter" in state and hasattr(student_module, "adapter"):
                student_module.adapter.load_state_dict(state["adapter"], strict=False)
            if "adapter_output_norm" in state and hasattr(student_module, "adapter_output_norm"):
                student_module.adapter_output_norm.load_state_dict(state["adapter_output_norm"], strict=False)
            if "adapter_output_gate" in state:
                student_module.adapter_output_gate.data.copy_(
                    state["adapter_output_gate"].to(student_module.adapter_output_gate.device)
                )
            if "resampler" in state and hasattr(student_module, "resampler"):
                student_module.resampler.load_state_dict(state["resampler"], strict=False)
            if "projector" in state and getattr(student_module, "projector", None) is not None:
                student_module.projector.load_state_dict(state["projector"], strict=False)
            if "smolvlm2_vision_head" in state and getattr(student_module, "smolvlm2_vision_head", None) is not None:
                student_module.smolvlm2_vision_head.load_state_dict(state["smolvlm2_vision_head"], strict=False)
            if "smolvlm2_text_trainable" in state:
                load_trainable_smolvlm2_state_dict(
                    student_module.smolvlm2_model,
                    state["smolvlm2_text_trainable"],
                    is_main=is_main,
                )
            if "smolvlm2_lora" in state:
                load_lora_state_dict(student_module.smolvlm2_model, state["smolvlm2_lora"], is_main=is_main)
        if is_main:
            logger.info("%s stage: loading student_state (done)", load_log_prefix)

        dit_state = None if preloaded_dit_resume else ckpt.get("dit_trainable_state")
        if dit_state:
            if is_main:
                logger.info("%s stage: loading dit_trainable_state (start) keys=%d", load_log_prefix, len(dit_state))
            if isinstance(diffusion_model, FSDP):
                with FSDP.summon_full_params(diffusion_model, writeback=True, recurse=True):
                    dit_module = diffusion_model.module if hasattr(diffusion_model, "module") else diffusion_model
                    missing, unexpected = dit_module.load_state_dict(dit_state, strict=False)
            else:
                dit_module = diffusion_model.module if hasattr(diffusion_model, "module") else diffusion_model
                missing, unexpected = dit_module.load_state_dict(dit_state, strict=False)
            if is_main:
                logger.info(
                    "Loaded dit_trainable_state: keys=%d missing=%d unexpected=%d",
                    len(dit_state),
                    len(missing),
                    len(unexpected),
                )
                logger.info("%s stage: loading dit_trainable_state (done)", load_log_prefix)

        if checkpoint_load_mode == "resume":
            sharded_resume_mode = isinstance(student, FSDP) or isinstance(diffusion_model, FSDP) or bool(use_dit_deepspeed)
            skip_resume_optim_state = bool(
                getattr(args, "resume_skip_optimizer_state", False)
                or getattr(cfg.run, "resume_skip_optimizer_state", False)
            )
            if sharded_resume_mode or skip_resume_optim_state:
                if is_main:
                    if sharded_resume_mode:
                        logger.warning(
                            "Sharded resume mode: skipping optimizer/scheduler state load due sharded "
                            "state incompatibility. Model weights and step counters are restored."
                        )
                    else:
                        logger.warning(
                            "Resume configured to skip optimizer/scheduler state load "
                            "(resume_skip_optimizer_state=True)."
                        )
            else:
                if optimizer is not None and "optimizer" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer"])
                    except (ValueError, RuntimeError) as exc:
                        if is_main:
                            ckpt_groups = len(ckpt.get("optimizer", {}).get("param_groups", []))
                            cur_groups = len(getattr(optimizer, "param_groups", []))
                            logger.warning(
                                "Skipping optimizer state load (likely trainable-params changed): %s "
                                "(ckpt_groups=%d current_groups=%d)",
                                str(exc),
                                ckpt_groups,
                                cur_groups,
                            )
                if scheduler is not None and "scheduler" in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt["scheduler"])
                    except (ValueError, RuntimeError) as exc:
                        if is_main:
                            logger.warning(
                                "Skipping scheduler state load (resume mismatch): %s",
                                str(exc),
                            )

            resume_step = int(ckpt.get("step", resume_step))
            resume_micro_step = int(ckpt.get("micro_step", resume_micro_step))
            if is_main:
                logger.info("Resumed from %s at step=%d", checkpoint_load_path, resume_step)
        else:
            resume_step = 0
            resume_micro_step = 0
            if is_main:
                logger.info("Initialized model weights from %s (starting fresh at step=0)", checkpoint_load_path)

    if bridge_optimizer is not None:
        bridge_optimizer.zero_grad(set_to_none=True)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if deepspeed_engine is not None:
        deepspeed_zero_grad_compat(deepspeed_engine)
    micro_step = resume_micro_step
    update_step = resume_step
    probe_only_collected = 0
    video_data_iter = iter(dataloader_video)
    image_data_iter = iter(dataloader_image) if dataloader_image is not None else None
    video_micro_count = 0
    image_micro_count = 0
    observed_train_latent_t = None
    observed_train_effective_latent_t = None
    observed_train_frame_num = None

    def get_fixed_uncond_batch(
        batch_size_local: int,
        target_len: int,
        target_dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if fixed_uncond_embed is None:
            return None, None
        emb = fixed_uncond_embed.to(device=device, dtype=target_dtype)
        emb = pad_or_trim_tokens(emb, target_len)
        emb = emb.expand(batch_size_local, -1, -1).contiguous()
        if fixed_uncond_mask is None:
            return emb, None
        m = fixed_uncond_mask.to(device=device, dtype=torch.long)
        m = pad_or_trim_token_mask(m, target_len)
        m = m.expand(batch_size_local, -1).contiguous()
        return emb, m

    def _extract_first_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            if isinstance(value, torch.Tensor):
                if value.numel() <= 0:
                    return None
                return int(value.detach().flatten()[0].item())
            if isinstance(value, np.ndarray):
                if value.size <= 0:
                    return None
                return int(np.asarray(value).reshape(-1)[0])
            if isinstance(value, (list, tuple)):
                if len(value) <= 0:
                    return None
                return _extract_first_int(value[0])
            return int(value)
        except Exception:
            return None

    # --------------------------
    # 3) Train loop
    # micro_step: dataloader step
    # update_step: optimizer step
    # --------------------------
    while update_step < total_steps:
        loop_t0 = time.time()
        is_first_logged_iter = bool(micro_step == 0 or micro_step == resume_micro_step)
        if sampler_video is not None:
            sampler_video.set_epoch(micro_step)
        if sampler_image is not None:
            sampler_image.set_epoch(micro_step)

        if joint_enabled and (dataloader_image is not None):
            if joint_use_image_ratio:
                # Ratio schedule: 1 video step followed by N image steps.
                # Example image_per_video=5 => V, I, I, I, I, I, V, ...
                cycle_len = int(image_per_video) + 1
                use_image_step = (update_step % cycle_len) != 0
            else:
                use_image_step = bool(
                    (joint_interval > 0)
                    and (update_step > 0)
                    and (update_step % joint_interval == 0)
                )
        else:
            use_image_step = False
        active_batch_modality = "image" if use_image_step else "video"

        if is_main and is_first_logged_iter:
            logger.info("Entering training loop; waiting for first batch...")
        fetch_t0 = time.time()
        if use_image_step:
            try:
                batch = next(image_data_iter)
            except StopIteration:
                image_data_iter = iter(dataloader_image)
                batch = next(image_data_iter)
            image_micro_count += 1
        else:
            try:
                batch = next(video_data_iter)
            except StopIteration:
                video_data_iter = iter(dataloader_video)
                batch = next(video_data_iter)
            video_micro_count += 1
        fetch_dt = time.time() - fetch_t0
        if is_main and is_first_logged_iter:
            logger.info("First batch fetched in %.2fs (mode=%s)", fetch_dt, active_batch_modality)

        # Prompt path: dataset caption -> optional normalization/template/truncation.
        prompts = preprocess_prompts(batch["prompt"], cfg, rng, tokenizer=tokenizer, chi_prompt=chi_prompt_text)
        prompts_cond = list(prompts)
        dropped_prompts = 0
        dropped_mask_list = [False] * len(prompts_cond)
        if cfg_dropout_prob > 0.0:
            prompts = []
            dropped_mask_list = []
            for p in prompts_cond:
                if rng.random() < cfg_dropout_prob:
                    prompts.append(cfg_delta_uncond_prompt)
                    dropped_mask_list.append(True)
                    dropped_prompts += 1
                else:
                    prompts.append(p)
                    dropped_mask_list.append(False)
        if is_first_logged_iter:
            logger.info("Rank %s before student forward", rank)
        need_student_aux = bool(distill_enabled and distill_hidden0_geom_weight > 0.0)
        student_aux = None
        student_out = student(prompts, return_mask=True, return_aux=need_student_aux)
        if need_student_aux:
            student_embeds_raw, student_prompt_mask, student_aux = student_out
        else:
            student_embeds_raw, student_prompt_mask = student_out
        if student_embeds_raw.dim() == 1:
            student_embeds_raw = student_embeds_raw.unsqueeze(0).unsqueeze(0)
        elif student_embeds_raw.dim() == 2:
            student_embeds_raw = student_embeds_raw.unsqueeze(0)
        elif student_embeds_raw.dim() != 3:
            raise RuntimeError(f"Unexpected student embedding shape: {tuple(student_embeds_raw.shape)}")
        if cfg_uncond_detach_bridge and dropped_prompts > 0:
            drop_mask = torch.tensor(dropped_mask_list, device=student_embeds_raw.device, dtype=torch.bool)
            fixed_emb_raw, fixed_mask_raw = get_fixed_uncond_batch(
                batch_size_local=student_embeds_raw.shape[0],
                target_len=student_embeds_raw.shape[1],
                target_dtype=student_embeds_raw.dtype,
            )
            if fixed_emb_raw is not None:
                student_embeds_raw = torch.where(
                    drop_mask.view(-1, 1, 1),
                    fixed_emb_raw,
                    student_embeds_raw,
                )
                if student_prompt_mask is None:
                    student_prompt_mask = torch.ones(
                        student_embeds_raw.shape[:2], device=student_embeds_raw.device, dtype=torch.long
                    )
                else:
                    student_prompt_mask = student_prompt_mask.to(device=student_embeds_raw.device, dtype=torch.long)
                    if student_prompt_mask.dim() == 1:
                        student_prompt_mask = student_prompt_mask.unsqueeze(0)
                    student_prompt_mask = pad_or_trim_token_mask(
                        student_prompt_mask, student_embeds_raw.shape[1]
                    )
                if fixed_mask_raw is None:
                    fixed_mask_raw = torch.ones_like(student_prompt_mask, dtype=torch.long)
                student_prompt_mask = torch.where(
                    drop_mask.view(-1, 1),
                    fixed_mask_raw.to(device=student_prompt_mask.device, dtype=student_prompt_mask.dtype),
                    student_prompt_mask,
                )
        student_fwd_dt = time.time() - fetch_t0 - fetch_dt
        if is_first_logged_iter:
            logger.info("Rank %s reached student forward", rank)
        if student_pre_dit_layernorm:
            student_embeds_for_dit = F.layer_norm(student_embeds_raw, (student_embeds_raw.shape[-1],))
        else:
            student_embeds_for_dit = student_embeds_raw
        student_embeds_for_dit = student_embeds_for_dit.to(dtype)
        if student_embeds_for_dit.dim() == 1:
            student_embeds_for_dit = student_embeds_for_dit.view(1, 1, -1)
        elif student_embeds_for_dit.dim() == 2:
            student_embeds_for_dit = student_embeds_for_dit.unsqueeze(0)
        elif student_embeds_for_dit.dim() != 3:
            raise RuntimeError(f"Unexpected student_embeds_for_dit shape: {tuple(student_embeds_for_dit.shape)}")
        teacher_embeds = None
        teacher_mask = None
        step_id = int(update_step + 1)
        distill_run_this_step = bool(
            distill_enabled
            and step_id >= aux_start_step
            and (step_id % distill_every_steps == 0)
        )
        functional_run_this_step = bool(
            functional_enabled
            and step_id >= aux_start_step
            and (step_id % functional_every_steps == 0)
        )
        teacher_supervision_run_this_step = bool(distill_run_this_step or functional_run_this_step)
        if teacher_supervision_run_this_step:
            if distill_teacher_store is None:
                if distill_online_teacher is None:
                    raise RuntimeError(
                        "teacher supervision enabled but neither precomputed nor online teacher is available"
                    )
                distill_online_fallback_count += 1
                with torch.no_grad():
                    teacher_out = distill_online_teacher.forward_chi(
                        prompts_cond,
                        use_chi_prompt=distill_online_use_chi_prompt,
                    )
                teacher_embeds = teacher_out["prompt_embeds"].to(
                    device=device,
                    dtype=student_embeds_raw.dtype,
                    non_blocking=True,
                )
                teacher_mask = teacher_out["mask"].to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                teacher_embeds, teacher_mask = pad_or_trim_teacher(
                    teacher_embeds, teacher_mask, student_embeds_raw.shape[1]
                )
            else:
                sample_idx = batch.get("sample_idx")
                if sample_idx is None:
                    raise RuntimeError("distill enabled but batch missing sample_idx")
                if not isinstance(sample_idx, torch.Tensor):
                    sample_idx = torch.tensor(sample_idx, dtype=torch.long)
                try:
                    teacher_embeds_cpu, teacher_mask_cpu = distill_teacher_store.fetch(sample_idx.cpu())
                except KeyError as exc:
                    if not distill_skip_missing and not distill_online_fallback:
                        raise
                    distill_missing_count += 1
                    should_warn = bool(
                        is_main
                        and (
                            distill_missing_count <= 5
                            or (distill_missing_warn_every > 0 and distill_missing_count % distill_missing_warn_every == 0)
                        )
                    )
                    if distill_online_fallback and distill_online_teacher is not None:
                        distill_online_fallback_count += 1
                        if should_warn:
                            logger.warning(
                                "Distill teacher missing for batch sample_idx=%s (missing_count=%d); "
                                "using online teacher fallback (fallback_count=%d). err=%s",
                                sample_idx.detach().cpu().tolist(),
                                distill_missing_count,
                                distill_online_fallback_count,
                                str(exc),
                            )
                        with torch.no_grad():
                            teacher_out = distill_online_teacher.forward_chi(
                                prompts_cond,
                                use_chi_prompt=distill_online_use_chi_prompt,
                            )
                        teacher_embeds = teacher_out["prompt_embeds"].to(
                            device=device,
                            dtype=student_embeds_raw.dtype,
                            non_blocking=True,
                        )
                        teacher_mask = teacher_out["mask"].to(
                            device=device,
                            dtype=torch.long,
                            non_blocking=True,
                        )
                        teacher_embeds, teacher_mask = pad_or_trim_teacher(
                            teacher_embeds, teacher_mask, student_embeds_raw.shape[1]
                        )
                    else:
                        if should_warn:
                            logger.warning(
                                "Distill teacher missing for batch sample_idx=%s (missing_count=%d); "
                                "skipping distill on this batch. err=%s",
                                sample_idx.detach().cpu().tolist(),
                                distill_missing_count,
                                str(exc),
                            )
                        teacher_embeds = None
                        teacher_mask = None
                else:
                    teacher_embeds = teacher_embeds_cpu.to(device=device, dtype=student_embeds_raw.dtype, non_blocking=True)
                    teacher_mask = teacher_mask_cpu.to(device=device, dtype=torch.long, non_blocking=True)
                    teacher_embeds, teacher_mask = pad_or_trim_teacher(
                        teacher_embeds, teacher_mask, student_embeds_raw.shape[1]
                    )

        # Use prompt mask from student tokenizer/bridge (important for variable-length prompts).
        if student_prompt_mask is None:
            if strict_fail_fast_mask:
                raise RuntimeError(
                    "Strict fail-fast mask enabled: student returned None prompt mask. "
                    "Check tokenizer/bridge output parity."
                )
            mask = torch.ones(student_embeds_for_dit.shape[:2], device=device, dtype=torch.long)
        else:
            mask = student_prompt_mask.to(device=device, dtype=torch.long)
            if mask.shape[:2] != student_embeds_for_dit.shape[:2]:
                if strict_fail_fast_mask:
                    raise RuntimeError(
                        "Strict fail-fast mask enabled: prompt mask shape mismatch "
                        f"mask={tuple(mask.shape)} embeds={tuple(student_embeds_for_dit.shape)}"
                    )
                # TODO(cleanup): remove this silent fallback once mask pipeline is stable.
                # Keeping all-ones here can hide conditioning-mask bugs during triage.
                mask = torch.ones(student_embeds_for_dit.shape[:2], device=device, dtype=torch.long)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.dim() == 2:
            # SANA forward expects mask compatible with squeeze(1).squeeze(1) path.
            mask = mask.unsqueeze(1).unsqueeze(1)

        latents = batch.get("latent_feature")
        if latents is None:
            raise RuntimeError("Dataset does not provide latent_feature. Please run extract_openvid_features.py.")
        latents = latents.to(device=device, dtype=dtype)
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        raw_latent_t = int(latents.shape[2])
        if expected_latent_t is not None and raw_latent_t != expected_latent_t:
            sample_idx_val = _extract_first_int(batch.get("sample_idx"))
            raise RuntimeError(
                "Latent T mismatch: expected data.openvid.expected_latent_t="
                f"{expected_latent_t}, got {raw_latent_t} (mode={active_batch_modality}, sample_idx={sample_idx_val})."
            )
        if observed_train_latent_t is None:
            observed_train_latent_t = raw_latent_t
            infer_hints["train_latent_t"] = int(observed_train_latent_t)
            if is_main:
                logger.info("Observed train latent_t=%d (raw dataset)", observed_train_latent_t)
        frame_num_val = _extract_first_int(batch.get("frame_num"))
        if expected_frame_num is not None:
            if frame_num_val is None:
                raise RuntimeError(
                    "Missing frame_num in batch while data.openvid.expected_frame_num is set "
                    f"to {expected_frame_num}."
                )
            if frame_num_val != expected_frame_num:
                sample_idx_val = _extract_first_int(batch.get("sample_idx"))
                raise RuntimeError(
                    "Frame count mismatch: expected data.openvid.expected_frame_num="
                    f"{expected_frame_num}, got {frame_num_val} (mode={active_batch_modality}, sample_idx={sample_idx_val})."
                )
        if frame_num_val is not None and observed_train_frame_num is None:
            observed_train_frame_num = int(frame_num_val)
            infer_hints["train_frame_num"] = int(observed_train_frame_num)
            if is_main:
                logger.info("Observed train frame_num=%d", observed_train_frame_num)
        if is_main and is_first_logged_iter:
            logger.info("Latent shape %s dtype=%s", tuple(latents.shape), latents.dtype)

        # Optional temporal/spatial windowing to reduce activation memory.
        # NOTE: For SANA chunk-causal parity, do not combine train-time latent windowing
        # with chunk-aware/frame-aware timestep sampling. Keep full latent T and use
        # chunk_index + process_timesteps to model temporal chunks.
        window_cfg = cfg.train.get("latent_window", AttrDict())
        win_frames = getattr(window_cfg, "frames", None)
        win_h = getattr(window_cfg, "height", None)
        win_w = getattr(window_cfg, "width", None)
        if win_frames and (chunk_index is not None or use_sana_process_timesteps):
            strict_chunk_parity = bool(getattr(cfg.train, "strict_chunk_parity", True))
            msg = (
                "train.latent_window.frames is enabled while chunk-aware training is active "
                f"(chunk_index={chunk_index}, use_process_timesteps={use_sana_process_timesteps}). "
                "Temporal windowing changes effective latent T and breaks SANA chunk-causal parity."
            )
            if strict_chunk_parity:
                raise RuntimeError(msg)
            if is_main and micro_step == 0:
                logger.warning("%s (strict_chunk_parity=false: continuing anyway)", msg)
        if (win_h or win_w) and (chunk_index is not None or use_sana_process_timesteps):
            # Spatial-only windowing keeps latent T unchanged (chunk/timestep semantics stay temporal-parity),
            # but it still changes spatial AR distribution versus upstream full-resolution training.
            if is_main and micro_step == 0:
                logger.warning(
                    "train.latent_window.{height,width} is enabled with chunk-aware training. "
                    "Temporal chunk parity is preserved (T unchanged), but spatial AR parity is not."
                )
        if win_frames or win_h or win_w:
            _, _, t_total, h_total, w_total = latents.shape
            t_win = int(win_frames) if win_frames else t_total
            h_win = int(win_h) if win_h else h_total
            w_win = int(win_w) if win_w else w_total
            t_win = min(t_win, t_total)
            h_win = min(h_win, h_total)
            w_win = min(w_win, w_total)
            t0 = 0 if t_total == t_win else rng.randint(0, t_total - t_win)
            h0 = 0 if h_total == h_win else rng.randint(0, h_total - h_win)
            w0 = 0 if w_total == w_win else rng.randint(0, w_total - w_win)
            latents = latents[:, :, t0 : t0 + t_win, h0 : h0 + h_win, w0 : w0 + w_win]
            if is_main and micro_step == 0:
                logger.info("Latent window applied: %s", tuple(latents.shape))

        # Optional downsample for smoke tests to avoid OOM on large resolutions.
        smoke_cfg = cfg.train.get("smoke_latent", AttrDict())
        smoke_frames = getattr(smoke_cfg, "frames", None)
        smoke_size = getattr(smoke_cfg, "size", None)
        if smoke_frames or smoke_size:
            # latents: [B, C, T, H, W]
            if smoke_frames and latents.shape[2] > smoke_frames:
                latents = latents[:, :, : int(smoke_frames)]
            if smoke_size:
                new_h, new_w = int(smoke_size[0]), int(smoke_size[1])
                b0, c0, t0, h0, w0 = latents.shape
                latents_btchw = latents.permute(0, 2, 1, 3, 4).reshape(b0 * t0, c0, h0, w0)
                latents_btchw = F.interpolate(
                    latents_btchw, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                latents = latents_btchw.view(b0, t0, c0, new_h, new_w).permute(0, 2, 1, 3, 4).contiguous()

        b, c, t, h, w = latents.shape
        if chunk_index is not None:
            # Chunk starts must be valid indices in [0, t-1], with 0 included.
            if len(chunk_index) == 0 or int(chunk_index[0]) != 0:
                raise RuntimeError(f"Invalid chunk_index={chunk_index}: must start at 0.")
            if int(chunk_index[-1]) >= int(t):
                raise RuntimeError(
                    f"Invalid chunk_index={chunk_index} for latent_t={t}: "
                    "last chunk start must be < latent_t."
                )
        if observed_train_effective_latent_t is None:
            observed_train_effective_latent_t = int(t)
            infer_hints["train_effective_latent_t"] = int(observed_train_effective_latent_t)
            if is_main:
                logger.info("Observed train effective latent_t=%d (post-window)", observed_train_effective_latent_t)
        timesteps = torch.randint(0, train_sampling_steps, (b,), device=device).long()
        process_supported_weighting = {"logit_normal", "stretched_logit_normal", "mode"}
        if use_sana_process_timesteps and (weighting_scheme in process_supported_weighting):
            timesteps = process_timesteps(
                weighting_scheme=weighting_scheme,
                train_sampling_steps=train_sampling_steps,
                size=(b,),
                device=device,
                logit_mean=weighting_logit_mean,
                logit_std=weighting_logit_std,
                p_low=weighting_p_low,
                p_high=weighting_p_high,
                num_frames=t,
                chunk_index=chunk_index,
                chunk_sampling_strategy=chunk_sampling_strategy,
                same_timestep_prob=same_timestep_prob,
                time_sampler=time_sampler,
            ).long()
        elif weighting_scheme in ["logit_normal", "mode"]:
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=b,
                logit_mean=weighting_logit_mean,
                logit_std=weighting_logit_std,
                mode_scale=weighting_mode_scale,
                p_low=weighting_p_low,
                p_high=weighting_p_high,
            )
            timesteps = (u * train_sampling_steps).long().to(device)
        elif use_sana_process_timesteps and is_main and update_step == 0 and micro_step == 0:
            logger.warning(
                "use_sana_process_timesteps=true but weighting_scheme=%s not supported by process_timesteps; "
                "falling back to legacy timestep sampling.",
                weighting_scheme,
            )

        data_info = build_data_info(b, h, w, device=device)
        batch_img_hw = batch.get("img_hw")
        if batch_img_hw is not None:
            if isinstance(batch_img_hw, torch.Tensor):
                img_hw = batch_img_hw.to(device=device, dtype=torch.float32)
            elif isinstance(batch_img_hw, list):
                if len(batch_img_hw) > 0 and isinstance(batch_img_hw[0], torch.Tensor):
                    img_hw = torch.stack([x.to(dtype=torch.float32) for x in batch_img_hw], dim=0).to(device=device)
                else:
                    img_hw = torch.tensor(batch_img_hw, device=device, dtype=torch.float32)
            else:
                img_hw = torch.tensor(batch_img_hw, device=device, dtype=torch.float32)
            if img_hw.dim() == 1:
                img_hw = img_hw.unsqueeze(0)
            if img_hw.shape[0] == 1 and b > 1:
                img_hw = img_hw.repeat(b, 1)
            if img_hw.shape[0] == b and img_hw.shape[1] == 2:
                data_info["img_hw"] = img_hw

        batch_aspect_ratio = batch.get("aspect_ratio")
        if batch_aspect_ratio is not None:
            if isinstance(batch_aspect_ratio, torch.Tensor):
                aspect_ratio = batch_aspect_ratio.to(device=device, dtype=torch.float32)
            elif isinstance(batch_aspect_ratio, list):
                if len(batch_aspect_ratio) > 0 and isinstance(batch_aspect_ratio[0], torch.Tensor):
                    aspect_ratio = torch.stack([x.to(dtype=torch.float32) for x in batch_aspect_ratio], dim=0).to(device=device)
                else:
                    aspect_ratio = torch.tensor(batch_aspect_ratio, device=device, dtype=torch.float32)
            else:
                aspect_ratio = torch.tensor(batch_aspect_ratio, device=device, dtype=torch.float32)
            if aspect_ratio.dim() == 0:
                aspect_ratio = aspect_ratio.view(1, 1)
            elif aspect_ratio.dim() == 1:
                aspect_ratio = aspect_ratio.unsqueeze(1)
            if aspect_ratio.shape[0] == 1 and b > 1:
                aspect_ratio = aspect_ratio.repeat(b, 1)
            if aspect_ratio.shape[0] == b:
                data_info["aspect_ratio"] = aspect_ratio
        # SANA DiT expects condition as [B, 1, L, C] and mask as [B, L].
        model_kwargs = {
            "y": student_embeds_for_dit.unsqueeze(1),
            "mask": mask,
            "data_info": data_info,
        }
        if chunk_index is not None:
            model_kwargs["chunk_index"] = chunk_index
        cond_y_shape_str = "x".join(str(v) for v in model_kwargs["y"].shape)
        cond_mask_shape_str = "x".join(str(v) for v in mask.shape) if isinstance(mask, torch.Tensor) else "none"
        cond_mask_nonpad = float("nan")
        cond_mask_tok_mean = float("nan")
        if isinstance(mask, torch.Tensor):
            cond_mask_2d = mask
            if cond_mask_2d.dim() == 4:
                cond_mask_2d = cond_mask_2d.squeeze(1).squeeze(1)
            elif cond_mask_2d.dim() == 3:
                cond_mask_2d = cond_mask_2d.squeeze(1)
            if cond_mask_2d.dim() == 2:
                cond_mask_nonpad = float(cond_mask_2d.float().mean().item())
                cond_mask_tok_mean = float(cond_mask_2d.float().sum(dim=1).mean().item())
        cond_y_token_norm = student_embeds_for_dit.float().norm(dim=-1)
        cond_y_norm_mean = float(cond_y_token_norm.mean().item())
        cond_y_norm_std = float(cond_y_token_norm.std(unbiased=False).item())
        if is_first_logged_iter:
            logger.info("Rank %s before SANA training_losses forward", rank)
        loss_term = sana_train_diffusion.training_losses(
            diffusion_model,
            latents,
            timesteps,
            model_kwargs=model_kwargs,
            timestep_weight=timestep_weight,
        )
        dit_fwd_dt = time.time() - fetch_t0 - fetch_dt - student_fwd_dt
        if is_first_logged_iter:
            logger.info("Rank %s after SANA training_losses forward", rank)
        # Primary objective from SANA scheduler (flow/diffusion depending on config).
        diff_loss = loss_term["loss"].mean()
        noisy_for_cfg = loss_term.get("x_t", latents)
        diff_weight = float(getattr(cfg.loss.get("diff", AttrDict()), "weight", 1.0))
        is_sync_step = (micro_step + 1) % grad_accum_steps == 0
        run_conditioning_diag = bool(
            is_sync_step
            and conditioning_diag_every > 0
            and ((update_step + 1) % conditioning_diag_every == 0)
        )
        run_conditioning_pred_probe = bool(
            is_sync_step
            and conditioning_pred_probe_every > 0
            and ((update_step + 1) % conditioning_pred_probe_every == 0)
            and len(conditioning_pred_probe_prompts) >= 2
        )
        cond_diag_shuffle_dloss = float("nan")
        cond_diag_uncond_dloss = float("nan")
        cond_diag_grad_norm = float("nan")
        cond_pred_delta_l2 = float("nan")
        cond_pred_delta_ratio = float("nan")
        if run_conditioning_diag and conditioning_diag_grad:
            # Non-leaf tensor; retain_grad is required to inspect conditioning gradient.
            student_embeds_for_dit.retain_grad()

        if run_conditioning_diag and (conditioning_diag_shuffle or conditioning_diag_uncond):
            with torch.no_grad():
                fixed_noise = loss_term.get("noise", None)
                if isinstance(fixed_noise, torch.Tensor):
                    fixed_noise = fixed_noise.detach()

                if conditioning_diag_shuffle and b >= 2:
                    perm = torch.randperm(b, device=device)
                    if bool(torch.all(perm == torch.arange(b, device=device))):
                        perm = torch.roll(perm, shifts=1, dims=0)
                    perm_model_kwargs = {
                        "y": model_kwargs["y"][perm],
                        "mask": (model_kwargs["mask"][perm] if isinstance(model_kwargs.get("mask"), torch.Tensor) else model_kwargs.get("mask")),
                        "data_info": data_info,
                    }
                    if chunk_index is not None:
                        perm_model_kwargs["chunk_index"] = chunk_index
                    perm_term = sana_train_diffusion.training_losses(
                        diffusion_model,
                        latents,
                        timesteps,
                        model_kwargs=perm_model_kwargs,
                        noise=fixed_noise,
                        timestep_weight=timestep_weight,
                    )
                    cond_diag_shuffle_dloss = float((perm_term["loss"].mean() - diff_loss.detach()).item())

                if conditioning_diag_uncond:
                    uncond_eval_emb, uncond_eval_mask = get_fixed_uncond_batch(
                        batch_size_local=student_embeds_for_dit.shape[0],
                        target_len=student_embeds_for_dit.shape[1],
                        target_dtype=student_embeds_for_dit.dtype,
                    )
                    if uncond_eval_emb is None:
                        uncond_prompts_eval = [cfg_delta_uncond_prompt] * len(prompts_cond)
                        uncond_eval_emb, uncond_eval_mask = student(uncond_prompts_eval, return_mask=True)
                        if uncond_eval_emb.dim() == 1:
                            uncond_eval_emb = uncond_eval_emb.unsqueeze(0).unsqueeze(0)
                        elif uncond_eval_emb.dim() == 2:
                            uncond_eval_emb = uncond_eval_emb.unsqueeze(0)
                    if uncond_eval_emb.dim() == 1:
                        uncond_eval_emb = uncond_eval_emb.unsqueeze(0).unsqueeze(0)
                    elif uncond_eval_emb.dim() == 2:
                        uncond_eval_emb = uncond_eval_emb.unsqueeze(0)
                    uncond_eval_emb = F.layer_norm(
                        uncond_eval_emb, (uncond_eval_emb.shape[-1],)
                    ).to(dtype)
                    if uncond_eval_mask is None:
                        uncond_eval_mask = torch.ones(
                            (uncond_eval_emb.shape[0], uncond_eval_emb.shape[1]),
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        uncond_eval_mask = uncond_eval_mask.to(device=device, dtype=torch.long)
                        if uncond_eval_mask.dim() == 1:
                            uncond_eval_mask = uncond_eval_mask.unsqueeze(0)
                        uncond_eval_mask = pad_or_trim_token_mask(uncond_eval_mask, uncond_eval_emb.shape[1])
                        if uncond_eval_mask.shape[0] == 1 and uncond_eval_emb.shape[0] > 1:
                            uncond_eval_mask = uncond_eval_mask.repeat(uncond_eval_emb.shape[0], 1)
                    uncond_model_kwargs = {
                        "y": uncond_eval_emb.unsqueeze(1),
                        "mask": uncond_eval_mask.unsqueeze(1).unsqueeze(1),
                        "data_info": data_info,
                    }
                    if chunk_index is not None:
                        uncond_model_kwargs["chunk_index"] = chunk_index
                    uncond_term = sana_train_diffusion.training_losses(
                        diffusion_model,
                        latents,
                        timesteps,
                        model_kwargs=uncond_model_kwargs,
                        noise=fixed_noise,
                        timestep_weight=timestep_weight,
                    )
                    cond_diag_uncond_dloss = float((uncond_term["loss"].mean() - diff_loss.detach()).item())
        if run_conditioning_pred_probe:
            probe_was_training = student_module.training
            try:
                student_module.eval()
                with torch.no_grad():
                    probe_prompts = conditioning_pred_probe_prompts[:2]
                    probe_emb, probe_mask = student_module(probe_prompts, return_mask=True)
                    if probe_emb.dim() == 1:
                        probe_emb = probe_emb.unsqueeze(0).unsqueeze(0)
                    elif probe_emb.dim() == 2:
                        probe_emb = probe_emb.unsqueeze(0)
                    probe_emb = probe_emb.float()
                    if student_pre_dit_layernorm:
                        probe_emb = F.layer_norm(probe_emb, (probe_emb.shape[-1],))
                    probe_emb = probe_emb.to(dtype)
                    if probe_mask is None:
                        probe_mask = torch.ones(
                            (probe_emb.shape[0], probe_emb.shape[1]), device=device, dtype=torch.long
                        )
                    else:
                        probe_mask = probe_mask.to(device=device, dtype=torch.long)
                        if probe_mask.dim() == 1:
                            probe_mask = probe_mask.unsqueeze(0)
                        probe_mask = pad_or_trim_token_mask(probe_mask, probe_emb.shape[1])

                    probe_data_info = {}
                    for k, v in data_info.items():
                        if isinstance(v, torch.Tensor) and v.shape[0] == b:
                            probe_data_info[k] = v[:1]
                        else:
                            probe_data_info[k] = v

                    probe_xt = noisy_for_cfg[:1]
                    probe_t = timesteps[:1]
                    probe_kwargs_a = {
                        "mask": probe_mask[:1].unsqueeze(1).unsqueeze(1),
                        "data_info": probe_data_info,
                    }
                    probe_kwargs_b = {
                        "mask": probe_mask[1:2].unsqueeze(1).unsqueeze(1),
                        "data_info": probe_data_info,
                    }
                    if chunk_index is not None:
                        probe_kwargs_a["chunk_index"] = chunk_index
                        probe_kwargs_b["chunk_index"] = chunk_index

                    pred_a = diffusion_model(
                        probe_xt,
                        probe_t,
                        probe_emb[:1].unsqueeze(1),
                        **probe_kwargs_a,
                    )
                    pred_b = diffusion_model(
                        probe_xt,
                        probe_t,
                        probe_emb[1:2].unsqueeze(1),
                        **probe_kwargs_b,
                    )
                    pred_delta = (pred_a.float() - pred_b.float()).reshape(1, -1)
                    pred_a_flat = pred_a.float().reshape(1, -1)
                    cond_pred_delta_l2 = float(pred_delta.norm(dim=1).mean().item())
                    cond_pred_delta_ratio = float(
                        cond_pred_delta_l2 / (pred_a_flat.norm(dim=1).mean().item() + 1e-8)
                    )
            except Exception as exc:
                if is_main:
                    logger.warning("conditioning_pred_probe failed: %s", exc)
            finally:
                if probe_was_training:
                    student_module.train()
        if probe_only and is_sync_step:
            if is_main:
                probe_log = (
                    "ProbeOnly Step %d | y_shape=%s mask_shape=%s mask_nonpad=%.4f mask_tok=%.2f "
                    "y_norm=%.4f±%.4f cond_shuffle_dloss=%.6f cond_uncond_dloss=%.6f "
                    "cond_pred_l2=%.6f cond_pred_ratio=%.6f"
                    % (
                        update_step + 1,
                        cond_y_shape_str,
                        cond_mask_shape_str,
                        cond_mask_nonpad,
                        cond_mask_tok_mean,
                        cond_y_norm_mean,
                        cond_y_norm_std,
                        cond_diag_shuffle_dloss,
                        cond_diag_uncond_dloss,
                        cond_pred_delta_l2,
                        cond_pred_delta_ratio,
                    )
                )
                logger.info(probe_log)
            probe_only_collected += 1
            if probe_only_collected >= probe_only_steps:
                if is_main:
                    logger.info(
                        "Probe-only mode complete: collected %d step(s), exiting before backward/optimizer.",
                        probe_only_collected,
                    )
                break
            micro_step += 1
            continue

        aux_active = (update_step + 1) >= aux_start_step
        distill_mse_loss = torch.tensor(0.0, device=device)
        distill_cos_loss = torch.tensor(0.0, device=device)
        distill_pooled_loss = torch.tensor(0.0, device=device)
        distill_contrastive_loss = torch.tensor(0.0, device=device)
        distill_hidden0_geom_loss = torch.tensor(0.0, device=device)
        functional_pred_mse_loss = torch.tensor(0.0, device=device)
        functional_pred_cos_loss = torch.tensor(0.0, device=device)
        if aux_active and distill_enabled and teacher_embeds is not None:
            student_mask_2d = student_prompt_mask
            if isinstance(student_mask_2d, torch.Tensor):
                student_mask_2d = student_mask_2d.to(device=device, dtype=torch.long)
                if student_mask_2d.dim() == 1:
                    student_mask_2d = student_mask_2d.unsqueeze(0)
                student_mask_2d = pad_or_trim_token_mask(student_mask_2d, student_embeds_for_dit.shape[1])
            student_distill = project_sana_conditioning_space(
                diffusion_model,
                student_embeds_for_dit,
                student_mask_2d,
                target_space=distill_target_space,
                freeze_modules_for_grad=distill_freeze_sana_conditioner,
            )
            with torch.no_grad():
                teacher_distill = project_sana_conditioning_space(
                    diffusion_model,
                    teacher_embeds.to(device=device, dtype=dtype, non_blocking=True),
                    teacher_mask,
                    target_space=distill_target_space,
                    freeze_modules_for_grad=False,
                )
            dmask = teacher_mask if distill_use_mask else None
            pooled_student = None
            pooled_teacher = None
            if distill_w_mse > 0.0:
                distill_mse_loss = compute_masked_mse(student_distill, teacher_distill, dmask)
            if distill_w_cos > 0.0:
                distill_cos_loss = compute_masked_token_cos(student_distill, teacher_distill, dmask)
            if (
                distill_w_pool > 0.0
                or distill_w_contrastive > 0.0
                or distill_hidden0_geom_weight > 0.0
            ):
                if dmask is None:
                    dmask_pool = torch.ones(student_distill.shape[:2], device=device, dtype=torch.long)
                else:
                    dmask_pool = dmask
                pooled_student = masked_mean_pool(student_distill, dmask_pool)
                pooled_teacher = masked_mean_pool(teacher_distill, dmask_pool)
            if distill_w_pool > 0.0 and pooled_student is not None and pooled_teacher is not None:
                distill_pooled_loss = 1.0 - F.cosine_similarity(pooled_student, pooled_teacher, dim=-1).mean()
            if distill_w_contrastive > 0.0 and pooled_student is not None and pooled_teacher is not None:
                distill_contrastive_loss = compute_inbatch_contrastive_distill_loss(
                    pooled_student,
                    pooled_teacher,
                    temperature=distill_contrastive_temp,
                )
            if (
                distill_hidden0_geom_weight > 0.0
                and pooled_student is not None
                and student_aux is not None
                and isinstance(student_aux, dict)
                and isinstance(student_aux.get("hidden0", None), torch.Tensor)
            ):
                hidden0_tokens = student_aux["hidden0"].to(device=device, dtype=torch.float32)
                if distill_hidden0_geom_layernorm:
                    hidden0_tokens = F.layer_norm(hidden0_tokens, (hidden0_tokens.shape[-1],))
                if student_mask_2d is None:
                    student_geom_mask = torch.ones(hidden0_tokens.shape[:2], device=device, dtype=torch.long)
                else:
                    student_geom_mask = pad_or_trim_token_mask(student_mask_2d, hidden0_tokens.shape[1])
                pooled_hidden0 = masked_mean_pool(hidden0_tokens, student_geom_mask)
                pooled_student_geom = masked_mean_pool(student_distill, student_geom_mask)
                distill_hidden0_geom_loss = compute_geometry_preservation_loss(
                    pooled_student_geom,
                    pooled_hidden0,
                )
        if aux_active and functional_run_this_step and teacher_embeds is not None:
            teacher_embeds_for_dit = teacher_embeds.to(device=device, dtype=dtype, non_blocking=True)
            if teacher_mask is None:
                teacher_mask_for_dit = torch.ones(
                    teacher_embeds_for_dit.shape[:2], device=device, dtype=torch.long
                )
            else:
                teacher_mask_for_dit = teacher_mask.to(device=device, dtype=torch.long)
                if teacher_mask_for_dit.dim() == 1:
                    teacher_mask_for_dit = teacher_mask_for_dit.unsqueeze(0)
            if teacher_mask_for_dit.dim() == 2:
                teacher_mask_for_dit = teacher_mask_for_dit.unsqueeze(1).unsqueeze(1)
            teacher_model_kwargs = {
                "mask": teacher_mask_for_dit,
                "data_info": data_info,
            }
            if chunk_index is not None:
                teacher_model_kwargs["chunk_index"] = chunk_index
            student_model_pred = extract_diffusion_tensor(loss_term["output"]).float()
            with torch.no_grad():
                teacher_model_pred = extract_diffusion_tensor(
                    diffusion_model(
                        noisy_for_cfg,
                        timesteps,
                        teacher_embeds_for_dit.unsqueeze(1),
                        **teacher_model_kwargs,
                    )
                ).float()
            if functional_pred_mse_weight > 0.0:
                functional_pred_mse_loss = F.mse_loss(student_model_pred, teacher_model_pred)
            if functional_pred_cos_weight > 0.0:
                functional_pred_cos_loss = 1.0 - F.cosine_similarity(
                    student_model_pred.reshape(student_model_pred.shape[0], -1),
                    teacher_model_pred.reshape(teacher_model_pred.shape[0], -1),
                    dim=-1,
                ).mean()

        semantic_var_loss = torch.tensor(0.0, device=device)
        semantic_cov_loss = torch.tensor(0.0, device=device)
        semantic_geom_loss = torch.tensor(0.0, device=device)
        semantic_probe_tok_mean = -1.0
        semantic_smol_tok_mean = -1.0
        if aux_active and semantic_enabled and is_sync_step and ((update_step + 1) % semantic_every_steps == 0):
            probe_emb, probe_mask = student(semantic_prompts, return_mask=True)
            probe_emb = probe_emb.float()
            if student_pre_dit_layernorm:
                probe_emb = F.layer_norm(probe_emb, (probe_emb.shape[-1],))
            if probe_mask is not None:
                probe_mask_long = probe_mask.to(device=probe_emb.device, dtype=torch.long)
                probe_pooled = masked_mean_pool(probe_emb, probe_mask_long)
                semantic_probe_tok_mean = float(probe_mask_long.float().sum(dim=1).mean().item())
            else:
                probe_pooled = probe_emb.mean(dim=1)
            semantic_var_loss, semantic_cov_loss = compute_prompt_anticollapse_losses(
                probe_pooled,
                target_std=semantic_target_std,
            )
            if semantic_geom_weight > 0.0:
                try:
                    with torch.no_grad():
                        if semantic_geom_source == "teacher":
                            if distill_online_teacher is None:
                                raise RuntimeError(
                                    "semantic_probe.geom_source=teacher requires an online teacher instance"
                                )
                            teacher_out = distill_online_teacher.forward_chi(
                                semantic_prompts,
                                use_chi_prompt=distill_online_use_chi_prompt,
                            )
                            teacher_h = F.layer_norm(
                                teacher_out["prompt_embeds"].float(),
                                (teacher_out["prompt_embeds"].shape[-1],),
                            )
                            teacher_mask = teacher_out.get("mask", None)
                            if teacher_mask is not None:
                                teacher_mask_long = teacher_mask.to(device=teacher_h.device, dtype=torch.long)
                                raw_pooled = masked_mean_pool(teacher_h, teacher_mask_long)
                                semantic_smol_tok_mean = float(teacher_mask_long.float().sum(dim=1).mean().item())
                            else:
                                raw_pooled = teacher_h.mean(dim=1)
                                semantic_smol_tok_mean = float(teacher_h.shape[1])
                        else:
                            raw_h, raw_mask, _ = student_module.encode_prompts(
                                semantic_prompts,
                                return_mask=True,
                                return_all_hidden_states=False,
                            )
                            raw_h = raw_h.float()
                            if raw_mask is not None:
                                raw_mask_long = raw_mask.to(device=raw_h.device, dtype=torch.long)
                                raw_pooled = masked_mean_pool(raw_h, raw_mask_long)
                                semantic_smol_tok_mean = float(raw_mask_long.float().sum(dim=1).mean().item())
                            else:
                                raw_pooled = raw_h.mean(dim=1)
                    semantic_geom_loss = compute_geometry_preservation_loss(probe_pooled, raw_pooled)
                except Exception as exc:
                    if is_main:
                        logger.warning("semantic geom loss compute failed: %s", exc)

        norm_loss = torch.tensor(0.0, device=device)
        if aux_active and cfg.loss.norm.enabled:
            token_norm = student_embeds_for_dit.float().norm(dim=-1)
            norm_target = float(cfg.loss.norm.target)
            norm_loss = ((token_norm - norm_target) ** 2).mean().to(device=device, dtype=student_embeds_for_dit.dtype)
        # Disable gradient sync on accumulation micro-steps for wrapped modules.
        with contextlib.ExitStack() as sync_stack:
            if not is_sync_step:
                if isinstance(student, DDP):
                    sync_stack.enter_context(student.no_sync())
                if isinstance(student, FSDP):
                    sync_stack.enter_context(student.no_sync())
                if isinstance(diffusion_model, (DDP, FSDP)):
                    sync_stack.enter_context(diffusion_model.no_sync())
            gate_loss = torch.tensor(0.0, device=device, dtype=student_embeds_for_dit.dtype)
            if aux_active and gate_loss_enabled and gate_loss_weight > 0 and not isinstance(student, FSDP):
                gate_now = student_module.adapter_output_gate.float()
                gate_loss = ((gate_now - gate_loss_target) ** 2).mean().to(
                    device=device, dtype=student_embeds_for_dit.dtype
                )

            loss = (
                diff_weight * diff_loss
                + cfg.loss.norm.weight * norm_loss
                + gate_loss_weight * gate_loss
                + distill_w_mse * distill_mse_loss
                + distill_w_cos * distill_cos_loss
                + distill_w_pool * distill_pooled_loss
                + distill_w_contrastive * distill_contrastive_loss
                + distill_hidden0_geom_weight * distill_hidden0_geom_loss
                + functional_pred_mse_weight * functional_pred_mse_loss
                + functional_pred_cos_weight * functional_pred_cos_loss
                + semantic_weight
                * (
                    semantic_var_weight * semantic_var_loss
                    + semantic_cov_weight * semantic_cov_loss
                    + semantic_geom_weight * semantic_geom_loss
                )
            )
            loss = loss / grad_accum_steps
            if deepspeed_engine is not None:
                deepspeed_engine.backward(loss)
            else:
                loss.backward()
        bwd_dt = time.time() - fetch_t0 - fetch_dt - student_fwd_dt - dit_fwd_dt
        if run_conditioning_diag and conditioning_diag_grad:
            grad_tensor = getattr(student_embeds_for_dit, "grad", None)
            if grad_tensor is not None:
                cond_diag_grad_norm = float(grad_tensor.detach().float().norm().item())
        if micro_step == 0:
            logger.info("Rank %s after backward", rank)

        grad_norm = 0.0
        should_log_update = bool(is_main and is_sync_step and log_every and (update_step + 1) % log_every == 0)
        lora_a_mean = 0.0
        lora_b_mean = 0.0
        lora_a_nonzero = 0
        lora_b_nonzero = 0
        lora_a_total = 0
        lora_b_total = 0
        cfg_delta_l2 = -1.0
        cfg_delta_mean_abs = -1.0
        cfg_ctx_cos = -1.0
        cfg_cond_ctx_norm = -1.0
        cfg_uncond_ctx_norm = -1.0

        if is_sync_step:
            # Manual all-reduce is only used for the non-DDP/non-FSDP fallback path.
            if world_size > 1 and not isinstance(student, (DDP, FSDP)) and student_sync:
                if micro_step == 0:
                    logger.info("Rank %s before student grad allreduce", rank)
                allreduce_gradients(
                    bridge_params,
                    world_size,
                    rank=rank,
                    tag="student_grad",
                    debug_log_every=sync_debug_every,
                    bucket_max_elems=sync_bucket_max_elems,
                    group=manual_sync_group,
                    backend=manual_sync_backend,
                )
                if micro_step == 0:
                    logger.info("Rank %s after student grad allreduce", rank)
            if world_size > 1 and (deepspeed_engine is None) and not isinstance(diffusion_model, (DDP, FSDP)) and dit_sync:
                if micro_step == 0:
                    logger.info("Rank %s before DiT grad allreduce", rank)
                allreduce_gradients(
                    dit_params,
                    world_size,
                    rank=rank,
                    tag="dit_grad",
                    debug_log_every=sync_debug_every,
                    trace_each=trace_each_dit_sync,
                    bucket_max_elems=sync_bucket_max_elems,
                    group=manual_sync_group,
                    backend=manual_sync_backend,
                )
                if micro_step == 0:
                    logger.info("Rank %s after DiT grad allreduce", rank)
            if should_log_update:
                grad_sq = 0.0
                for p in bridge_params + dit_params:
                    if p.grad is not None:
                        grad_sq += p.grad.detach().float().norm().item() ** 2
                grad_norm = math.sqrt(grad_sq) if grad_sq > 0 else 0.0
                if log_lora_grad:
                    for name, p in student_module.smolvlm2_model.named_parameters():
                        if "lora_A" in name:
                            lora_a_total += 1
                            if p.grad is not None:
                                g = p.grad.detach().float()
                                gnorm = float(g.norm().item())
                                lora_a_mean += gnorm
                                if g.abs().sum().item() > 0:
                                    lora_a_nonzero += 1
                        elif "lora_B" in name:
                            lora_b_total += 1
                            if p.grad is not None:
                                g = p.grad.detach().float()
                                gnorm = float(g.norm().item())
                                lora_b_mean += gnorm
                                if g.abs().sum().item() > 0:
                                    lora_b_nonzero += 1
                    lora_a_mean /= max(1, lora_a_total)
                    lora_b_mean /= max(1, lora_b_total)
            max_grad_norm = float(getattr(cfg.train.lr, "max_grad_norm", 0.0))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(bridge_params + dit_params, max_grad_norm)
            # Real optimizer update happens only on sync steps.
            if micro_step == 0:
                logger.info("Rank %s before optimizer.step", rank)
            if bridge_optimizer is not None:
                bridge_optimizer.step()
            if deepspeed_engine is not None:
                deepspeed_engine.step()
            elif optimizer is not None:
                optimizer.step()
            if micro_step == 0:
                logger.info("Rank %s after optimizer.step", rank)
            if gate_min_value > 0.0:
                with torch.no_grad():
                    student_module.adapter_output_gate.clamp_(min=gate_min_value)
            if bridge_scheduler is not None:
                bridge_scheduler.step()
            elif scheduler is not None:
                scheduler.step()
            if micro_step == 0:
                logger.info("Rank %s after scheduler.step", rank)
            if bridge_optimizer is not None:
                bridge_optimizer.zero_grad(set_to_none=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if deepspeed_engine is not None:
                deepspeed_zero_grad_compat(deepspeed_engine)
            update_step += 1

            if cfg_delta_every > 0 and update_step % cfg_delta_every == 0:
                with torch.no_grad():
                    cond_emb, cond_mask = student(prompts_cond, return_mask=True)
                    if student_pre_dit_layernorm:
                        cond_emb = F.layer_norm(cond_emb, (cond_emb.shape[-1],))
                    cond_emb = cond_emb.to(dtype)
                    if cfg_uncond_detach_bridge:
                        uncond_emb_raw, uncond_mask = get_fixed_uncond_batch(
                            batch_size_local=len(prompts_cond),
                            target_len=cond_emb.shape[1],
                            target_dtype=cond_emb.dtype,
                        )
                        if uncond_emb_raw is None:
                            uncond_prompts = [cfg_delta_uncond_prompt] * len(prompts_cond)
                            uncond_emb, uncond_mask = student(uncond_prompts, return_mask=True)
                        else:
                            uncond_emb = uncond_emb_raw
                    else:
                        uncond_prompts = [cfg_delta_uncond_prompt] * len(prompts_cond)
                        uncond_emb, uncond_mask = student(uncond_prompts, return_mask=True)
                    if student_pre_dit_layernorm:
                        uncond_emb = F.layer_norm(uncond_emb, (uncond_emb.shape[-1],))
                    uncond_emb = uncond_emb.to(dtype)
                    cond_mask_4d = None
                    uncond_mask_4d = None
                    cond_mask_2d = None
                    uncond_mask_2d = None
                    if cond_mask is not None:
                        cond_mask_2d = cond_mask.to(device=device, dtype=torch.long)
                        cond_mask_4d = cond_mask_2d.unsqueeze(1).unsqueeze(1)
                    if uncond_mask is not None:
                        uncond_mask_2d = uncond_mask.to(device=device, dtype=torch.long)
                        uncond_mask_4d = uncond_mask_2d.unsqueeze(1).unsqueeze(1)
                    if cond_mask_2d is not None:
                        cond_ctx = masked_mean_pool(cond_emb.float(), cond_mask_2d)
                    else:
                        cond_ctx = cond_emb.float().mean(dim=1)
                    if uncond_mask_2d is not None:
                        uncond_ctx = masked_mean_pool(uncond_emb.float(), uncond_mask_2d)
                    else:
                        uncond_ctx = uncond_emb.float().mean(dim=1)
                    cfg_ctx_cos = float(F.cosine_similarity(cond_ctx, uncond_ctx, dim=-1).mean().item())
                    cfg_cond_ctx_norm = float(cond_ctx.norm(dim=-1).mean().item())
                    cfg_uncond_ctx_norm = float(uncond_ctx.norm(dim=-1).mean().item())
                    pred_cond = diffusion_model(
                        noisy_for_cfg, timesteps, cond_emb.unsqueeze(1), mask=cond_mask_4d, data_info=data_info
                    )
                    pred_uncond = diffusion_model(
                        noisy_for_cfg, timesteps, uncond_emb.unsqueeze(1), mask=uncond_mask_4d, data_info=data_info
                    )
                    if isinstance(pred_cond, (tuple, list)):
                        pred_cond = pred_cond[0]
                    if isinstance(pred_uncond, (tuple, list)):
                        pred_uncond = pred_uncond[0]
                    delta = (pred_cond - pred_uncond).float()
                    cfg_delta_l2 = float(delta.norm().item())
                    cfg_delta_mean_abs = float(delta.abs().mean().item())
                    if dist.is_initialized() and world_size > 1:
                        delta_stats = torch.tensor(
                            [
                                cfg_delta_l2,
                                cfg_delta_mean_abs,
                                cfg_ctx_cos,
                                cfg_cond_ctx_norm,
                                cfg_uncond_ctx_norm,
                            ],
                            device=device,
                            dtype=torch.float32,
                        )
                        if manual_sync_backend == "gloo":
                            # Keep collectives on the same backend/group as manual grad sync
                            # to avoid mixing NCCL watchdog stalls with GLOO sync mode.
                            delta_stats_cpu = delta_stats.cpu()
                            dist.all_reduce(delta_stats_cpu, op=dist.ReduceOp.SUM, group=manual_sync_group)
                            delta_stats_cpu /= float(world_size)
                            delta_stats = delta_stats_cpu.to(device=device)
                        else:
                            dist.all_reduce(delta_stats, op=dist.ReduceOp.SUM, group=manual_sync_group)
                            delta_stats /= float(world_size)
                        cfg_delta_l2 = float(delta_stats[0].item())
                        cfg_delta_mean_abs = float(delta_stats[1].item())
                        cfg_ctx_cos = float(delta_stats[2].item())
                        cfg_cond_ctx_norm = float(delta_stats[3].item())
                        cfg_uncond_ctx_norm = float(delta_stats[4].item())

        if is_main and is_sync_step and log_every and update_step % log_every == 0:
            with torch.no_grad():
                if student_prompt_mask is not None and student_prompt_mask.shape[:2] == student_embeds_for_dit.shape[:2]:
                    pooled = masked_mean_pool(
                        student_embeds_for_dit.float(),
                        student_prompt_mask.to(device=student_embeds_for_dit.device, dtype=torch.long),
                    )
                else:
                    pooled = student_embeds_for_dit.float().mean(dim=1)
                pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-6)
                # This offdiag_cos is per-mini-batch only; with batch=1 it is NaN by design.
                if pooled.shape[0] < 2:
                    offdiag = float("nan")
                else:
                    sim = pooled @ pooled.T
                    offdiag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)].mean().item()
                emb_mean = student_embeds_for_dit.float().mean().item()
                emb_std = student_embeds_for_dit.float().std(unbiased=False).item()
                token_norm_mean = student_embeds_for_dit.float().norm(dim=-1).mean().item()
                gate_raw = float(student_module.adapter_output_gate.detach().float().item())
                lexical_gate_value = float("nan")
                projector_obj = getattr(student_module, "projector", None)
                if projector_obj is not None and hasattr(projector_obj, "lexical_gate_logit"):
                    lexical_gate_value = float(
                        torch.sigmoid(projector_obj.lexical_gate_logit.detach().float()).item()
                    )
            log_line = (
                "Step %d | mode=%s loss=%.6f diff=%.6f d_mse=%.6f d_cos=%.6f d_pool=%.6f d_nce=%.6f d_h0geom=%.6f norm=%.6f offdiag_cos=%.4f grad=%.4f "
                "emb_mean=%.4f emb_std=%.4f tok_norm=%.4f gate=%.4f cfg_drop=%.3f"
            ) % (
                update_step,
                active_batch_modality,
                loss.item() * grad_accum_steps,
                diff_loss.item(),
                distill_mse_loss.item(),
                distill_cos_loss.item(),
                distill_pooled_loss.item(),
                distill_contrastive_loss.item(),
                distill_hidden0_geom_loss.item(),
                norm_loss.item(),
                offdiag,
                grad_norm,
                emb_mean,
                emb_std,
                token_norm_mean,
                gate_raw,
                (float(dropped_prompts) / float(max(1, len(prompts_cond)))),
            )
            if not math.isnan(lexical_gate_value):
                log_line += " lex_gate=%.4f" % lexical_gate_value
            if joint_enabled:
                log_line += " v_micro=%d i_micro=%d" % (video_micro_count, image_micro_count)
            if log_lora_grad:
                log_line += (
                    " loraA=%d/%d:%.6f loraB=%d/%d:%.6f"
                    % (
                        lora_a_nonzero,
                        lora_a_total,
                        lora_a_mean,
                        lora_b_nonzero,
                        lora_b_total,
                        lora_b_mean,
                    )
                )
            if cfg_delta_l2 >= 0.0 and cfg_delta_mean_abs >= 0.0:
                log_line += " cfg_delta_l2=%.6f cfg_delta_abs=%.6f" % (cfg_delta_l2, cfg_delta_mean_abs)
            if cfg_ctx_cos >= -0.5:
                log_line += " cfg_ctx_cos=%.6f cfg_cond_norm=%.4f cfg_uncond_norm=%.4f" % (
                    cfg_ctx_cos,
                    cfg_cond_ctx_norm,
                    cfg_uncond_ctx_norm,
                )
            if conditioning_shape_log:
                log_line += " y_shape=%s mask_shape=%s mask_nonpad=%.4f mask_tok=%.2f y_norm=%.4f±%.4f" % (
                    cond_y_shape_str,
                    cond_mask_shape_str,
                    cond_mask_nonpad,
                    cond_mask_tok_mean,
                    cond_y_norm_mean,
                    cond_y_norm_std,
                )
            if not math.isnan(cond_diag_shuffle_dloss):
                log_line += " cond_shuffle_dloss=%.6f" % cond_diag_shuffle_dloss
            if not math.isnan(cond_diag_uncond_dloss):
                log_line += " cond_uncond_dloss=%.6f" % cond_diag_uncond_dloss
            if not math.isnan(cond_diag_grad_norm):
                log_line += " cond_grad=%.6f" % cond_diag_grad_norm
            if not math.isnan(cond_pred_delta_ratio):
                log_line += " cond_pred_l2=%.6f cond_pred_ratio=%.6f" % (
                    cond_pred_delta_l2,
                    cond_pred_delta_ratio,
                )
            if functional_enabled:
                log_line += " f_pred_mse=%.6f f_pred_cos=%.6f" % (
                    functional_pred_mse_loss.item(),
                    functional_pred_cos_loss.item(),
                )
            if semantic_enabled:
                log_line += " sem_var=%.6f sem_cov=%.6f sem_geom=%.6f" % (
                    semantic_var_loss.item(),
                    semantic_cov_loss.item(),
                    semantic_geom_loss.item(),
                )
                if semantic_probe_tok_mean >= 0.0:
                    log_line += " sem_tok_mcp=%.2f" % semantic_probe_tok_mean
                if semantic_smol_tok_mean >= 0.0:
                    log_line += " sem_tok_smol=%.2f" % semantic_smol_tok_mean
            logger.info(log_line)

        if is_main and is_sync_step and debug_probe_every > 0 and debug_probe_prompts and update_step % debug_probe_every == 0:
            was_training = student_module.training
            try:
                student_module.eval()
                with torch.no_grad():
                    probe_emb, probe_mask = student_module(debug_probe_prompts, return_mask=True)
                    probe_emb = probe_emb.float()
                    probe_tok_mean = -1.0
                    raw_tok_mean = -1.0
                    if probe_mask is not None:
                        probe_mask_long = probe_mask.to(device=probe_emb.device, dtype=torch.long)
                        probe_tok_mean = float(probe_mask_long.float().sum(dim=1).mean().item())
                        pooled = masked_mean_pool(probe_emb, probe_mask_long)
                    else:
                        pooled = probe_emb.mean(dim=1)
                    pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-6)
                    if pooled.shape[0] > 1:
                        sim = pooled @ pooled.T
                        offdiag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
                        probe_offdiag_mean = float(offdiag.mean().item())
                        probe_offdiag_min = float(offdiag.min().item())
                        probe_offdiag_max = float(offdiag.max().item())
                    else:
                        probe_offdiag_mean = 0.0
                        probe_offdiag_min = 0.0
                        probe_offdiag_max = 0.0
                    smol_offdiag_mean = None
                    if (
                        hasattr(student_module, "projector_type")
                        and str(getattr(student_module, "projector_type", "")).lower()
                        in {"mcp_tiny", "mcp_full", "mcp_lexical_gated", "mcp_lexical_bottleneck"}
                        and hasattr(student_module, "encode_prompts")
                    ):
                        try:
                            raw_h, raw_mask, _ = student_module.encode_prompts(
                                debug_probe_prompts,
                                return_mask=True,
                                return_all_hidden_states=False,
                            )
                            if raw_mask is not None:
                                raw_mask_long = raw_mask.to(device=raw_h.device, dtype=torch.long)
                                raw_tok_mean = float(raw_mask_long.float().sum(dim=1).mean().item())
                                raw_pooled = masked_mean_pool(raw_h.float(), raw_mask_long)
                            else:
                                raw_pooled = raw_h.float().mean(dim=1)
                            raw_pooled = raw_pooled / (raw_pooled.norm(dim=-1, keepdim=True) + 1e-6)
                            if raw_pooled.shape[0] > 1:
                                raw_sim = raw_pooled @ raw_pooled.T
                                raw_offdiag = raw_sim[
                                    ~torch.eye(raw_sim.shape[0], dtype=torch.bool, device=raw_sim.device)
                                ]
                                smol_offdiag_mean = float(raw_offdiag.mean().item())
                        except Exception as exc:
                            logger.warning("probe_semantic raw_smol compute failed: %s", exc)
                if smol_offdiag_mean is None:
                    logger.info(
                        "Step %d | probe_semantic offdiag_cos(mean/min/max)=%.6f/%.6f/%.6f prompts=%d",
                        update_step,
                        probe_offdiag_mean,
                        probe_offdiag_min,
                        probe_offdiag_max,
                        len(debug_probe_prompts),
                    )
                    if probe_tok_mean >= 0.0:
                        logger.info(
                            "Step %d | probe_semantic_tokens mcp_tok_mean=%.2f prompts=%d",
                            update_step,
                            probe_tok_mean,
                            len(debug_probe_prompts),
                        )
                else:
                    logger.info(
                        "Step %d | probe_semantic mcp_offdiag(mean/min/max)=%.6f/%.6f/%.6f smol_offdiag=%.6f prompts=%d"
                        " mcp_tok=%.2f smol_tok=%.2f",
                        update_step,
                        probe_offdiag_mean,
                        probe_offdiag_min,
                        probe_offdiag_max,
                        smol_offdiag_mean,
                        len(debug_probe_prompts),
                        probe_tok_mean,
                        raw_tok_mean,
                    )
            finally:
                if was_training:
                    student_module.train()

        if is_sync_step and save_every and update_step % save_every == 0:
            # Save trainable student + trainable DiT slices for fast resume/infer iteration.
            ckpt_path = os.path.join(run_dir, f"checkpoint_step{update_step}.pt")
            student_state = build_student_checkpoint_state(student, student_module)
            dit_state = get_trainable_dit_state_dict(diffusion_model)
            if is_main:
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(
                    {
                        "step": update_step,
                        "micro_step": micro_step + 1,
                        "student_state": student_state,
                        "dit_trainable_state": dit_state,
                        "optimizer": (
                            optimizer.state_dict()
                            if optimizer is not None
                            else (bridge_optimizer.state_dict() if bridge_optimizer is not None else {})
                        ),
                        "scheduler": (
                            scheduler.state_dict()
                            if scheduler is not None
                            else (bridge_scheduler.state_dict() if bridge_scheduler is not None else {})
                        ),
                        "dit_train_modules": list(dit_train_modules),
                        "infer_hints": infer_hints,
                    },
                    ckpt_path,
                )
                logger.info("Saved checkpoint: %s", ckpt_path)
            if use_barrier:
                safe_barrier(local_rank)

        micro_step += 1

    student_state = build_student_checkpoint_state(student, student_module)
    dit_state = get_trainable_dit_state_dict(diffusion_model)
    if is_main and (not probe_only):
        final_ckpt_path = os.path.join(run_dir, "checkpoint_final.pt")
        os.makedirs(os.path.dirname(final_ckpt_path), exist_ok=True)
        torch.save(
            {
                "step": update_step,
                "micro_step": micro_step,
                "student_state": student_state,
                "dit_trainable_state": dit_state,
                "optimizer": (
                    optimizer.state_dict()
                    if optimizer is not None
                    else (bridge_optimizer.state_dict() if bridge_optimizer is not None else {})
                ),
                "scheduler": (
                    scheduler.state_dict()
                    if scheduler is not None
                    else (bridge_scheduler.state_dict() if bridge_scheduler is not None else {})
                ),
                "dit_train_modules": list(dit_train_modules),
                "infer_hints": infer_hints,
            },
            final_ckpt_path,
        )
        logger.info("Saved final checkpoint: %s", final_ckpt_path)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    elif is_main and probe_only:
        logger.info("Probe-only mode: skipped final checkpoint save.")

    if dist.is_initialized():
        if use_barrier:
            safe_barrier(local_rank)
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Teacher-free Stage1 training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-gpus", type=int, default=1)
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from checkpoint path")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Initialize model weights from checkpoint path but start training fresh at step 0.",
    )
    parser.add_argument(
        "--resume-skip-optimizer-state",
        action="store_true",
        help="Skip loading optimizer/scheduler state when resuming (useful for debug resume or trainable-set changes).",
    )
    parser.add_argument(
        "--single-gpu-debug",
        action="store_true",
        help="Force single-process/single-GPU debug mode (disables DDP/FSDP/DeepSpeed in config).",
    )
    parser.add_argument(
        "--debug-gpu-id",
        type=int,
        default=None,
        help="Physical GPU id to use in --single-gpu-debug mode (sets CUDA_VISIBLE_DEVICES).",
    )
    args = parser.parse_args()
    if args.resume_from and args.init_from:
        raise RuntimeError("--resume-from and --init-from are mutually exclusive; choose one.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # This warning is non-fatal (layout/perf hint) and can spam distributed logs heavily.
    # Keep scope narrow to only the known DDP bucket-view stride warning.
    warnings.filterwarnings(
        "ignore",
        message=r"Grad strides do not match bucket view strides\..*",
        category=UserWarning,
    )
    if args.single_gpu_debug:
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if env_world_size > 1:
            raise RuntimeError(
                "--single-gpu-debug must be launched without torchrun "
                f"(detected WORLD_SIZE={env_world_size})."
            )
        args.max_gpus = 1
        if args.debug_gpu_id is not None:
            # Map selected physical GPU to logical cuda:0 for stable local debugging.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.debug_gpu_id)
            if args.device is None:
                args.device = "cuda:0"

    cfg = load_stage1_config(args.config)
    if args.single_gpu_debug:
        # Disable all distributed wrappers to make breakpoint debugging deterministic.
        cfg.run.student_ddp = False
        cfg.run.student_fsdp = False
        cfg.run.student_sync = False
        cfg.run.dit_sync = False
        cfg.run.use_barrier = False
        cfg.model.dit.fsdp = False
        cfg.model.dit.ddp = False
        cfg.model.dit.deepspeed = False
        logger.info(
            "Single-GPU debug mode enabled: device=%s max_gpus=%d (DDP/FSDP/DeepSpeed disabled)",
            args.device or "cuda",
            args.max_gpus,
        )
    train_teacher_free(cfg, args)


if __name__ == "__main__":
    main()
