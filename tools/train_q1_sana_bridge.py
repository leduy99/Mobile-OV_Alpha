#!/usr/bin/env python3
"""
Stage 1 distillation: align SmolVLM2->bridge outputs to SANA text encoder embeddings.

Supports two modes:
- Legacy CSV prompts (OpenVid captions) for quick sanity checks.
- Config-driven multi-source prompt mixing for generalization.
"""

import argparse
import json
import time
import logging
import math
import os
import random
import re
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

import yaml
from dataclasses import fields, is_dataclass
from typing import get_args, get_origin
from datasets import load_dataset

from nets.omni.modules.sana_prompt_bridge import SanaPromptBridge
from nets.third_party.sana.diffusion.longsana.utils.model_wrapper import SanaTextEncoder
from diffusion.utils.config import SanaVideoConfig, model_video_init_config
from diffusion.model.builder import build_model
logger = logging.getLogger(__name__)


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def load_prompts_from_csv(csv_path: str, num_samples: int, seed: int, shuffle: bool) -> List[str]:
    df = pd.read_csv(csv_path)
    if "caption" not in df.columns:
        raise ValueError(f"CSV missing 'caption' column: {csv_path}")

    captions = df["caption"].dropna().astype(str).tolist()
    if shuffle:
        random.Random(seed).shuffle(captions)
    if num_samples is not None and num_samples > 0:
        captions = captions[:num_samples]

    logger.info("Loaded %d prompts from %s", len(captions), csv_path)
    return captions


def apply_motion_and_chi(prompts: List[str], config: SanaVideoConfig, motion_score: int, use_chi_prompt: bool) -> List[str]:
    if motion_score is not None and motion_score > 0:
        prompts = [f"{p.strip()} motion score: {int(motion_score)}." for p in prompts]

    if use_chi_prompt:
        chi_list = getattr(config.text_encoder, "chi_prompt", None)
        if chi_list:
            chi_prompt = "\n".join(chi_list)
            prompts = [chi_prompt + p for p in prompts]
    return prompts


def pad_or_trim_teacher(teacher_embeds: torch.Tensor, teacher_mask: torch.Tensor, target_len: int):
    current_len = teacher_embeds.shape[1]
    if current_len == target_len:
        return teacher_embeds, teacher_mask

    if current_len > target_len:
        teacher_embeds = teacher_embeds[:, :target_len, :]
        teacher_mask = teacher_mask[:, :target_len]
        return teacher_embeds, teacher_mask

    pad_len = target_len - current_len
    pad_embed = torch.zeros(
        teacher_embeds.shape[0], pad_len, teacher_embeds.shape[2],
        device=teacher_embeds.device, dtype=teacher_embeds.dtype
    )
    pad_mask = torch.zeros(
        teacher_mask.shape[0], pad_len,
        device=teacher_mask.device, dtype=teacher_mask.dtype
    )
    teacher_embeds = torch.cat([teacher_embeds, pad_embed], dim=1)
    teacher_mask = torch.cat([teacher_mask, pad_mask], dim=1)
    return teacher_embeds, teacher_mask


def calibrate_student_embeds(
    student_embeds: torch.Tensor,
    teacher_embeds: torch.Tensor,
    teacher_mask: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match student token distribution to teacher (mean/std) with mask-aware LN."""
    if teacher_mask is None:
        teacher_mask = torch.ones(
            teacher_embeds.shape[:2], device=teacher_embeds.device, dtype=teacher_embeds.dtype
        )
    mask = teacher_mask.unsqueeze(-1).to(student_embeds.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    mu_t = (teacher_embeds * mask).sum(dim=1, keepdim=True) / denom
    var_t = ((teacher_embeds - mu_t) ** 2 * mask).sum(dim=1, keepdim=True) / denom
    std_t = torch.sqrt(var_t + eps)

    mu_s = (student_embeds * mask).sum(dim=1, keepdim=True) / denom
    var_s = ((student_embeds - mu_s) ** 2 * mask).sum(dim=1, keepdim=True) / denom
    std_s = torch.sqrt(var_s + eps)

    mu_t = mu_t.detach()
    std_t = std_t.detach()

    student_ln = (student_embeds - mu_s) / std_s
    return student_ln * std_t + mu_t


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def masked_mean_norm(embeds: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(embeds.dtype)
    token_norm = embeds.norm(dim=-1)
    denom = mask_f.sum(dim=1).clamp_min(eps)
    return (token_norm * mask_f.squeeze(-1)).sum(dim=1) / denom


def masked_mean_pool(embeds: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(embeds.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(eps)
    return (embeds * mask_f).sum(dim=1) / denom.squeeze(1)


def info_nce_loss(
    student_vecs: torch.Tensor,
    teacher_vecs: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if student_vecs.shape[0] <= 1:
        return torch.tensor(0.0, device=student_vecs.device)
    student_norm = F.normalize(student_vecs, dim=-1)
    teacher_norm = F.normalize(teacher_vecs, dim=-1)
    logits = student_norm @ teacher_norm.T
    logits = logits / max(temperature, 1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def sample_effect_timesteps(cfg: "AttrDict", batch_size: int, device: torch.device) -> List[torch.Tensor]:
    ranges = getattr(cfg.loss.effect, "timestep_ranges", None)
    if ranges:
        timesteps_list = []
        max_t = int(cfg.sana.timesteps.num_train_timesteps) - 1
        for pair in ranges:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            lo, hi = int(pair[0]), int(pair[1])
            lo = max(0, min(lo, max_t))
            hi = max(0, min(hi, max_t))
            hi = max(lo, hi)
            timesteps_list.append(torch.randint(lo, hi + 1, (batch_size,), device=device))
        if timesteps_list:
            return timesteps_list
    num_steps = int(getattr(cfg.loss.effect, "num_timesteps_per_batch", 1) or 1)
    return [
        torch.randint(0, cfg.sana.timesteps.num_train_timesteps, (batch_size,), device=device)
        for _ in range(max(1, num_steps))
    ]


def load_sana_diffusion_model(
    sana_cfg: SanaVideoConfig,
    sana_ckpt_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    skip_ckpt_load: bool = False,
    use_grad_checkpoint: bool = False,
    grad_checkpoint_step: int = 1,
) -> torch.nn.Module:
    logger = logging.getLogger(__name__)
    try:
        import diffusion.model.nets  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"Failed to import SANA model registry: {exc}") from exc

    latent_size = sana_cfg.model.image_size // sana_cfg.vae.vae_downsample_rate
    model_cfg = model_video_init_config(sana_cfg, latent_size=latent_size)
    model_cfg["type"] = sana_cfg.model.model
    logger.info("Building SANA model...")
    logger.info(
        "SANA build options: use_grad_checkpoint=%s gc_step=%d use_fp32_attention=%s",
        bool(use_grad_checkpoint),
        max(1, int(grad_checkpoint_step)),
        bool(getattr(sana_cfg.model, "fp32_attention", False)),
    )
    sana_model = build_model(
        model_cfg,
        use_grad_checkpoint=bool(use_grad_checkpoint),
        use_fp32_attention=getattr(sana_cfg.model, "fp32_attention", False),
        gc_step=max(1, int(grad_checkpoint_step)),
    )
    logger.info("Built SANA model")
    sana_model.eval()

    if not skip_ckpt_load:
        ckpt_path = sana_cfg.model.load_from
        if isinstance(ckpt_path, str) and ckpt_path.startswith("hf://"):
            suffix = ckpt_path.replace("hf://Efficient-Large-Model/SANA-Video_2B_480p/", "")
            ckpt_path = os.path.join(sana_ckpt_dir, suffix)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(ckpt_path)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SANA checkpoint not found: {ckpt_path}")

        t0 = time.time()
        logger.info("Loading SANA checkpoint from %s ...", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        logger.info("Loaded SANA checkpoint in %.1fs", time.time() - t0)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if "pos_embed" in state_dict and "pos_embed" in sana_model.state_dict():
            if state_dict["pos_embed"].shape != sana_model.state_dict()["pos_embed"].shape:
                state_dict.pop("pos_embed")
        sana_model.load_state_dict(state_dict, strict=False)
    sana_model = sana_model.to(device=device, dtype=dtype)
    for param in sana_model.parameters():
        param.requires_grad = False
    return sana_model


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__


def to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attrdict(v) for v in obj]
    return obj


def load_stage1_config(path: str) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return to_attrdict(cfg)


def _dataclass_from_dict(cls, data: Dict[str, Any]):
    if not is_dataclass(cls):
        return data

    kwargs = {}
    for field in fields(cls):
        if field.name not in data:
            continue
        value = data[field.name]
        ftype = field.type
        dataclass_type = None
        if is_dataclass(ftype):
            dataclass_type = ftype
        else:
            origin = get_origin(ftype)
            if origin is not None:
                for arg in get_args(ftype):
                    if is_dataclass(arg):
                        dataclass_type = arg
                        break
        if dataclass_type and isinstance(value, dict):
            kwargs[field.name] = _dataclass_from_dict(dataclass_type, value)
        else:
            kwargs[field.name] = value
    return cls(**kwargs)


def load_sana_config(path: str) -> SanaVideoConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _dataclass_from_dict(SanaVideoConfig, cfg)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def init_distributed(max_gpus: int):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > max_gpus:
        raise RuntimeError(f"WORLD_SIZE={world_size} exceeds max_gpus={max_gpus}")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=60),
        )

    return rank, world_size, local_rank


def normalize_prompt(text: str, normalize_whitespace: bool, strip: bool, remove_double_newlines: bool) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n")
    if remove_double_newlines:
        text = re.sub(r"\n{2,}", "\n", text)
    text = text.replace("<image>", "").replace("<video>", "")
    if normalize_whitespace:
        text = " ".join(text.split())
    if strip:
        text = text.strip()
    return text


def apply_prompt_dropout(prompt: str, prob: float, rng: random.Random) -> str:
    if prob and rng.random() < prob:
        return ""
    return prompt


def apply_span_mask(prompt: str, prob: float, max_tokens: int, rng: random.Random) -> str:
    if not prompt or prob <= 0.0 or rng.random() > prob:
        return prompt
    tokens = prompt.split()
    if not tokens:
        return prompt
    span_len = min(max_tokens, max(1, len(tokens)))
    if span_len >= len(tokens):
        return ""
    start = rng.randint(0, len(tokens) - span_len)
    masked = tokens[:start] + tokens[start + span_len :]
    return " ".join(masked)


def truncate_prompt(prompt: str, tokenizer, max_tokens: Optional[int]) -> str:
    if not prompt or max_tokens is None:
        return prompt
    try:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    except Exception:
        return prompt
    if len(token_ids) <= max_tokens:
        return prompt
    truncated = tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
    return truncated.strip()


def extract_from_conversations(conversations: List[dict]) -> str:
    if not conversations:
        return ""
    human_msgs = [c.get("value", "") for c in conversations if c.get("from") in ("human", "user")]
    if not human_msgs:
        human_msgs = [c.get("value", "") for c in conversations]
    return "\n".join([m for m in human_msgs if m])


def extract_from_captions(captions: List[dict]) -> str:
    if not captions:
        return ""
    # Prefer summary caption if present (idx == -1).
    for item in captions:
        if str(item.get("idx")) == "-1" and item.get("content"):
            return item["content"]
    return captions[0].get("content", "")


def extract_prompt(record: Dict[str, Any], candidates: List[str]) -> str:
    for key in candidates:
        if key not in record:
            continue
        val = record[key]
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            if val and isinstance(val[0], dict):
                if key == "conversations":
                    return extract_from_conversations(val)
                if key == "captions":
                    return extract_from_captions(val)
                # fallback to "value" field if present
                values = [v.get("value", "") for v in val]
                return "\n".join([v for v in values if v])
            if val and isinstance(val[0], str):
                return "\n".join([v for v in val if v])
        if isinstance(val, dict):
            return val.get("value", "") or val.get("text", "") or ""
    # extra fallback
    if "captions" in record:
        return extract_from_captions(record.get("captions", []))
    return ""


def find_files(root: str, suffixes: Tuple[str, ...]) -> List[str]:
    paths = []
    for base, _, files in os.walk(root):
        for name in files:
            if name.endswith(suffixes):
                paths.append(os.path.join(base, name))
    return sorted(paths)


def load_source_dataset(source_cfg: AttrDict, streaming: bool) -> Any:
    source_type = source_cfg.type
    path = source_cfg.path
    if source_type in ("hf_parquet", "parquet"):
        files = find_files(path, (".parquet",))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        return load_dataset("parquet", data_files=files, split="train", streaming=streaming)
    if source_type in ("json", "jsonl", "hf_dataset"):
        if os.path.isfile(path):
            files = [path]
        else:
            files = find_files(path, (".json", ".jsonl"))
        if not files:
            raise FileNotFoundError(f"No json/jsonl files found in {path}")
        if source_cfg.get("native_json", False):
            return JsonFileDataset(files)
        return load_dataset("json", data_files=files, split="train", streaming=streaming)
    if source_type in ("csv", "manifest_csv"):
        if os.path.isfile(path):
            files = [path]
        else:
            files = find_files(path, (".csv",))
        if not files:
            raise FileNotFoundError(f"No csv files found in {path}")
        return CsvFileDataset(files)
    raise ValueError(f"Unsupported data source type: {source_type}")


class PromptMixerDataset(IterableDataset):
    def __init__(self, sources: List[Dict[str, Any]], seed: int, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.sources = sources
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + self.rank * 1000 + worker_id)

        datasets = []
        for source in self.sources:
            dataset = source["dataset"]
            if hasattr(dataset, "shard"):
                if self.world_size > 1:
                    dataset = dataset.shard(num_shards=self.world_size, index=self.rank)
                if num_workers > 1:
                    dataset = dataset.shard(num_shards=num_workers, index=worker_id)
            datasets.append(dataset)

        iterators = [iter(ds) for ds in datasets]
        weights = [float(src["weight"]) for src in self.sources]
        if not any(weights):
            weights = [1.0] * len(weights)

        while True:
            idx = rng.choices(range(len(iterators)), weights=weights, k=1)[0]
            try:
                example = next(iterators[idx])
            except StopIteration:
                iterators[idx] = iter(datasets[idx])
                example = next(iterators[idx])

            prompt = extract_prompt(example, self.sources[idx]["candidates"])
            if not prompt:
                continue
            yield prompt


class JsonFileDataset(IterableDataset):
    def __init__(self, files: List[str]):
        super().__init__()
        self.files = files

    def _iter_json_array(self, fp):
        decoder = json.JSONDecoder()
        buffer = ""
        in_array = False

        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk

            if not in_array:
                while buffer and buffer[0].isspace():
                    buffer = buffer[1:]
                if buffer.startswith("["):
                    buffer = buffer[1:]
                    in_array = True
                else:
                    continue

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "]":
                    return
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                buffer = buffer[idx:]
                yield obj

    def __iter__(self):
        for file_path in self.files:
            if file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    first = f.read(1)
                    f.seek(0)
                    if first == "[":
                        for item in self._iter_json_array(f):
                            yield item
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                yield item
                        elif isinstance(data, dict):
                            for _, item in data.items():
                                if isinstance(item, list):
                                    for entry in item:
                                        yield entry
                                else:
                                    yield item


class CsvFileDataset(IterableDataset):
    def __init__(self, files: List[str], chunksize: int = 8192):
        super().__init__()
        self.files = files
        self.chunksize = int(max(1, chunksize))

    def __iter__(self):
        for file_path in self.files:
            for chunk in pd.read_csv(file_path, chunksize=self.chunksize):
                for row in chunk.to_dict("records"):
                    yield row


def build_data_info(batch_size: int, height: int, width: int, device: torch.device) -> Dict[str, torch.Tensor]:
    img_hw = torch.tensor([[height, width]], device=device, dtype=torch.float)
    return {"img_hw": img_hw.repeat(batch_size, 1)}


def compute_guided_pred(
    model: torch.nn.Module,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    cond_y: torch.Tensor,
    uncond_y: torch.Tensor,
    cond_mask: Optional[torch.Tensor],
    uncond_mask: Optional[torch.Tensor],
    data_info: Dict[str, torch.Tensor],
    guidance_scale: float,
) -> torch.Tensor:
    latents_in = torch.cat([latents, latents], dim=0)
    timesteps_in = torch.cat([timesteps, timesteps], dim=0)
    y_in = torch.cat([uncond_y, cond_y], dim=0)
    mask_in = None
    if cond_mask is not None and uncond_mask is not None:
        mask_in = torch.cat([uncond_mask, cond_mask], dim=0)

    data_info_in = dict(data_info)
    if "img_hw" in data_info_in and data_info_in["img_hw"].shape[0] == latents.shape[0]:
        data_info_in["img_hw"] = data_info_in["img_hw"].repeat(2, 1)

    pred = model(latents_in, timesteps_in, y_in, mask=mask_in, data_info=data_info_in)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    pred_uncond, pred_cond = pred.chunk(2)
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


def get_effect_weight(cfg: AttrDict, step: int) -> float:
    schedule = cfg.loss.effect.weight_schedule
    if not schedule or schedule.type != "linear_ramp":
        return float(cfg.loss.effect.weight)
    start = int(schedule.start_step)
    end = int(schedule.end_step)
    if step <= start:
        return float(schedule.start_value)
    if step >= end:
        return float(schedule.end_value)
    ratio = float(step - start) / float(max(end - start, 1))
    return float(schedule.start_value + ratio * (schedule.end_value - schedule.start_value))


def get_distill_batch_size(cfg_section: AttrDict, batch_size: int) -> int:
    value = getattr(cfg_section, "batch_size", None)
    if value is None:
        return batch_size
    try:
        value = int(value)
    except (TypeError, ValueError):
        return batch_size
    if value <= 0:
        return batch_size
    return min(batch_size, value)


def build_prompt_batch(
    prompts: List[str],
    cfg: AttrDict,
    teacher: SanaTextEncoder,
    step: int,
    rng: random.Random,
) -> Tuple[List[str], List[str]]:
    preprocessing = cfg.data.preprocessing
    processed = []
    for prompt in prompts:
        prompt = normalize_prompt(
            prompt,
            normalize_whitespace=preprocessing.normalize_whitespace,
            strip=preprocessing.strip,
            remove_double_newlines=preprocessing.remove_double_newlines,
        )
        prompt = apply_prompt_dropout(prompt, preprocessing.prompt_dropout_prob, rng)
        prompt = apply_span_mask(prompt, preprocessing.span_mask_prob, preprocessing.span_mask_max_tokens, rng)

        max_tokens = None
        curriculum = preprocessing.length_curriculum
        if curriculum and curriculum.enabled:
            if step <= curriculum.phase0_steps:
                max_tokens = curriculum.short_max_tokens
            elif step <= curriculum.phase1_steps:
                max_tokens = curriculum.medium_max_tokens
        if max_tokens is not None:
            prompt = truncate_prompt(prompt, teacher.tokenizer, max_tokens)

        if prompt and preprocessing.motion_score and preprocessing.motion_score > 0:
            prompt = f"{prompt} motion score: {int(preprocessing.motion_score)}."
        processed.append(prompt)

    if preprocessing.use_chi_prompt:
        chi_list = getattr(teacher.cfg.text_encoder, "chi_prompt", None)
        if chi_list:
            chi_prompt = "\n".join(chi_list)
            student_prompts = [chi_prompt + p for p in processed]
            return processed, student_prompts

    return processed, processed


def train_with_config(cfg: AttrDict, args: argparse.Namespace):
    from tools.train_stage1_teacher_free import (
        build_student as build_teacher_free_student,
        build_student_checkpoint_state,
        ensure_sana_assets_available,
        ensure_smolvlm2_checkpoint_available,
        load_lora_state_dict,
        load_trainable_smolvlm2_state_dict,
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_home = str(getattr(cfg.run, "hf_home", "") or "").strip()
    hf_cache = str(getattr(cfg.run, "huggingface_hub_cache", "") or "").strip()
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
    if hf_cache:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache)

    rank, world_size, local_rank = init_distributed(args.max_gpus)
    is_main = rank == 0

    set_seed(cfg.run.seed + rank)
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device or "cuda")

    precision = (args.precision or cfg.run.precision or "bf16").lower()
    if precision in ("fp16", "float16"):
        dtype = torch.float16
    elif precision in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    output_dir = args.output_dir or cfg.run.output_dir
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        run_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = output_dir

    sana_cfg = load_sana_config(cfg.sana.config)

    if cfg.sana.dit_ckpt and cfg.sana.dit_ckpt.endswith(".pth"):
        sana_cfg.model.load_from = cfg.sana.dit_ckpt
        sana_ckpt_dir = os.path.dirname(cfg.sana.dit_ckpt)
    else:
        sana_ckpt_dir = cfg.sana.dit_ckpt or cfg.sana.get("ckpt_dir", "omni_ckpts/sana_video_2b_480p")

    auto_download_pretrained = bool(getattr(cfg.run, "auto_download_pretrained", True))
    auto_download_sana = bool(getattr(cfg.run, "auto_download_sana", auto_download_pretrained))
    auto_download_smol = bool(getattr(cfg.run, "auto_download_smolvlm2", auto_download_pretrained))
    smol_ckpt_path = str(cfg.model.student.text_encoder.ckpt_path)

    if auto_download_sana:
        ensure_sana_assets_available(
            sana_cfg=sana_cfg,
            sana_ckpt_dir=sana_ckpt_dir,
            is_main=is_main,
            local_rank=local_rank,
        )
    if auto_download_smol:
        ensure_smolvlm2_checkpoint_available(
            ckpt_path=smol_ckpt_path,
            is_main=is_main,
            local_rank=local_rank,
        )

    teacher = SanaTextEncoder(sana_cfg, device=device, dtype=dtype)

    student_cfg = cfg.model.student
    strict_sana_parity = bool(getattr(cfg.run, "strict_sana_parity_text_path", False))
    strict_fail_fast_mask = bool(getattr(cfg.run, "strict_fail_fast_mask", strict_sana_parity))
    sana_model_max_length = int(getattr(getattr(sana_cfg, "text_encoder", AttrDict()), "model_max_length", 300) or 300)
    chi_prompt_text = ""
    if bool(getattr(cfg.data.preprocessing, "use_chi_prompt", False)):
        chi_list = getattr(getattr(sana_cfg, "text_encoder", AttrDict()), "chi_prompt", None)
        if isinstance(chi_list, (list, tuple)) and len(chi_list) > 0:
            chi_prompt_text = "\n".join(str(x) for x in chi_list)

    student = build_teacher_free_student(
        cfg,
        device,
        dtype,
        strict_sana_parity_text_path=strict_sana_parity,
        strict_fail_fast_mask=strict_fail_fast_mask,
        sana_model_max_length=sana_model_max_length,
        sana_chi_prompt_text=chi_prompt_text,
    )

    if world_size > 1:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    student_module = student.module if isinstance(student, DDP) else student
    projector_cfg = cfg.model.student.get("projector", AttrDict())
    student_lora_cfg = cfg.model.student.get("lora", AttrDict())
    infer_hints = {
        "projector_type": str(getattr(projector_cfg, "type", "legacy")),
        "mcp_hidden_dim": int(getattr(projector_cfg, "mcp_hidden_dim", 512) or 512),
        "mcp_num_fuse_layers": int(getattr(projector_cfg, "mcp_num_fuse_layers", 4) or 4),
        "mcp_use_refine": bool(getattr(projector_cfg, "mcp_use_refine", False)),
        "mcp_refine_kernel_size": int(getattr(projector_cfg, "mcp_refine_kernel_size", 3) or 3),
        "student_lora_enable": bool(getattr(student_lora_cfg, "enable", False)),
        "student_lora_r": int(getattr(student_lora_cfg, "r", 0) or 0),
        "student_lora_alpha": int(getattr(student_lora_cfg, "alpha", 0) or 0),
        "student_text_train_top_layers": int(getattr(cfg.model.student.text_encoder, "train_top_layers", 0) or 0),
        "student_text_train_final_norm": bool(getattr(cfg.model.student.text_encoder, "train_final_norm", False)),
        "train_use_chi_prompt": bool(getattr(cfg.data.preprocessing, "use_chi_prompt", False)),
        "train_use_prompt_templates": False,
        "train_motion_score": int(getattr(cfg.data.preprocessing, "motion_score", 0) or 0),
        "strict_sana_parity_text_path": strict_sana_parity,
        "strict_fail_fast_mask": strict_fail_fast_mask,
        "sana_model_max_length": sana_model_max_length,
        "train_student_max_length": int(getattr(cfg.model.student.text_encoder, "max_length", 0) or 0),
        "stage1_prompt_only_distill": True,
        "teacher_name": str(getattr(getattr(sana_cfg, "text_encoder", AttrDict()), "text_encoder_name", "unknown")),
    }

    diffusion_model = None
    if cfg.loss.effect.enabled or cfg.loss.cfg.enabled:
        diffusion_model = load_sana_diffusion_model(
            sana_cfg=sana_cfg,
            sana_ckpt_dir=sana_ckpt_dir,
            device=device,
            dtype=dtype,
        )

    sources = []
    streaming = cfg.data.preprocessing.get("streaming", True)
    for source in cfg.data.sources:
        dataset = load_source_dataset(source, streaming=streaming)
        if cfg.data.batching.shuffle:
            if hasattr(dataset, "shuffle"):
                buffer_size = cfg.data.batching.get("shuffle_buffer", 10000)
                try:
                    dataset = dataset.shuffle(seed=cfg.run.seed, buffer_size=buffer_size)
                except TypeError:
                    dataset = dataset.shuffle(seed=cfg.run.seed)
        sources.append(
            {
                "dataset": dataset,
                "weight": source.weight,
                "candidates": source.text_field_candidates,
            }
        )

    prompt_dataset = PromptMixerDataset(sources, seed=cfg.run.seed, rank=rank, world_size=world_size)
    dataloader = DataLoader(
        prompt_dataset,
        batch_size=args.batch_size or cfg.data.batching.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else cfg.run.num_workers,
        drop_last=cfg.data.batching.drop_last,
        collate_fn=lambda x: x,
        pin_memory=True,
    )
    data_iter = iter(dataloader)

    total_steps = args.total_steps or cfg.train.total_steps
    grad_accum_steps = args.grad_accum_steps or cfg.data.batching.grad_accum_steps
    log_every = args.log_every or cfg.run.log_every
    save_every = args.save_every or cfg.run.save_every_steps

    bridge_params = [
        p for n, p in student_module.named_parameters()
        if p.requires_grad and not n.startswith("smolvlm2_model")
    ]
    text_params = [
        p for n, p in student_module.named_parameters()
        if p.requires_grad and n.startswith("smolvlm2_model")
    ]

    param_groups = []
    if bridge_params:
        param_groups.append({"params": bridge_params, "lr": cfg.train.lr.conditioner_bridge})
    if text_params:
        param_groups.append({"params": text_params, "lr": cfg.train.lr.text_encoder})

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(cfg.train.optimizer.betas),
        eps=cfg.train.optimizer.eps,
        weight_decay=cfg.train.optimizer.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.train.lr.warmup_steps:
            return float(step) / float(max(1, cfg.train.lr.warmup_steps))
        progress = float(step - cfg.train.lr.warmup_steps) / float(max(1, total_steps - cfg.train.lr.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.train.lr.min_lr_ratio + (1.0 - cfg.train.lr.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    rng = random.Random(cfg.run.seed)
    if world_size > 1 and dist.is_initialized():
        logger.info("Waiting for all ranks to finish init...")
        dist.barrier()
    if is_main:
        logger.info("Trainable params: %.2fM", sum(p.numel() for p in bridge_params + text_params) / 1e6)

    resume_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        state = ckpt.get("student_state", ckpt)
        if "smolvlm2_vision_head" in state and getattr(student_module, "smolvlm2_vision_head", None) is not None:
            student_module.smolvlm2_vision_head.load_state_dict(state["smolvlm2_vision_head"], strict=False)
        if "adapter" in state and hasattr(student_module, "adapter"):
            student_module.adapter.load_state_dict(state["adapter"], strict=False)
        if "adapter_output_norm" in state and hasattr(student_module, "adapter_output_norm"):
            student_module.adapter_output_norm.load_state_dict(state["adapter_output_norm"], strict=False)
        if "adapter_output_gate" in state:
            student_module.adapter_output_gate.data.copy_(state["adapter_output_gate"].to(student_module.adapter_output_gate.device))
        if "resampler" in state and hasattr(student_module, "resampler"):
            student_module.resampler.load_state_dict(state["resampler"], strict=False)
        if "projector" in state and getattr(student_module, "projector", None) is not None:
            student_module.projector.load_state_dict(state["projector"], strict=False)
        if "smolvlm2_text_trainable" in state:
            load_trainable_smolvlm2_state_dict(student_module.smolvlm2_model, state["smolvlm2_text_trainable"], is_main=is_main)
        if "smolvlm2_lora" in state:
            load_lora_state_dict(student_module.smolvlm2_model, state["smolvlm2_lora"], is_main=is_main)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        resume_step = int(ckpt.get("step", 0))
        if is_main:
            logger.info("Resumed from %s at step %d", args.resume_from, resume_step)

    calibrate = bool(getattr(cfg.loss.embed, "calibrate_to_teacher", False))
    calib_eps = float(getattr(cfg.loss.embed, "calibration_eps", 1e-6))

    optimizer.zero_grad(set_to_none=True)
    global_step = resume_step
    while global_step < total_steps:
        batch_prompts = next(data_iter)
        teacher_prompts, student_prompts = build_prompt_batch(batch_prompts, cfg, teacher, global_step, rng)

        with torch.no_grad():
            teacher_out = teacher.forward_chi(teacher_prompts, use_chi_prompt=cfg.data.preprocessing.use_chi_prompt)
            teacher_embeds = teacher_out["prompt_embeds"]
            teacher_mask = teacher_out["mask"]

        student_embeds_raw = student(student_prompts)

        teacher_embeds, teacher_mask = pad_or_trim_teacher(
            teacher_embeds, teacher_mask, student_embeds_raw.shape[1]
        )

        if calibrate:
            student_embeds_cal = calibrate_student_embeds(
                student_embeds_raw, teacher_embeds, teacher_mask, eps=calib_eps
            )
        else:
            student_embeds_cal = student_embeds_raw

        mask = teacher_mask if cfg.loss.embed.use_attention_mask else torch.ones_like(teacher_mask)
        mask = mask.unsqueeze(-1).to(student_embeds_raw.dtype)

        teacher_norm = F.layer_norm(teacher_embeds, (teacher_embeds.shape[-1],))
        student_norm = F.layer_norm(student_embeds_raw, (student_embeds_raw.shape[-1],))

        diff = (student_norm - teacher_norm) * mask
        denom = mask.sum() * student_norm.shape[-1]
        mse_loss = (diff ** 2).sum() / (denom + 1e-8)

        valid_mask = mask.squeeze(-1).bool()
        s_flat = student_norm[valid_mask]
        t_flat = teacher_norm[valid_mask]
        cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1).mean()
        cos_loss = 1.0 - cos_sim

        mask_sum = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        s_mean = (student_norm * mask).sum(dim=1, keepdim=True) / mask_sum
        t_mean = (teacher_norm * mask).sum(dim=1, keepdim=True) / mask_sum
        s_var = ((student_norm - s_mean) ** 2 * mask).sum(dim=1, keepdim=True) / mask_sum
        t_var = ((teacher_norm - t_mean) ** 2 * mask).sum(dim=1, keepdim=True) / mask_sum
        s_std = torch.sqrt(s_var + 1e-6)
        t_std = torch.sqrt(t_var + 1e-6)
        stat_loss = F.mse_loss(s_mean, t_mean) + F.mse_loss(s_std, t_std)

        pooled_student = masked_mean_pool(student_norm, mask.squeeze(-1))
        pooled_teacher = masked_mean_pool(teacher_norm, mask.squeeze(-1))

        rel_loss = torch.tensor(0.0, device=device)
        rel_weight = float(getattr(cfg.loss.embed, "rel_weight", 0.0) or 0.0)
        if rel_weight > 0.0 and student_norm.shape[0] > 1:
            pooled_student_norm = F.normalize(pooled_student, dim=-1)
            pooled_teacher_norm = F.normalize(pooled_teacher, dim=-1)
            sim_student = pooled_student_norm @ pooled_student_norm.T
            sim_teacher = pooled_teacher_norm @ pooled_teacher_norm.T
            rel_loss = F.mse_loss(sim_student, sim_teacher)

        nce_loss = torch.tensor(0.0, device=device)
        nce_weight = float(getattr(cfg.loss.embed, "nce_weight", 0.0) or 0.0)
        nce_temp = float(getattr(cfg.loss.embed, "nce_temperature", 0.07) or 0.07)
        if nce_weight > 0.0 and student_norm.shape[0] > 1:
            nce_loss = info_nce_loss(pooled_student, pooled_teacher, temperature=nce_temp)

        loss = (
            cfg.loss.embed.mse_weight * mse_loss
            + cfg.loss.embed.cos_weight * cos_loss
            + cfg.loss.embed.stat_weight * stat_loss
            + rel_weight * rel_loss
            + nce_weight * nce_loss
        )

        effect_loss = torch.tensor(0.0, device=device)
        cfg_loss = torch.tensor(0.0, device=device)
        if diffusion_model is not None:
            latent_cfg = cfg.sana.latent
            base_batch = student_embeds_cal.shape[0]

            if cfg.loss.effect.enabled:
                effect_bs = get_distill_batch_size(cfg.loss.effect, base_batch)
                effect_shape = (
                    effect_bs,
                    latent_cfg.channels,
                    latent_cfg.frames,
                    latent_cfg.height,
                    latent_cfg.width,
                )
                data_info = build_data_info(effect_shape[0], latent_cfg.height, latent_cfg.width, device=device)

                teacher_y = teacher_embeds[:effect_bs].unsqueeze(1)
                student_y = student_embeds_cal[:effect_bs].unsqueeze(1)
                effect_mask = teacher_mask[:effect_bs] if teacher_mask is not None else None

                effect_losses = []
                timesteps_list = sample_effect_timesteps(cfg, effect_shape[0], device)
                for timesteps in timesteps_list:
                    latents = torch.randn(effect_shape, device=device, dtype=dtype)
                    with torch.no_grad():
                        pred_teacher = diffusion_model(
                            latents, timesteps, teacher_y, mask=effect_mask, data_info=data_info
                        )
                    pred_student = diffusion_model(
                        latents, timesteps, student_y, mask=effect_mask, data_info=data_info
                    )
                    if isinstance(pred_teacher, (tuple, list)):
                        pred_teacher = pred_teacher[0]
                    if isinstance(pred_student, (tuple, list)):
                        pred_student = pred_student[0]
                    effect_losses.append(F.mse_loss(pred_student, pred_teacher))
                effect_loss = torch.stack(effect_losses).mean()
                effect_weight = get_effect_weight(cfg, global_step)
                loss = loss + effect_weight * effect_loss

            if cfg.loss.cfg.enabled:
                cfg_bs = get_distill_batch_size(cfg.loss.cfg, base_batch)
                cfg_shape = (
                    cfg_bs,
                    latent_cfg.channels,
                    latent_cfg.frames,
                    latent_cfg.height,
                    latent_cfg.width,
                )
                latents = torch.randn(cfg_shape, device=device, dtype=dtype)
                timesteps = torch.randint(
                    0,
                    cfg.sana.timesteps.num_train_timesteps,
                    (cfg_shape[0],),
                    device=device,
                )
                data_info = build_data_info(cfg_shape[0], latent_cfg.height, latent_cfg.width, device=device)

                teacher_y = teacher_embeds[:cfg_bs].unsqueeze(1)
                student_y = student_embeds_cal[:cfg_bs].unsqueeze(1)
                cond_mask = teacher_mask[:cfg_bs] if teacher_mask is not None else None

                neg_prompt = cfg.loss.cfg.negative_prompt
                neg_prompts = [neg_prompt] * cfg_bs
                with torch.no_grad():
                    teacher_neg = teacher.forward_chi(neg_prompts, use_chi_prompt=False)
                student_neg_raw = student(neg_prompts)
                teacher_neg_embeds, teacher_neg_mask = pad_or_trim_teacher(
                    teacher_neg["prompt_embeds"], teacher_neg["mask"], student_neg_raw.shape[1]
                )
                if calibrate:
                    student_neg = calibrate_student_embeds(
                        student_neg_raw, teacher_neg_embeds, teacher_neg_mask, eps=calib_eps
                    )
                else:
                    student_neg = student_neg_raw

                teacher_neg_y = teacher_neg_embeds.unsqueeze(1)
                student_neg_y = student_neg.unsqueeze(1)

                guidance_scale = random.uniform(
                    cfg.loss.cfg.guidance_scale_sampler.min,
                    cfg.loss.cfg.guidance_scale_sampler.max,
                )
                with torch.no_grad():
                    teacher_guided = compute_guided_pred(
                        diffusion_model,
                        latents,
                        timesteps,
                        teacher_y,
                        teacher_neg_y,
                        cond_mask,
                        teacher_neg_mask,
                        data_info,
                        guidance_scale,
                    )
                student_guided = compute_guided_pred(
                    diffusion_model,
                    latents,
                    timesteps,
                    student_y,
                    student_neg_y,
                    cond_mask,
                    teacher_neg_mask,
                    data_info,
                    guidance_scale,
                )
                cfg_loss = F.mse_loss(student_guided, teacher_guided)
                loss = loss + cfg.loss.cfg.weight * cfg_loss

        loss = loss / grad_accum_steps
        loss.backward()

        if (global_step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(bridge_params + text_params, cfg.train.lr.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if is_main and global_step == 0:
            logger.info(
                "Shapes | teacher: %s mask: %s student: %s",
                tuple(teacher_embeds.shape),
                tuple(teacher_mask.shape),
                tuple(student_embeds_cal.shape),
            )
            logger.info("Sample prompt (student): %s", student_prompts[0][:160])

        if is_main and log_every and (global_step + 1) % log_every == 0:
            with torch.no_grad():
                raw_ratio = (
                    masked_mean_norm(student_embeds_raw, teacher_mask)
                    / masked_mean_norm(teacher_embeds, teacher_mask)
                ).mean().item()
                cal_ratio = (
                    masked_mean_norm(student_embeds_cal, teacher_mask)
                    / masked_mean_norm(teacher_embeds, teacher_mask)
                ).mean().item()
            logger.info(
                "Step %d | loss=%.6f (mse=%.6f cos=%.6f stat=%.6f rel=%.6f nce=%.6f effect=%.6f cfg=%.6f norm_ratio_raw=%.3f norm_ratio_cal=%.3f)",
                global_step + 1,
                loss.item() * grad_accum_steps,
                mse_loss.item(),
                cos_loss.item(),
                stat_loss.item(),
                rel_loss.item(),
                nce_loss.item(),
                effect_loss.item(),
                cfg_loss.item(),
                raw_ratio,
                cal_ratio,
            )

        if is_main and save_every and (global_step + 1) % save_every == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_step{global_step + 1}.pt")
            student_state = build_student_checkpoint_state(student, student_module)
            torch.save(
                {
                    "step": global_step + 1,
                    "student_state": student_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "infer_hints": infer_hints,
                    "stage_name": "stage1_prompt_only_distill",
                },
                ckpt_path,
            )
            logger.info("Saved checkpoint: %s", ckpt_path)

        global_step += 1

    if is_main:
        final_ckpt = os.path.join(run_dir, "checkpoint_final.pt")
        student_state = build_student_checkpoint_state(student, student_module)
        torch.save(
            {
                "step": global_step,
                "student_state": student_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "infer_hints": infer_hints,
                "stage_name": "stage1_prompt_only_distill",
            },
            final_ckpt,
        )

        with open(os.path.join(run_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        logger.info("Training completed. Final checkpoint: %s", final_ckpt)

    if dist.is_initialized():
        dist.destroy_process_group()


def legacy_train(args: argparse.Namespace):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    prompts = load_prompts_from_csv(args.csv_path, args.num_samples, args.seed, args.shuffle)
    if args.prompt_index is not None:
        if not prompts:
            raise ValueError("No prompts available to select by index")
        prompt = prompts[args.prompt_index % len(prompts)]
        prompts = [prompt]
    dataset = PromptDataset(prompts)
    drop_last = len(dataset) > args.batch_size
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=drop_last)

    try:
        sana_cfg = load_sana_config(args.sana_config)
    except Exception as exc:
        raise RuntimeError(f"Failed to load SANA config from {args.sana_config}: {exc}") from exc

    teacher = SanaTextEncoder(sana_cfg, device=device, dtype=dtype)

    student = SanaPromptBridge(
        smolvlm2_ckpt_path=args.smolvlm2_ckpt,
        adapter_ckpt_dir=args.adapter_ckpt_dir,
        adapter_in_channels=1152,
        adapter_out_channels=4096,
        adapter_query_length=64,
        smol_vh_num_queries=1,
        num_prompt_queries=sana_cfg.text_encoder.model_max_length,
        caption_channels=getattr(sana_cfg.text_encoder, "caption_channels", 2304),
        precision_dtype=dtype,
        device=device,
        force_adapter_query_length=args.force_adapter_query_length,
    )

    for param in student.smolvlm2_model.parameters():
        param.requires_grad = False

    diffusion_model = None
    if args.effect_weight > 0.0:
        diffusion_model = load_sana_diffusion_model(
            sana_cfg=sana_cfg,
            sana_ckpt_dir=args.sana_ckpt_dir,
            device=device,
            dtype=dtype,
        )
        logger.info("Effect distill enabled (weight=%.3f)", args.effect_weight)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    logger.info("Trainable params: %.2fM", sum(p.numel() for p in trainable_params) / 1e6)
    logger.info("Adapter query length: %s", student.adapter.learnable_query_length)
    smol_cfg = getattr(student.smolvlm2_model, "config", None)
    smol_hidden = None
    if smol_cfg is not None and hasattr(smol_cfg, "text_config") and smol_cfg.text_config is not None:
        smol_hidden = getattr(smol_cfg.text_config, "hidden_size", None)
    if smol_hidden is None:
        smol_hidden = getattr(smol_cfg, "hidden_size", "unknown") if smol_cfg is not None else "unknown"
    logger.info("SmolVLM2 hidden_size: %s", smol_hidden)

    calibrate = bool(args.calibrate_to_teacher)
    calib_eps = float(args.calibration_eps)

    total_steps = 0
    for epoch in range(args.epochs):
        for batch_prompts in dataloader:
            total_steps += 1
            if args.max_steps and total_steps > args.max_steps:
                break

            processed_prompts = apply_motion_and_chi(
                list(batch_prompts),
                sana_cfg,
                args.motion_score,
                args.use_chi_prompt,
            )

            with torch.no_grad():
                teacher_out = teacher.forward_chi(processed_prompts, use_chi_prompt=False)
                teacher_embeds = teacher_out["prompt_embeds"]
                teacher_mask = teacher_out["mask"]

            student_embeds_raw = student(processed_prompts)

            teacher_embeds, teacher_mask = pad_or_trim_teacher(
                teacher_embeds, teacher_mask, student_embeds_raw.shape[1]
            )

            if calibrate:
                student_embeds = calibrate_student_embeds(
                    student_embeds_raw, teacher_embeds, teacher_mask, eps=calib_eps
                )
            else:
                student_embeds = student_embeds_raw

            teacher_norm = F.layer_norm(teacher_embeds, (teacher_embeds.shape[-1],))
            student_norm = F.layer_norm(student_embeds, (student_embeds.shape[-1],))

            mask = teacher_mask.unsqueeze(-1).to(student_norm.dtype)
            diff = (student_norm - teacher_norm) * mask
            denom = mask.sum() * student_norm.shape[-1]
            mse_loss = (diff ** 2).sum() / (denom + 1e-8)

            valid_mask = mask.squeeze(-1).bool()
            s_flat = student_norm[valid_mask]
            t_flat = teacher_norm[valid_mask]
            cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1).mean()
            cos_loss = 1.0 - cos_sim

            mask_sum = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            s_mean = (student_norm * mask).sum(dim=1, keepdim=True) / mask_sum
            t_mean = (teacher_norm * mask).sum(dim=1, keepdim=True) / mask_sum
            s_var = ((student_norm - s_mean) ** 2 * mask).sum(dim=1, keepdim=True) / mask_sum
            t_var = ((teacher_norm - t_mean) ** 2 * mask).sum(dim=1, keepdim=True) / mask_sum
            s_std = torch.sqrt(s_var + 1e-6)
            t_std = torch.sqrt(t_var + 1e-6)
            stat_loss = F.mse_loss(s_mean, t_mean) + F.mse_loss(s_std, t_std)

            loss = mse_loss + args.cos_weight * cos_loss + args.stat_weight * stat_loss

            effect_loss = torch.tensor(0.0, device=device)
            if diffusion_model is not None:
                effect_height = args.effect_height or sana_cfg.model.image_size
                effect_width = args.effect_width or sana_cfg.model.image_size
                latent_h = effect_height // sana_cfg.vae.vae_downsample_rate
                latent_w = effect_width // sana_cfg.vae.vae_downsample_rate
                vae_stride = getattr(sana_cfg.vae, "vae_stride", [1, sana_cfg.vae.vae_downsample_rate])
                vae_stride_t = vae_stride[0] if isinstance(vae_stride, list) and len(vae_stride) >= 1 else 1
                latent_t = int(args.effect_num_frames - 1) // vae_stride_t + 1
                latent_shape = (
                    student_embeds.shape[0],
                    sana_cfg.vae.vae_latent_dim,
                    latent_t,
                    latent_h,
                    latent_w,
                )
                latents = torch.randn(latent_shape, device=device, dtype=dtype)
                timestep = torch.randint(0, 1000, (latent_shape[0],), device=device)
                teacher_y = teacher_embeds.unsqueeze(1)
                student_y = student_embeds.unsqueeze(1)
                data_info = {"img_hw": torch.tensor([[effect_height, effect_width]], device=device, dtype=torch.float)}

                with torch.no_grad():
                    pred_teacher = diffusion_model(latents, timestep, teacher_y, mask=teacher_mask, data_info=data_info)
                pred_student = diffusion_model(latents, timestep, student_y, mask=teacher_mask, data_info=data_info)

                if isinstance(pred_teacher, (tuple, list)):
                    pred_teacher = pred_teacher[0]
                if isinstance(pred_student, (tuple, list)):
                    pred_student = pred_student[0]

                effect_loss = F.mse_loss(pred_student, pred_teacher)
                effect_weight = args.effect_weight
                if args.effect_warmup_steps > 0 and total_steps <= args.effect_warmup_steps:
                    effect_weight = 0.0
                elif args.effect_ramp_steps > 0 and total_steps > args.effect_warmup_steps:
                    ramp_step = min(
                        args.effect_ramp_steps,
                        total_steps - args.effect_warmup_steps,
                    )
                    effect_weight = effect_weight * (ramp_step / args.effect_ramp_steps)
                loss = loss + effect_weight * effect_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if total_steps == 1:
                logger.info(
                    "Shapes | teacher: %s mask: %s student: %s",
                    tuple(teacher_embeds.shape),
                    tuple(teacher_mask.shape),
                    tuple(student_embeds.shape),
                )
                logger.info("Sample prompt (processed): %s", processed_prompts[0][:160])
            if total_steps % args.log_every == 0:
                with torch.no_grad():
                    raw_ratio = (
                        masked_mean_norm(student_embeds_raw, teacher_mask)
                        / masked_mean_norm(teacher_embeds, teacher_mask)
                    ).mean().item()
                    cal_ratio = (
                        masked_mean_norm(student_embeds, teacher_mask)
                        / masked_mean_norm(teacher_embeds, teacher_mask)
                    ).mean().item()
                logger.info(
                    "Epoch %d Step %d | loss=%.6f (mse=%.6f cos=%.6f stat=%.6f effect=%.6f norm_ratio_raw=%.3f norm_ratio_cal=%.3f)",
                    epoch + 1,
                    total_steps,
                    loss.item(),
                    mse_loss.item(),
                    cos_loss.item(),
                    stat_loss.item(),
                    effect_loss.item(),
                    raw_ratio,
                    cal_ratio,
                )

            if total_steps % args.save_every == 0:
                ckpt_path = os.path.join(run_dir, f"checkpoint_step{total_steps}.pt")
                student_state = {
                    "smolvlm2_vision_head": student.smolvlm2_vision_head.state_dict(),
                    "adapter": student.adapter.state_dict(),
                    "adapter_output_norm": student.adapter_output_norm.state_dict(),
                    "adapter_output_gate": student.adapter_output_gate.detach().cpu(),
                    "resampler": student.resampler.state_dict(),
                }
                torch.save(
                    {
                        "step": total_steps,
                        "epoch": epoch + 1,
                        "student_state": student_state,
                    },
                    ckpt_path,
                )
                logger.info("Saved checkpoint: %s", ckpt_path)

        if args.max_steps and total_steps > args.max_steps:
            break

    final_ckpt = os.path.join(run_dir, "checkpoint_final.pt")
    student_state = {
        "smolvlm2_vision_head": student.smolvlm2_vision_head.state_dict(),
        "adapter": student.adapter.state_dict(),
        "adapter_output_norm": student.adapter_output_norm.state_dict(),
        "adapter_output_gate": student.adapter_output_gate.detach().cpu(),
        "resampler": student.resampler.state_dict(),
    }
    torch.save(
        {
            "step": total_steps,
            "epoch": epoch + 1,
            "student_state": student_state,
        },
        final_ckpt,
    )

    with open(os.path.join(run_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Training completed. Final checkpoint: %s", final_ckpt)


def main():
    parser = argparse.ArgumentParser(description="Q1 distillation for Mobile-OV-SANA bridge")
    parser.add_argument("--config", type=str, help="Stage1 YAML config path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--precision", type=str, default=None, help="Precision override (bf16/fp16/fp32)")
    parser.add_argument("--max-gpus", type=int, default=2, help="Maximum GPUs to use for DDP")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Grad accumulation steps")
    parser.add_argument("--total-steps", type=int, default=None, help="Total steps override")
    parser.add_argument("--num-workers", type=int, default=None, help="Data loader workers override")
    parser.add_argument("--log-every", type=int, default=None, help="Logging interval override")
    parser.add_argument("--save-every", type=int, default=None, help="Checkpoint interval override")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from checkpoint")

    parser.add_argument("--csv-path", type=str, required=False, help="OpenVid-1M CSV path")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of prompts to use")
    parser.add_argument("--prompt-index", type=int, default=None, help="Optional single prompt index to overfit")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps override")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle prompts before sampling")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"], help="Training dtype")
    parser.add_argument("--adapter-ckpt-dir", type=str, default="omni_ckpts/wan/wanxiang1_3b/adapter")
    parser.add_argument("--smolvlm2-ckpt", type=str, default="omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt")
    parser.add_argument("--sana-config", type=str, default="configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml")
    parser.add_argument("--sana-ckpt-dir", type=str, default="omni_ckpts/sana_video_2b_480p")
    parser.add_argument("--force-adapter-query-length", type=int, default=64, help="Force adapter query length")
    parser.add_argument("--use-chi-prompt", action="store_true", help="Use SANA chi_prompt in teacher encoder")
    parser.add_argument("--motion-score", type=int, default=0, help="Append motion score to prompts (match SANA)")
    parser.add_argument("--cos-weight", type=float, default=0.1, help="Cosine loss weight")
    parser.add_argument("--stat-weight", type=float, default=0.1, help="Mean/std matching weight")
    parser.add_argument("--calibrate-to-teacher", action="store_true", help="Calibrate student embeds to teacher stats")
    parser.add_argument("--calibration-eps", type=float, default=1.0e-6, help="Epsilon for calibration")
    parser.add_argument("--effect-weight", type=float, default=0.0, help="Effect distill loss weight")
    parser.add_argument("--effect-warmup-steps", type=int, default=0, help="Steps to train embed-only before effect loss")
    parser.add_argument("--effect-ramp-steps", type=int, default=0, help="Steps to ramp effect loss to full weight")
    parser.add_argument("--effect-num-frames", type=int, default=16, help="Frames for effect distill (latent)")
    parser.add_argument("--effect-height", type=int, default=None, help="Effect distill height override")
    parser.add_argument("--effect-width", type=int, default=None, help="Effect distill width override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.config:
        cfg = load_stage1_config(args.config)
        train_with_config(cfg, args)
        return

    if not args.csv_path:
        raise ValueError("Either --config or --csv-path must be provided")

    if args.output_dir is None:
        args.output_dir = "output/q1_sana_bridge"
    if args.device is None:
        args.device = "cuda:0"
    if args.log_every is None:
        args.log_every = 50
    if args.save_every is None:
        args.save_every = 500

    legacy_train(args)


if __name__ == "__main__":
    main()
