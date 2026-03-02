#!/usr/bin/env python3
"""Precompute SANA teacher text embeddings for OpenVid samples with multi-GPU torchrun."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from dataclasses import fields, is_dataclass
from typing import get_args, get_origin
from torch.utils.data import DataLoader, Subset

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sana_root = project_root / "nets" / "third_party" / "sana"
if str(sana_root) not in sys.path:
    sys.path.insert(0, str(sana_root))

from diffusion.utils.config import SanaVideoConfig
from nets.omni.datasets.openvid_dataset import OpenVidDataset, openvid_collate_fn
from nets.third_party.sana.diffusion.longsana.utils.model_wrapper import SanaTextEncoder
from tools.train_q1_sana_bridge import init_distributed, normalize_prompt, truncate_prompt

logger = logging.getLogger(__name__)


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


def load_stage1_config(path: str) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return to_attrdict(cfg)


def load_sana_config(path: str) -> SanaVideoConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _dataclass_from_dict(SanaVideoConfig, cfg)


def preprocess_prompts(prompts: List[str], cfg: AttrDict, tokenizer) -> List[str]:
    prep_cfg = cfg.data.get("preprocessing", AttrDict())
    normalize_whitespace = bool(getattr(prep_cfg, "normalize_whitespace", True))
    strip = bool(getattr(prep_cfg, "strip", True))
    remove_double_newlines = bool(getattr(prep_cfg, "remove_double_newlines", True))
    max_prompt_tokens = getattr(prep_cfg, "max_prompt_tokens", None)
    use_templates = bool(getattr(prep_cfg, "use_prompt_templates", False))
    templates = list(cfg.data.get("prompt_templates", []) or [])
    motion_score = int(cfg.data.get("motion_score", 10))

    out = []
    for p in prompts:
        text = normalize_prompt(p, normalize_whitespace, strip, remove_double_newlines)
        if max_prompt_tokens is not None:
            text = truncate_prompt(text, tokenizer, int(max_prompt_tokens))
        # Precompute must be deterministic: use only the first template when enabled.
        if use_templates and templates:
            text = templates[0].format(prompt=text, motion_score=motion_score)
        out.append(text)
    return out


def flush_shard(
    shard_items: List[Dict[str, torch.Tensor]],
    output_dir: str,
    rank: int,
    shard_id: int,
) -> Dict[str, Any]:
    sample_idx = torch.cat([x["sample_idx"] for x in shard_items], dim=0)
    prompt_embeds = torch.cat([x["prompt_embeds"] for x in shard_items], dim=0)
    mask = torch.cat([x["mask"] for x in shard_items], dim=0)

    file_name = f"teacher_rank{rank:02d}_shard{shard_id:05d}.pt"
    out_path = os.path.join(output_dir, file_name)
    torch.save(
        {
            "sample_idx": sample_idx,
            "prompt_embeds": prompt_embeds,
            "mask": mask,
        },
        out_path,
    )
    return {
        "file": file_name,
        "count": int(sample_idx.shape[0]),
        "sample_idx_min": int(sample_idx.min().item()),
        "sample_idx_max": int(sample_idx.max().item()),
    }


def safe_barrier(local_rank: int) -> None:
    if not torch.distributed.is_initialized():
        return
    try:
        torch.distributed.barrier(device_ids=[local_rank])
    except TypeError:
        torch.distributed.barrier()


def main():
    parser = argparse.ArgumentParser(description="Precompute SANA teacher embeddings for OpenVid")
    parser.add_argument("--config", type=str, required=True, help="Stage1 config YAML")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--shard-size", type=int, default=512)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    rank, world_size, local_rank = init_distributed(args.max_gpus)
    is_main = rank == 0

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    cfg = load_stage1_config(args.config)
    sana_cfg = load_sana_config(cfg.sana.config)
    teacher = SanaTextEncoder(sana_cfg, device=device, dtype=dtype)

    openvid_cfg = cfg.data.openvid
    dataset = OpenVidDataset(
        csv_path=openvid_cfg.csv_path,
        video_dir=openvid_cfg.video_dir,
        preprocessed_dir=openvid_cfg.preprocessed_dir,
        use_preprocessed=openvid_cfg.use_preprocessed,
        max_samples=args.max_samples if args.max_samples is not None else openvid_cfg.get("max_samples"),
    )

    all_indices = list(range(rank, len(dataset), world_size))
    subset = Subset(dataset, all_indices)

    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=openvid_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if is_main:
        logger.info(
            "Precompute start | world_size=%d dataset=%d subset_per_rank~%d output=%s",
            world_size,
            len(dataset),
            len(subset),
            args.output_dir,
        )

    shard_items: List[Dict[str, torch.Tensor]] = []
    shard_id = 0
    shard_meta: List[Dict[str, Any]] = []
    total_written = 0
    use_chi_prompt = bool(getattr(cfg.data.get("preprocessing", AttrDict()), "use_chi_prompt", False))

    for batch_idx, batch in enumerate(dataloader):
        prompts = preprocess_prompts(batch["prompt"], cfg, teacher.tokenizer)
        sample_idx = batch["sample_idx"].view(-1).long()

        with torch.no_grad():
            teacher_out = teacher.forward_chi(prompts, use_chi_prompt=use_chi_prompt)
            embeds = teacher_out["prompt_embeds"].detach().to(device="cpu", dtype=torch.float16)
            mask = teacher_out["mask"].detach().to(device="cpu", dtype=torch.uint8)

        shard_items.append(
            {
                "sample_idx": sample_idx.cpu(),
                "prompt_embeds": embeds,
                "mask": mask,
            }
        )

        buffered = sum(x["sample_idx"].shape[0] for x in shard_items)
        if buffered >= args.shard_size:
            meta = flush_shard(shard_items, args.output_dir, rank, shard_id)
            shard_meta.append(meta)
            total_written += meta["count"]
            shard_items = []
            shard_id += 1

        if batch_idx % 50 == 0:
            logger.info("Rank %d progress batch=%d written=%d", rank, batch_idx, total_written + buffered)

    if shard_items:
        meta = flush_shard(shard_items, args.output_dir, rank, shard_id)
        shard_meta.append(meta)
        total_written += meta["count"]

    rank_manifest = {
        "rank": rank,
        "world_size": world_size,
        "total_written": total_written,
        "shards": shard_meta,
    }
    rank_manifest_path = os.path.join(args.output_dir, f"manifest_rank{rank:02d}.json")
    with open(rank_manifest_path, "w", encoding="utf-8") as f:
        json.dump(rank_manifest, f, indent=2)

    if world_size > 1 and torch.distributed.is_initialized():
        safe_barrier(local_rank)

    if is_main:
        all_rank_manifests = []
        for r in range(world_size):
            path = os.path.join(args.output_dir, f"manifest_rank{r:02d}.json")
            with open(path, "r", encoding="utf-8") as f:
                all_rank_manifests.append(json.load(f))

        merged_shards: List[Dict[str, Any]] = []
        total = 0
        for rm in all_rank_manifests:
            for s in rm.get("shards", []):
                merged_shards.append(s)
                total += int(s.get("count", 0))

        merged = {
            "created_at": datetime.now().isoformat(),
            "config": os.path.abspath(args.config),
            "dataset_size": len(dataset),
            "world_size": world_size,
            "dtype": "float16",
            "shards": merged_shards,
            "total_samples": total,
            "embedding_shape": "[N, 300, 2304] (depends on teacher config)",
        }
        manifest_path = os.path.join(args.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        logger.info("Done. manifest=%s total_samples=%d shards=%d", manifest_path, total, len(merged_shards))

    if world_size > 1 and torch.distributed.is_initialized():
        safe_barrier(local_rank)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
