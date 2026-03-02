#!/usr/bin/env python3
"""Sanity checks for Stage-1 distillation without affecting training."""

import argparse
import statistics as st
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F

from nets.omni.modules.sana_prompt_bridge import SanaPromptBridge
from nets.third_party.sana.diffusion.longsana.utils.model_wrapper import SanaTextEncoder
from tools.train_q1_sana_bridge import (
    apply_motion_and_chi,
    build_data_info,
    compute_guided_pred,
    load_sana_config,
    load_sana_diffusion_model,
    pad_or_trim_teacher,
)


def parse_indices(spec: str, max_len: int) -> List[int]:
    if not spec:
        return list(range(min(10, max_len)))
    indices: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                start, end = end, start
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return [i for i in indices if 0 <= i < max_len]


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill sanity checks")
    parser.add_argument("--bridge-ckpt", required=True, help="Student bridge checkpoint")
    parser.add_argument(
        "--sana-config",
        default="configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml",
        help="SANA config path",
    )
    parser.add_argument(
        "--sana-ckpt-dir",
        default="omni_ckpts/sana_video_2b_480p",
        help="SANA checkpoint dir",
    )
    parser.add_argument("--csv-path", default="data/openvid_q1/OpenVid_prompt_subset.csv")
    parser.add_argument(
        "--prompt-indices",
        default="0-9",
        help="Comma list or ranges (e.g. 0-9,12,15)",
    )
    parser.add_argument("--motion-score", type=int, default=10)
    parser.add_argument("--use-chi-prompt", action="store_true")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--effect-batch-size", type=int, default=1)
    parser.add_argument("--cfg-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--latent-frames", type=int, default=17)
    parser.add_argument("--latent-height", type=int, default=32)
    parser.add_argument("--latent-width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    sana_cfg = load_sana_config(args.sana_config)
    teacher = SanaTextEncoder(sana_cfg, device=device, dtype=dtype)
    diffusion_model = load_sana_diffusion_model(
        sana_cfg=sana_cfg,
        sana_ckpt_dir=args.sana_ckpt_dir,
        device=device,
        dtype=dtype,
    )
    diffusion_model.eval()

    bridge = SanaPromptBridge(
        smolvlm2_ckpt_path="omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt",
        adapter_ckpt_dir="omni_ckpts/wan/wanxiang1_3b/adapter",
        adapter_in_channels=1152,
        adapter_out_channels=4096,
        adapter_query_length=64,
        smol_vh_num_queries=1,
        num_prompt_queries=sana_cfg.text_encoder.model_max_length,
        caption_channels=getattr(sana_cfg.text_encoder, "caption_channels", 2304),
        precision_dtype=dtype,
        device=device,
        tokenizer_model_id="HuggingFaceTB/SmolVLM-Instruct",
        force_adapter_query_length=256,
        max_length=512,
    )

    ckpt = torch.load(args.bridge_ckpt, map_location="cpu")
    state = ckpt.get("student_state", ckpt)
    bridge.smolvlm2_vision_head.load_state_dict(state["smolvlm2_vision_head"])
    bridge.adapter.load_state_dict(state["adapter"], strict=False)
    bridge.adapter_output_norm.load_state_dict(state["adapter_output_norm"])
    bridge.adapter_output_gate.data.copy_(state["adapter_output_gate"])
    bridge.resampler.load_state_dict(state["resampler"])

    df = pd.read_csv(args.csv_path)
    prompts = df["caption"].dropna().astype(str).tolist()
    indices = parse_indices(args.prompt_indices, len(prompts))
    if not indices:
        raise ValueError("No valid prompt indices found")

    base_prompts = [prompts[i] for i in indices]
    teacher_prompts = apply_motion_and_chi(
        base_prompts, sana_cfg, args.motion_score, use_chi_prompt=False
    )
    student_prompts = apply_motion_and_chi(
        base_prompts, sana_cfg, args.motion_score, use_chi_prompt=args.use_chi_prompt
    )

    with torch.no_grad():
        teacher_out = teacher.forward_chi(teacher_prompts, use_chi_prompt=args.use_chi_prompt)
        teacher_embeds = teacher_out["prompt_embeds"]
        teacher_mask = teacher_out["mask"]
        student_embeds = bridge(student_prompts)

    teacher_embeds, teacher_mask = pad_or_trim_teacher(
        teacher_embeds, teacher_mask, student_embeds.shape[1]
    )

    # Norm/cos stats (layernorm cos, raw norm)
    teacher_ln = F.layer_norm(teacher_embeds, (teacher_embeds.shape[-1],))
    student_ln = F.layer_norm(student_embeds, (student_embeds.shape[-1],))
    mask = teacher_mask.float()
    mask_sum = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    cos_list = []
    norm_ratio_list = []
    token_counts = []
    for i in range(teacher_embeds.shape[0]):
        valid = mask[i].bool()
        cos = F.cosine_similarity(student_ln[i, valid], teacher_ln[i, valid], dim=-1).mean()

        t_norm = teacher_embeds[i].norm(dim=-1)
        s_norm = student_embeds[i].norm(dim=-1)
        t_mean = (t_norm * mask[i]).sum() / mask_sum[i].squeeze(0)
        s_mean = (s_norm * mask[i]).sum() / mask_sum[i].squeeze(0)
        ratio = (s_mean / t_mean).item() if t_mean > 0 else float("inf")

        cos_list.append(float(cos))
        norm_ratio_list.append(float(ratio))
        token_counts.append(int(mask_sum[i].item()))

    print("Prompt indices:", indices)
    print("Avg tokens (non-pad):", sum(token_counts) / len(token_counts))
    print("Cosine (LN) mean:", st.mean(cos_list), "std:", st.pstdev(cos_list))
    print("Norm ratio (student/teacher) mean:", st.mean(norm_ratio_list), "std:", st.pstdev(norm_ratio_list))

    # Copy-teacher effect + CFG sanity
    effect_bs = min(args.effect_batch_size, teacher_embeds.shape[0])
    cfg_bs = min(args.cfg_batch_size, teacher_embeds.shape[0])

    latent_shape = (
        effect_bs,
        args.latent_channels,
        args.latent_frames,
        args.latent_height,
        args.latent_width,
    )
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    timesteps = torch.randint(0, 1000, (effect_bs,), device=device)
    data_info = build_data_info(effect_bs, args.latent_height, args.latent_width, device=device)

    teacher_y = teacher_embeds[:effect_bs].unsqueeze(1)
    effect_mask = teacher_mask[:effect_bs] if teacher_mask is not None else None

    with torch.no_grad():
        pred_teacher = diffusion_model(latents, timesteps, teacher_y, mask=effect_mask, data_info=data_info)
        pred_copy = diffusion_model(latents, timesteps, teacher_y, mask=effect_mask, data_info=data_info)
    if isinstance(pred_teacher, (tuple, list)):
        pred_teacher = pred_teacher[0]
    if isinstance(pred_copy, (tuple, list)):
        pred_copy = pred_copy[0]
    effect_copy_mse = F.mse_loss(pred_copy, pred_teacher).item()
    print("Copy-teacher effect MSE:", effect_copy_mse)

    # CFG sanity (guided output with teacher vs teacher)
    cfg_shape = (
        cfg_bs,
        args.latent_channels,
        args.latent_frames,
        args.latent_height,
        args.latent_width,
    )
    latents = torch.randn(cfg_shape, device=device, dtype=dtype)
    timesteps = torch.randint(0, 1000, (cfg_bs,), device=device)
    data_info = build_data_info(cfg_bs, args.latent_height, args.latent_width, device=device)

    teacher_y = teacher_embeds[:cfg_bs].unsqueeze(1)
    cond_mask = teacher_mask[:cfg_bs] if teacher_mask is not None else None
    neg_prompts = [args.negative_prompt] * cfg_bs
    with torch.no_grad():
        teacher_neg = teacher.forward_chi(neg_prompts, use_chi_prompt=False)
    teacher_neg_embeds, teacher_neg_mask = pad_or_trim_teacher(
        teacher_neg["prompt_embeds"], teacher_neg["mask"], teacher_embeds.shape[1]
    )
    teacher_neg_y = teacher_neg_embeds.unsqueeze(1)

    for gs in (1.0, 5.0):
        with torch.no_grad():
            guided_teacher = compute_guided_pred(
                diffusion_model,
                latents,
                timesteps,
                teacher_y,
                teacher_neg_y,
                cond_mask,
                teacher_neg_mask,
                data_info,
                gs,
            )
            guided_copy = compute_guided_pred(
                diffusion_model,
                latents,
                timesteps,
                teacher_y,
                teacher_neg_y,
                cond_mask,
                teacher_neg_mask,
                data_info,
                gs,
            )
        cfg_copy_mse = F.mse_loss(guided_copy, guided_teacher).item()
        print(f"Copy-teacher CFG MSE (gs={gs}):", cfg_copy_mse)


if __name__ == "__main__":
    main()
