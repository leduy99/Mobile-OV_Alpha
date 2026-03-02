#!/usr/bin/env python3
"""
Simple SmolVLM2 VQA/chat runner.

Supports text-only chat and optional image/video inputs via HuggingFace AutoProcessor.
"""

import argparse
import logging
import os
from typing import List, Optional

import torch

from nets.smolvlm2 import load_smolvlm2_from_ckpt, SmolVLMForConditionalGeneration


logger = logging.getLogger(__name__)


def _load_images(image_path: Optional[str], video_path: Optional[str], num_frames: int) -> Optional[List["PIL.Image.Image"]]:
    if image_path is None and video_path is None:
        return None

    images = []
    if image_path is not None:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        from PIL import Image
        images.append(Image.open(image_path).convert("RGB"))

    if video_path is not None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        from nets.third_party.llava.mm_utils import opencv_extract_frames
        frames, _ = opencv_extract_frames(video_path, frames=num_frames)
        images.extend(frames)

    return images if images else None


def _build_inputs(
    prompt: str,
    images: Optional[List["PIL.Image.Image"]],
    device: torch.device,
    model_id: str,
    prefer_processor: bool,
):
    tokenizer = None
    processor = None

    if prefer_processor and images is not None:
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            logger.info("Loaded AutoProcessor from %s", model_id)
        except Exception as exc:
            logger.warning("Failed to load AutoProcessor (%s). Falling back to text-only. Error: %s", model_id, exc)
            processor = None

    if processor is not None:
        inputs = processor(text=prompt, images=images, return_tensors="pt")
        input_ids = inputs.get("input_ids", None)
        attention_mask = inputs.get("attention_mask", None)
        pixel_values = inputs.get("pixel_values", None)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.get("input_ids", None)
        attention_mask = inputs.get("attention_mask", None)
        pixel_values = None

    if input_ids is None:
        raise RuntimeError("Failed to build input_ids for generation.")

    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

    return input_ids, attention_mask, pixel_values, tokenizer, processor


def generate_once(
    model,
    prompt: str,
    images: Optional[List["PIL.Image.Image"]],
    device: torch.device,
    model_id: str,
    prefer_processor: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: Optional[float],
):
    input_ids, attention_mask, pixel_values, tokenizer, processor = _build_inputs(
        prompt, images, device, model_id, prefer_processor
    )

    pad_token_id = None
    eos_token_id = None
    if processor is not None and hasattr(processor, "tokenizer"):
        pad_token_id = processor.tokenizer.pad_token_id
        eos_token_id = processor.tokenizer.eos_token_id
    elif tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
    }
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **gen_kwargs,
        )

    generated_ids = output_ids[0][input_ids.shape[1]:]
    if processor is not None and hasattr(processor, "tokenizer"):
        text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    elif tokenizer is not None:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        text = "<no tokenizer available>"
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="SmolVLM2 VQA/chat runner")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to converted SmolVLM2 checkpoint (.pt)")
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="HF model id for tokenizer/processor",
    )
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="User prompt")
    parser.add_argument("--image", type=str, default=None, help="Path to image (optional)")
    parser.add_argument("--video", type=str, default=None, help="Path to video (optional)")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames to sample from video")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device (e.g., cuda:0, cpu)")
    parser.add_argument("--no-processor", action="store_true", help="Disable AutoProcessor (text-only)")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat loop (text-only recommended)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device(args.device)
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    model = load_smolvlm2_from_ckpt(
        args.ckpt_path,
        device=device,
        model_class=SmolVLMForConditionalGeneration,
    )
    model.eval()

    if not hasattr(model, "generate"):
        raise RuntimeError("Loaded SmolVLM2 model does not support generate(). Use a checkpoint with lm_head.")

    images = _load_images(args.image, args.video, args.num_frames)
    prefer_processor = not args.no_processor

    if args.interactive:
        logger.info("Entering interactive mode. Type 'exit' to quit.")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            response = generate_once(
                model=model,
                prompt=user_input,
                images=None,
                device=device,
                model_id=args.model_id,
                prefer_processor=False,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"Assistant: {response}")
    else:
        response = generate_once(
            model=model,
            prompt=args.prompt,
            images=images,
            device=device,
            model_id=args.model_id,
            prefer_processor=prefer_processor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(response)


if __name__ == "__main__":
    main()
