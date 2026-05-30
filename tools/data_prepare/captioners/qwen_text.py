from __future__ import annotations

import logging
from typing import Optional

import torch

from .base import CaptionResult, normalize_caption_text, parse_caption_json


LOGGER = logging.getLogger(__name__)


class QwenTextCaptioner:
    """Text-only recaptioner backed by Qwen3.6-35B-A3B.

    The model rewrites an existing OpenVid caption into short/medium/long
    variants. It intentionally does not decode video frames.
    """

    source_name = "qwen3.6-35b-a3b_text"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.6-35B-A3B",
        *,
        device: str = "cuda:0",
        dtype: str = "bf16",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16 if dtype == "fp16" else torch.float32
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.trust_remote_code = bool(trust_remote_code)
        self.processor = None
        self.tokenizer = None
        self.model = None
        self._backend: Optional[str] = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            LOGGER.info("Loading Qwen recaption model with causal LM backend: %s", self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=self.trust_remote_code,
            )
            self.model.eval()
            self._backend = "causal_lm"
            return
        except Exception as exc:
            LOGGER.warning("Causal LM backend failed, falling back to image-text-to-text: %s", exc)

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            LOGGER.info("Loading Qwen recaption model with image-text-to-text backend: %s", self.model_id)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=self.trust_remote_code,
            )
            self.model.eval()
            self._backend = "image_text_to_text"
            return
        except Exception as exc:
            raise RuntimeError(f"Unable to load Qwen model with causal LM or image-text backend: {exc}") from exc

    def caption_from_text(self, caption: str) -> CaptionResult:
        self._load()
        prompt = self._build_prompt(caption)
        raw = self._generate(prompt)
        try:
            return parse_caption_json(raw, fallback_caption=caption)
        except Exception:
            LOGGER.warning("Failed to parse Qwen caption JSON. Raw response prefix=%r", raw[:1000])
            raise

    def _build_prompt(self, caption: str) -> str:
        caption = normalize_caption_text(caption)
        return f"""You are rewriting a text-to-video training caption.

Original caption:
{caption}

Return valid JSON only with exactly these keys:
{{
  "caption_short": "...",
  "caption_medium": "...",
  "caption_long": "..."
}}

Rules:
- caption_short: under 10 words, concise visual phrase.
- caption_medium: one natural sentence under 25 words.
- caption_long: one detailed sentence under 60 words, describing subject, action, scene, motion, and visual style.
- Preserve the original meaning. Do not add objects, people, actions, text, or locations not implied by the original caption.
- Do not mention that this is a video.
- Do not include explanations, markdown, or extra keys."""

    def _generate(self, prompt: str) -> str:
        do_sample = self.temperature > 0

        if self._backend == "image_text_to_text":
            assert self.processor is not None
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You produce strict JSON for dataset caption rewriting.\n\n" + prompt,
                        }
                    ],
                }
            ]
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )
            except TypeError:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            inputs = self._move_inputs_to_model_device(inputs)
            input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs.update({"temperature": self.temperature, "top_p": self.top_p})
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_ids = output_ids[:, input_len:] if input_len else output_ids
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        assert self.tokenizer is not None
        messages = [
            {"role": "system", "content": "You produce strict JSON for dataset caption rewriting."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = self._move_inputs_to_model_device(inputs)
        input_len = int(inputs["input_ids"].shape[-1])
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs.update({"temperature": self.temperature, "top_p": self.top_p})
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = output_ids[:, input_len:]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _move_inputs_to_model_device(self, inputs):
        device = next(self.model.parameters()).device
        moved = {}
        for key, value in inputs.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        return moved
