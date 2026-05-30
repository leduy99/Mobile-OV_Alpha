from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict


CAPTION_KEYS = ("caption_short", "caption_medium", "caption_long")


@dataclass
class CaptionResult:
    caption_short: str
    caption_medium: str
    caption_long: str
    raw_response: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "caption_short": self.caption_short,
            "caption_medium": self.caption_medium,
            "caption_long": self.caption_long,
            "caption_raw_response": self.raw_response,
        }


def normalize_caption_text(text: object) -> str:
    text = "" if text is None else str(text)
    return " ".join(text.replace("\n", " ").replace("\r", " ").split()).strip()


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_json_object(text: str) -> str:
    text = _strip_code_fence(text)
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def parse_caption_json(raw_text: str, fallback_caption: str) -> CaptionResult:
    raw_text = normalize_caption_text(raw_text)
    fallback = normalize_caption_text(fallback_caption)
    payload = json.loads(_extract_json_object(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("Caption response is not a JSON object.")

    values = {}
    for key in CAPTION_KEYS:
        text = normalize_caption_text(payload.get(key, ""))
        values[key] = text or fallback
    return CaptionResult(raw_response=raw_text, **values)


def fallback_caption_result(caption: str, raw_response: str = "") -> CaptionResult:
    caption = normalize_caption_text(caption)
    return CaptionResult(
        caption_short=caption,
        caption_medium=caption,
        caption_long=caption,
        raw_response=normalize_caption_text(raw_response),
    )
