from __future__ import annotations

from .base import CaptionResult, normalize_caption_text


class MockTextCaptioner:
    """Deterministic captioner for CPU-only pipeline tests."""

    source_name = "mock_text"

    def caption_from_text(self, caption: str) -> CaptionResult:
        base = normalize_caption_text(caption) or "an unspecified scene"
        words = base.split()
        short = " ".join(words[: min(8, len(words))])
        medium = f"A clear scene showing {base}."
        long = (
            f"A detailed training caption describing {base}, with the main subject, "
            "action, surrounding scene, and visible motion kept consistent."
        )
        return CaptionResult(
            caption_short=short,
            caption_medium=normalize_caption_text(medium),
            caption_long=normalize_caption_text(long),
            raw_response='{"caption_short": "...", "caption_medium": "...", "caption_long": "..."}',
        )
