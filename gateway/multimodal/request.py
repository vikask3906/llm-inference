from __future__ import annotations

"""Multi-modal request parsing + token-cost model.

A multi-modal request mixes text with media (images, audio). Two facts drive
routing:

  * Modality capability: only a vision-language backend can serve an image; an
    audio model is needed for audio. This is a hard filter (the modality
    analogue of LoRA-adapter routing).
  * Cost asymmetry: an image is not one token. Vision encoders expand an image
    into many tokens -- we estimate that with OpenAI's documented tiling
    formula (base + 170 per 512px tile after a fit-to-2048-then-768 rescale),
    so the prefill estimate reflects the real load an image imposes.

Each media item also hashes to a stable content id, so a backend that has
already encoded an image (its vision KV is warm) can be preferred on reuse.
"""

import dataclasses
import math

from ..hashing import stable_seed
from .config import MultiModalConfig


def image_tokens(width: int, height: int, detail: str, cfg: MultiModalConfig) -> int:
    """OpenAI-style image token cost.

    `low` detail is a flat base cost; `high` detail rescales to fit 2048x2048
    then shortest-side 768, tiles into 512px squares, and charges per tile.
    """
    base = cfg.image_base_tokens
    if detail == "low":
        return base
    w, h = float(width), float(height)
    if max(w, h) > 2048:
        s = 2048.0 / max(w, h)
        w, h = w * s, h * s
    if min(w, h) > 768:
        s = 768.0 / min(w, h)
        w, h = w * s, h * s
    tiles = math.ceil(w / cfg.image_tile_px) * math.ceil(h / cfg.image_tile_px)
    return base + cfg.image_tile_tokens * tiles


@dataclasses.dataclass
class MediaItem:
    modality: str                 # "image" | "audio"
    ref: str                      # url / data-uri (hashed for affinity)
    width: int | None = None
    height: int | None = None
    detail: str = "high"
    seconds: float | None = None  # audio duration

    @property
    def content_id(self) -> int:
        return stable_seed(self.ref)

    def tokens(self, cfg: MultiModalConfig) -> int:
        if self.modality == "image":
            return image_tokens(self.width or cfg.default_image_width,
                                self.height or cfg.default_image_height,
                                self.detail, cfg)
        if self.modality == "audio":
            secs = self.seconds or 0.0
            return int(secs * cfg.audio_tokens_per_second)
        return 0


@dataclasses.dataclass
class ModalRequest:
    text: str
    media: list[MediaItem]

    @property
    def required_modalities(self) -> set[str]:
        mods = {"text"}
        mods.update(m.modality for m in self.media)
        return mods

    def media_ids(self) -> list[int]:
        return [m.content_id for m in self.media]

    def prompt_tokens(self, cfg: MultiModalConfig) -> int:
        text_tokens = len(self.text) // max(1, cfg.chars_per_token)
        return text_tokens + sum(m.tokens(cfg) for m in self.media)


def _parse_image_part(part: dict) -> MediaItem | None:
    img = part.get("image_url")
    url = img.get("url") if isinstance(img, dict) else img
    if not isinstance(url, str) or not url:
        return None
    detail = img.get("detail", "high") if isinstance(img, dict) else "high"
    w = img.get("width") if isinstance(img, dict) else None
    h = img.get("height") if isinstance(img, dict) else None
    return MediaItem(modality="image", ref=url, width=w, height=h, detail=detail)


def _parse_audio_part(part: dict) -> MediaItem | None:
    aud = part.get("audio_url") or part.get("input_audio")
    ref = aud.get("url") if isinstance(aud, dict) else aud
    if not isinstance(ref, str) or not ref:
        return None
    secs = aud.get("seconds") if isinstance(aud, dict) else None
    return MediaItem(modality="audio", ref=ref, seconds=secs)


def parse_multimodal_request(body: dict,
                             cfg: MultiModalConfig | None = None) -> ModalRequest | None:
    """Extract a ModalRequest from OpenAI-style messages, or None if the request
    carries no media (so the caller falls back to text routing)."""
    field = (cfg or MultiModalConfig()).request_field_messages
    messages = body.get(field)
    if not isinstance(messages, list):
        return None
    texts: list[str] = []
    media: list[MediaItem] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif ptype == "image_url":
                item = _parse_image_part(part)
                if item:
                    media.append(item)
            elif ptype in ("audio_url", "input_audio"):
                item = _parse_audio_part(part)
                if item:
                    media.append(item)
    if not media:
        return None
    return ModalRequest(text="\n".join(texts), media=media)
