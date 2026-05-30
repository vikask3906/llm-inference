from __future__ import annotations

import os
from dataclasses import dataclass

"""Config for multi-modal capability/affinity routing (standalone)."""


@dataclass
class MultiModalConfig:
    enabled: bool = False

    # "b0:text,image;b1:text,image,audio;b2:text" -- modalities each backend serves.
    capabilities: str = ""
    request_field_messages: str = "messages"

    # --- Token cost model ---
    chars_per_token: int = 4
    # Assumed pixels when a request doesn't carry image dimensions.
    default_image_width: int = 1024
    default_image_height: int = 1024
    # OpenAI-style image token formula: base + per_tile * (#512px tiles).
    image_base_tokens: int = 85
    image_tile_tokens: int = 170
    image_tile_px: int = 512
    audio_tokens_per_second: int = 50      # rough cost for audio media

    # --- Routing cost model (mirrors the text router) ---
    prefill_ms_per_token: float = 0.05
    service_ms_per_request: float = 200.0
    max_inflight: int = 64

    # --- Media affinity cache (per-backend LRU set of media ids) ---
    cache_capacity_media: int = 1024

    @classmethod
    def from_env(cls) -> "MultiModalConfig":
        cfg = cls()
        for f in cls.__dataclass_fields__:
            env = os.environ.get(f"GW_MULTIMODAL_{f.upper()}")
            if env is None:
                continue
            cur = getattr(cfg, f)
            if isinstance(cur, bool):
                setattr(cfg, f, env.lower() in ("1", "true", "yes"))
            elif isinstance(cur, int):
                setattr(cfg, f, int(env))
            elif isinstance(cur, float):
                setattr(cfg, f, float(env))
            else:
                setattr(cfg, f, env)
        return cfg
