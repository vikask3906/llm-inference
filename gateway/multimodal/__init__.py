"""Multi-modal capability + affinity routing (standalone, not on the hot path).

Image and audio requests need backends that actually load that modality, and an
image costs many vision tokens, not one. This package:

  1. request -- parse OpenAI-style multi-modal messages into text + media, and
     estimate token cost (images via OpenAI's tiling formula).
  2. index   -- track which media each backend has already encoded (LRU set).
  3. router  -- hard-filter by modality capability, then pick the cheapest
     capable backend, crediting media whose encoder KV is already warm.

Isolated from server.py like gateway/disagg, gateway/rag, gateway/dag,
gateway/autoscale, and gateway/admission: the benchmarked text hot path is
untouched.
"""

from .config import MultiModalConfig
from .index import MediaAffinityIndex
from .request import (
    MediaItem,
    ModalRequest,
    image_tokens,
    parse_multimodal_request,
)
from .router import (
    MultiModalRouteResult,
    choose_multimodal_backend,
    parse_capabilities,
)

__all__ = [
    "MultiModalConfig",
    "MediaAffinityIndex",
    "MediaItem",
    "ModalRequest",
    "image_tokens",
    "parse_multimodal_request",
    "MultiModalRouteResult",
    "choose_multimodal_backend",
    "parse_capabilities",
]
