#!/usr/bin/env python
"""Worked example of multi-modal capability + affinity routing.

Shows three things with no GPU:

  1. image token cost via OpenAI's tiling formula (an image is many tokens),
  2. the hard modality filter -- an image request can only land on a backend
     that loads the vision modality, and
  3. media affinity -- once a backend has encoded an image, a reuse of that
     image routes back to it (warm vision-encoder KV) instead of paying the
     full re-encode elsewhere.

Usage:
    python scripts/multimodal_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gateway.load_tracker import LoadTracker
from gateway.multimodal import (
    MediaAffinityIndex,
    MediaItem,
    ModalRequest,
    MultiModalConfig,
    choose_multimodal_backend,
    image_tokens,
    parse_capabilities,
)


def main() -> None:
    cfg = MultiModalConfig()
    caps = parse_capabilities("b0:text;b1:text,image;b2:text,image")
    print("fleet capabilities:", {k: sorted(v) for k, v in caps.items()}, "\n")

    print("== 1. image token cost (OpenAI tiling) ==")
    for (w, h, detail) in [(512, 512, "high"), (1024, 1024, "high"),
                           (2048, 4096, "high"), (1024, 1024, "low")]:
        print(f"  {w}x{h:<5} {detail:>4} -> {image_tokens(w, h, detail, cfg):>5} tokens")

    req = ModalRequest(text="describe this scene",
                       media=[MediaItem("image", "http://img/scene.png", 1024, 1024)])
    idx = MediaAffinityIndex()
    load = LoadTracker()

    print("\n== 2. capability filter (image request) ==")
    res = choose_multimodal_backend(req=req, backends=["b0", "b1", "b2"],
                                    capabilities=caps, index=idx, load=load, cfg=cfg)
    print(f"  text-only b0 is excluded; chose {res.backend_id} "
          f"({res.est_tokens} prompt tokens, {res.est_ms:.1f}ms)")

    print("\n== 3. media affinity on reuse ==")
    idx.record(res.backend_id, req.media_ids())     # that backend encoded the image
    res2 = choose_multimodal_backend(req=req, backends=["b0", "b1", "b2"],
                                     capabilities=caps, index=idx, load=load, cfg=cfg)
    print(f"  same image again -> {res2.backend_id} "
          f"(media_overlap {res2.media_overlap}/{res2.total_media}, "
          f"est {res2.est_ms:.1f}ms vs cold {res.est_ms:.1f}ms)")


if __name__ == "__main__":
    main()
