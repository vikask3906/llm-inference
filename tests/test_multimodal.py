"""Unit tests for multi-modal routing (gateway/multimodal).

Properties: parse OpenAI-style media messages, cost images with OpenAI's
documented tiling formula, hard-filter by modality capability, and prefer a
backend whose media-encoder KV is already warm.
"""

from gateway.load_tracker import LoadTracker
from gateway.multimodal import (
    MediaAffinityIndex,
    MediaItem,
    ModalRequest,
    MultiModalConfig,
    choose_multimodal_backend,
    image_tokens,
    parse_capabilities,
    parse_multimodal_request,
)

CFG = MultiModalConfig()


# --- image token cost (against OpenAI's documented values) ---

def test_image_tokens_known_values():
    assert image_tokens(1024, 1024, "high", CFG) == 765
    assert image_tokens(512, 512, "high", CFG) == 255
    assert image_tokens(2048, 4096, "high", CFG) == 1105
    assert image_tokens(1024, 1024, "low", CFG) == 85


# --- request parsing ---

def test_parse_multimodal_valid():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "http://img/a.png", "detail": "high"}},
    ]}]}
    req = parse_multimodal_request(body, CFG)
    assert req is not None
    assert req.text == "what is this?"
    assert len(req.media) == 1
    assert req.media[0].modality == "image"
    assert req.required_modalities == {"text", "image"}


def test_parse_text_only_returns_none():
    assert parse_multimodal_request(
        {"messages": [{"role": "user", "content": "just text"}]}, CFG) is None
    assert parse_multimodal_request({"messages": []}, CFG) is None
    assert parse_multimodal_request({"prompt": "x"}, CFG) is None


def test_parse_audio():
    body = {"messages": [{"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "http://a/clip.wav", "seconds": 12}},
    ]}]}
    req = parse_multimodal_request(body, CFG)
    assert req.required_modalities == {"text", "audio"}
    assert req.media[0].seconds == 12


def test_prompt_tokens_includes_image_expansion():
    req = ModalRequest(text="a" * 40, media=[MediaItem("image", "u", 1024, 1024)])
    # 40 chars / 4 = 10 text tokens + 765 image tokens
    assert req.prompt_tokens(CFG) == 10 + 765


# --- capability parsing + filtering ---

def test_parse_capabilities():
    caps = parse_capabilities("b0:text,image;b1:text")
    assert caps == {"b0": {"text", "image"}, "b1": {"text"}}


def _image_req(url="http://img/x.png"):
    return ModalRequest(text="hi", media=[MediaItem("image", url, 1024, 1024)])


def test_image_routes_only_to_vision_backend():
    caps = {"b0": {"text"}, "b1": {"text", "image"}}
    res = choose_multimodal_backend(req=_image_req(), backends=["b0", "b1"],
                                    capabilities=caps, index=MediaAffinityIndex(),
                                    load=LoadTracker(), cfg=CFG)
    assert res.backend_id == "b1"


def test_no_capable_backend_returns_none():
    caps = {"b0": {"text"}, "b1": {"text"}}
    res = choose_multimodal_backend(req=_image_req(), backends=["b0", "b1"],
                                    capabilities=caps, index=MediaAffinityIndex(),
                                    load=LoadTracker(), cfg=CFG)
    assert res is None


def test_no_backends_returns_none():
    res = choose_multimodal_backend(req=_image_req(), backends=[],
                                    capabilities={}, index=MediaAffinityIndex(),
                                    load=LoadTracker(), cfg=CFG)
    assert res is None


# --- media affinity ---

def test_prefers_backend_with_warm_media():
    caps = {"b0": {"text", "image"}, "b1": {"text", "image"}}
    idx = MediaAffinityIndex()
    req = _image_req()
    idx.record("b0", req.media_ids())          # b0 already encoded this image
    res = choose_multimodal_backend(req=req, backends=["b0", "b1"], capabilities=caps,
                                    index=idx, load=LoadTracker(), cfg=CFG)
    assert res.backend_id == "b0"
    assert res.media_overlap == 1
    assert res.total_media == 1


def test_saturated_capable_backend_excluded():
    caps = {"b0": {"text", "image"}, "b1": {"text", "image"}}
    idx = MediaAffinityIndex()
    req = _image_req()
    idx.record("b0", req.media_ids())          # b0 warm...
    load = LoadTracker()
    load.inflight["b0"] = CFG.max_inflight + 1  # ...but saturated
    res = choose_multimodal_backend(req=req, backends=["b0", "b1"], capabilities=caps,
                                    index=idx, load=load, cfg=CFG)
    assert res.backend_id == "b1"
