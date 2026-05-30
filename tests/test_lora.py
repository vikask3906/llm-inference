from __future__ import annotations

from gateway.backends import BackendRegistry
from gateway.config import Config
from gateway.extensions.lora import (
    apply_adapter_config,
    lora_filter,
    parse_adapter_config,
    parse_model_spec,
)


def test_parse_model_spec_base_only():
    assert parse_model_spec("mock-model") == ("mock-model", None)


def test_parse_model_spec_with_adapter():
    assert parse_model_spec("mock-model:summary-v2") == ("mock-model", "summary-v2")


def test_parse_model_spec_handles_whitespace():
    assert parse_model_spec("  m  :  a  ") == ("m", "a")


def test_parse_model_spec_empty():
    assert parse_model_spec("") == (None, None)
    assert parse_model_spec(None) == (None, None)


def test_parse_adapter_config_empty():
    assert parse_adapter_config("") == {}


def test_parse_adapter_config_single():
    assert parse_adapter_config("b0:summary,sql") == {"b0": {"summary", "sql"}}


def test_parse_adapter_config_multiple_backends():
    spec = "b0:summary,sql; b1:chat-ru;b2:legal"
    assert parse_adapter_config(spec) == {
        "b0": {"summary", "sql"},
        "b1": {"chat-ru"},
        "b2": {"legal"},
    }


def test_parse_adapter_config_drops_malformed():
    # missing colon, missing backend id, empty adapter list all dropped
    spec = "no-colon; :no-id; b0:; b1:ok"
    assert parse_adapter_config(spec) == {"b1": {"ok"}}


def _registry_with(spec: str) -> BackendRegistry:
    cfg = Config(backends="b0=http://h:9001,b1=http://h:9002,b2=http://h:9003")
    r = BackendRegistry(cfg)
    apply_adapter_config(r, spec)
    return r


def test_apply_adapter_config_sets_adapters():
    r = _registry_with("b0:sql; b1:summary")
    assert r._by_id["b0"].adapters == {"sql"}
    assert r._by_id["b1"].adapters == {"summary"}
    assert r._by_id["b2"].adapters == set()


def test_apply_adapter_config_ignores_unknown_backend():
    r = _registry_with("bX:not-a-real-backend")
    assert all(b.adapters == set() for b in r.all())


def test_lora_filter_no_adapter_returns_base_pool():
    r = _registry_with("b0:sql")
    assert sorted(lora_filter(r, "mock-model", None, fallback_to_base=True)) == \
           ["b0", "b1", "b2"]


def test_lora_filter_returns_only_capable_backend():
    r = _registry_with("b0:sql; b2:sql")
    assert sorted(lora_filter(r, "mock-model", "sql", fallback_to_base=True)) == \
           ["b0", "b2"]


def test_lora_filter_fallback_when_no_capable():
    r = _registry_with("b0:other")
    # nobody has "sql"; with fallback, return full base pool
    assert sorted(lora_filter(r, "mock-model", "sql", fallback_to_base=True)) == \
           ["b0", "b1", "b2"]


def test_lora_filter_strict_when_no_capable():
    r = _registry_with("b0:other")
    # nobody has "sql"; without fallback, return empty
    assert lora_filter(r, "mock-model", "sql", fallback_to_base=False) == []


def test_lora_filter_unhealthy_excluded():
    r = _registry_with("b0:sql; b1:sql")
    r.set_health("b0", False)
    assert lora_filter(r, "mock-model", "sql", fallback_to_base=True) == ["b1"]
