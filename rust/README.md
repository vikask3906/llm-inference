# gateway-core (Rust)

Rust port of the Python gateway's **data-plane hot path** — the part being
rewritten for predictable sub-millisecond routing overhead and high throughput.
Mirrors the Python modules 1:1 so behaviour is identical
(`gateway/hashing.py` → `src/hashing.rs`, etc.); see
[../docs/IMPLEMENTATION.md](../docs/IMPLEMENTATION.md).

## Status

Ported & tested (std-only, **20 tests passing**):

- `hashing.rs` — chained FNV block hashing + tenant `stable_seed`
- `radix_tree.rs` — path-compressed prefix tree: longest-contiguous match,
  edge-splitting insert, per-backend LRU eviction, membership eviction
- `router.rs` — est-TTFT routing + load-spread tiebreak (round_robin /
  consistent_hash / prefix_tree)
- `load.rs`, `config.rs` — real-time load signals + tunables

**Next:** the async HTTP layer (Tokio + hyper/axum) — OpenAI-compatible SSE
streaming proxy + control-plane scrape loop — then a profiled latency/throughput
comparison against the Python baseline (the "measured optimization" story).

## Build & test

```bash
cd rust
cargo test     # 20 tests
cargo check    # type-check only (no linker required)
```
