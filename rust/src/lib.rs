//! Prefix-aware LLM inference gateway — data-plane core (Rust).
//!
//! A faithful port of the Python MVP hot path: chained block hashing, a
//! path-compressed prefix radix tree with longest-prefix match + LRU eviction,
//! and the est-TTFT router with a load-spread tiebreak. std-only so it builds
//! without external crates; the Tokio/hyper streaming proxy is layered on next.

pub mod config;
pub mod hashing;
pub mod load;
pub mod radix_tree;
pub mod router;
