//! Routing-relevant configuration (mirrors the Python `Config` defaults).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    RoundRobin,
    ConsistentHash,
    PrefixTree,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub block_chars: usize,
    pub block_tokens: usize,
    pub hash_cutoff_blocks: usize,
    pub backend_cache_blocks: usize,
    pub prefill_ms_per_token: f64,
    pub service_ms_per_request: f64,
    pub kv_pressure_cutoff: f64,
    pub max_inflight: u32,
    pub hysteresis_ms: f64,
    pub strategy: Strategy,

    // --- fault tolerance ---
    pub max_retries: u32,
    pub circuit_fail_threshold: u32,
    pub circuit_cooldown_s: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            block_chars: 64,
            block_tokens: 16,
            hash_cutoff_blocks: 512,
            backend_cache_blocks: 2000,
            prefill_ms_per_token: 0.05,
            service_ms_per_request: 200.0,
            kv_pressure_cutoff: 0.90,
            max_inflight: 64,
            hysteresis_ms: 5.0,
            strategy: Strategy::PrefixTree,
            max_retries: 2,
            circuit_fail_threshold: 3,
            circuit_cooldown_s: 5.0,
        }
    }
}
