//! Routing strategies (mirrors Python `gateway/router.py`).
//!
//! prefix_tree minimizes estimated TTFT, then spreads within `hysteresis_ms` of
//! the best by least-load + round-robin so a tiny shared prefix can't pin all
//! traffic to one node while a large real affinity still isolates one backend.

use crate::config::{Config, Strategy};
use crate::hashing::block_hashes;
use crate::load::LoadTracker;
use crate::radix_tree::RadixTree;

#[derive(Debug, Clone)]
pub struct RouteResult {
    pub backend_id: String,
    pub hashes: Vec<u64>,
    pub tokens: usize,
    pub match_blocks: usize,
}

pub struct Router {
    pub cfg: Config,
    rr: usize,
}

impl Router {
    pub fn new(cfg: Config) -> Self {
        Router { cfg, rr: 0 }
    }

    pub fn choose(
        &mut self,
        prompt: &str,
        backends: &[String],
        tree: &RadixTree,
        load: &LoadTracker,
        strategy: Strategy,
        seed: u64,
    ) -> RouteResult {
        let hashes = block_hashes(prompt, self.cfg.block_chars, self.cfg.hash_cutoff_blocks, seed);
        let tokens = hashes.len() * self.cfg.block_tokens;

        match strategy {
            Strategy::RoundRobin => {
                let b = backends[self.rr % backends.len()].clone();
                self.rr += 1;
                RouteResult { backend_id: b, hashes, tokens, match_blocks: 0 }
            }
            Strategy::ConsistentHash => {
                let key = hashes.first().copied().unwrap_or(0);
                let idx = (key % backends.len() as u64) as usize;
                RouteResult { backend_id: backends[idx].clone(), hashes, tokens, match_blocks: 0 }
            }
            Strategy::PrefixTree => {
                let matches = tree.match_prefix(&hashes);
                let mut scored: Vec<(f64, String, usize)> = Vec::new();
                let mut saturated_fallback: Option<String> = None;

                for b in backends {
                    if load.kv_usage(b) > self.cfg.kv_pressure_cutoff
                        || load.inflight(b) > self.cfg.max_inflight
                    {
                        match &saturated_fallback {
                            None => saturated_fallback = Some(b.clone()),
                            Some(sb) => {
                                if load.inflight(b) < load.inflight(sb) {
                                    saturated_fallback = Some(b.clone());
                                }
                            }
                        }
                        continue;
                    }
                    let mb = *matches.get(b).unwrap_or(&0);
                    scored.push((self.est_ttft(tokens, mb, b, load), b.clone(), mb));
                }

                if scored.is_empty() {
                    let b = saturated_fallback.unwrap_or_else(|| backends[0].clone());
                    let mb = *matches.get(&b).unwrap_or(&0);
                    return RouteResult { backend_id: b, hashes, tokens, match_blocks: mb };
                }

                let min_ttft = scored.iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
                let good: Vec<(String, usize)> = scored
                    .iter()
                    .filter(|s| s.0 <= min_ttft + self.cfg.hysteresis_ms)
                    .map(|s| (s.1.clone(), s.2))
                    .collect();
                let min_load = good.iter().map(|(b, _)| load.inflight(b)).min().unwrap();
                let tied: Vec<(String, usize)> =
                    good.into_iter().filter(|(b, _)| load.inflight(b) == min_load).collect();
                let (b, mb) = tied[self.rr % tied.len()].clone();
                self.rr += 1;
                RouteResult { backend_id: b, hashes, tokens, match_blocks: mb }
            }
        }
    }

    fn est_ttft(&self, tokens: usize, match_blocks: usize, backend: &str, load: &LoadTracker) -> f64 {
        let cached = match_blocks * self.cfg.block_tokens;
        let uncached = tokens.saturating_sub(cached);
        let prefill = self.cfg.prefill_ms_per_token * uncached as f64;
        let queue = load.inflight(backend) as f64 * self.cfg.service_ms_per_request;
        prefill + queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashing::block_hashes;

    fn backends() -> Vec<String> {
        vec!["b0".into(), "b1".into(), "b2".into()]
    }

    #[test]
    fn round_robin_cycles() {
        let mut r = Router::new(Config::default());
        let tree = RadixTree::new(2000);
        let load = LoadTracker::new();
        let bs = backends();
        let seq: Vec<String> = (0..6)
            .map(|_| r.choose("p", &bs, &tree, &load, Strategy::RoundRobin, 0).backend_id)
            .collect();
        assert_eq!(seq, vec!["b0", "b1", "b2", "b0", "b1", "b2"]);
    }

    #[test]
    fn prefix_tree_prefers_cached_backend() {
        let cfg = Config::default();
        let mut tree = RadixTree::new(cfg.backend_cache_blocks);
        let load = LoadTracker::new();
        let prompt = "X".repeat(1300); // ~20 blocks -> affinity > hysteresis
        let h = block_hashes(&prompt, cfg.block_chars, cfg.hash_cutoff_blocks, 0);
        tree.insert(&h, "b1");
        let mut r = Router::new(cfg);
        let res = r.choose(&prompt, &backends(), &tree, &load, Strategy::PrefixTree, 0);
        assert_eq!(res.backend_id, "b1");
        assert_eq!(res.match_blocks, h.len());
    }

    #[test]
    fn prefix_tree_saturation_cutoff_excludes_overloaded() {
        let cfg = Config::default();
        let mut tree = RadixTree::new(cfg.backend_cache_blocks);
        let mut load = LoadTracker::new();
        let prompt = "X".repeat(1300);
        let h = block_hashes(&prompt, cfg.block_chars, cfg.hash_cutoff_blocks, 0);
        tree.insert(&h, "b1");
        load.set_inflight("b1", cfg.max_inflight + 1); // b1 has the cache but is saturated
        let mut r = Router::new(cfg);
        let res = r.choose(&prompt, &backends(), &tree, &load, Strategy::PrefixTree, 0);
        assert_ne!(res.backend_id, "b1");
    }
}
