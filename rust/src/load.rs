//! Real-time, gateway-owned load signals (mirrors Python `LoadTracker`).

use std::collections::HashMap;

#[derive(Default)]
pub struct LoadTracker {
    inflight: HashMap<String, u32>,
    inflight_tokens: HashMap<String, u64>,
    kv_usage: HashMap<String, f64>,
}

impl LoadTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inflight(&self, backend: &str) -> u32 {
        *self.inflight.get(backend).unwrap_or(&0)
    }

    pub fn kv_usage(&self, backend: &str) -> f64 {
        *self.kv_usage.get(backend).unwrap_or(&0.0)
    }

    pub fn set_inflight(&mut self, backend: &str, n: u32) {
        self.inflight.insert(backend.to_string(), n);
    }

    pub fn on_dispatch(&mut self, backend: &str, tokens: u64) {
        *self.inflight.entry(backend.to_string()).or_insert(0) += 1;
        *self.inflight_tokens.entry(backend.to_string()).or_insert(0) += tokens;
    }

    pub fn on_complete(&mut self, backend: &str, tokens: u64) {
        let c = self.inflight.entry(backend.to_string()).or_insert(0);
        *c = c.saturating_sub(1);
        let t = self.inflight_tokens.entry(backend.to_string()).or_insert(0);
        *t = t.saturating_sub(tokens);
    }

    pub fn update_scraped(&mut self, backend: &str, kv: f64, alpha: f64) {
        let prev = self.kv_usage(backend);
        self.kv_usage.insert(backend.to_string(), alpha * kv + (1.0 - alpha) * prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_and_complete() {
        let mut lt = LoadTracker::new();
        lt.on_dispatch("b0", 100);
        lt.on_dispatch("b0", 50);
        assert_eq!(lt.inflight("b0"), 2);
        lt.on_complete("b0", 100);
        assert_eq!(lt.inflight("b0"), 1);
    }

    #[test]
    fn complete_never_negative() {
        let mut lt = LoadTracker::new();
        lt.on_complete("b0", 100);
        assert_eq!(lt.inflight("b0"), 0);
    }

    #[test]
    fn ewma() {
        let mut lt = LoadTracker::new();
        lt.update_scraped("b0", 1.0, 0.5);
        assert_eq!(lt.kv_usage("b0"), 0.5);
        lt.update_scraped("b0", 1.0, 0.5);
        assert_eq!(lt.kv_usage("b0"), 0.75);
    }
}
