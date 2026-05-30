//! Multi-tenant fairness (mirrors `gateway/tenancy.py`).
//!
//! Admission control sits before routing. Per-tenant token buckets on both RPS
//! and TPS (OpenAI RPM+TPM style) + per-tenant in-flight cap. TPS is resource-
//! aligned; RPS is a cheap abuse guard; a request must pass both. The router /
//! est-TTFT cost function are untouched.

use std::collections::HashMap;

use crate::hashing::stable_seed;

#[derive(Clone, Copy, Debug)]
pub struct Tier {
    pub rps: f64,
    pub tps: f64,
    pub max_inflight: u32,
}

/// Quota tiers. Production would load these from config/DB; sensible defaults here.
pub fn default_tiers() -> HashMap<&'static str, Tier> {
    let mut m = HashMap::new();
    m.insert("gold",      Tier { rps: 50.0,  tps: 100_000.0, max_inflight: 64 });
    m.insert("silver",    Tier { rps: 20.0,  tps:  40_000.0, max_inflight: 24 });
    m.insert("bronze",    Tier { rps:  5.0,  tps:  10_000.0, max_inflight:  8 });
    m.insert("anonymous", Tier { rps:  2.0,  tps:   4_000.0, max_inflight:  4 });
    m
}

#[derive(Clone, Debug)]
pub struct Tenant {
    pub id: String,
    pub rps: f64,
    pub tps: f64,
    pub max_inflight: u32,
}

impl Tenant {
    pub fn from_tier(id: &str, t: Tier) -> Self {
        Tenant { id: id.to_string(), rps: t.rps, tps: t.tps, max_inflight: t.max_inflight }
    }
}

/// Lazy-refill token bucket: O(1), no timers. Starts full (allows a burst).
pub struct TokenBucket {
    pub capacity: f64,
    pub refill: f64,
    pub tokens: f64,
    pub ts: f64,
}

impl TokenBucket {
    pub fn new(capacity: f64, refill_per_sec: f64) -> Self {
        // ts starts at 0: an idle bucket refills to full on first use, and
        // explicit `now=...` values in tests aren't seen as "in the past".
        TokenBucket { capacity, refill: refill_per_sec, tokens: capacity, ts: 0.0 }
    }

    fn replenish(&mut self, now: f64) {
        if now > self.ts {
            self.tokens = (self.tokens + (now - self.ts) * self.refill).min(self.capacity);
            self.ts = now;
        }
    }

    pub fn try_consume(&mut self, n: f64, now: f64) -> bool {
        self.replenish(now);
        if self.tokens >= n {
            self.tokens -= n;
            true
        } else {
            false
        }
    }

    pub fn deficit_seconds(&mut self, n: f64, now: f64) -> f64 {
        self.replenish(now);
        if self.tokens >= n {
            0.0
        } else if self.refill <= 0.0 {
            f64::INFINITY
        } else {
            (n - self.tokens) / self.refill
        }
    }

    pub fn adjust(&mut self, delta: f64) {
        self.tokens = (self.tokens + delta).min(self.capacity);
    }
}

#[derive(Clone, Debug)]
pub struct Admission {
    pub allowed: bool,
    pub reason: Option<&'static str>, // "rps" | "tps" | "inflight"
    pub retry_after: f64,
    pub remaining_rps: f64,
    pub remaining_tps: f64,
}

/// Resolves a request to a Tenant via the Authorization bearer key.
pub struct TenantRegistry {
    by_key: HashMap<String, Tenant>,
    anonymous: Tenant,
}

impl TenantRegistry {
    pub fn new(spec: &str) -> Self {
        let tiers = default_tiers();
        let mut by_key = HashMap::new();
        for pair in spec.split(',') {
            let p = pair.trim();
            if p.is_empty() {
                continue;
            }
            let (key, rest) = match p.split_once('=') {
                Some(x) => x,
                None => continue,
            };
            let (tid, tier_name) = match rest.split_once(':') {
                Some((t, n)) => (t, n),
                None => (rest, "bronze"),
            };
            let tier = tiers
                .get(tier_name.trim())
                .copied()
                .unwrap_or_else(|| tiers["bronze"]);
            let id = if tid.trim().is_empty() { key.trim() } else { tid.trim() };
            by_key.insert(key.trim().to_string(), Tenant::from_tier(id, tier));
        }
        let anonymous = Tenant::from_tier("anonymous", tiers["anonymous"]);
        TenantRegistry { by_key, anonymous }
    }

    pub fn resolve(&self, auth_header: Option<&str>) -> Tenant {
        if let Some(h) = auth_header {
            let bytes = h.as_bytes();
            if bytes.len() > 7 && bytes[..7].eq_ignore_ascii_case(b"Bearer ") {
                let key = h[7..].trim();
                if let Some(t) = self.by_key.get(key) {
                    return t.clone();
                }
            }
        }
        self.anonymous.clone()
    }
}

/// Per-tenant RPS + TPS token buckets and an in-flight cap.
pub struct RateLimiter {
    rps: HashMap<String, TokenBucket>,
    tps: HashMap<String, TokenBucket>,
    pub inflight: HashMap<String, u32>,
}

impl Default for RateLimiter {
    fn default() -> Self { Self::new() }
}

impl RateLimiter {
    pub fn new() -> Self {
        RateLimiter { rps: HashMap::new(), tps: HashMap::new(), inflight: HashMap::new() }
    }

    fn ensure(&mut self, t: &Tenant) {
        if !self.rps.contains_key(&t.id) {
            self.rps.insert(t.id.clone(), TokenBucket::new(t.rps.max(1.0), t.rps));
            self.tps.insert(t.id.clone(), TokenBucket::new(t.tps, t.tps));
        }
    }

    pub fn admit(&mut self, t: &Tenant, cost: f64, now: f64) -> Admission {
        self.ensure(t);
        let inflight = self.inflight.get(&t.id).copied().unwrap_or(0);
        if inflight >= t.max_inflight {
            return Admission {
                allowed: false,
                reason: Some("inflight"),
                retry_after: 1.0,
                remaining_rps: self.rps[&t.id].tokens,
                remaining_tps: self.tps[&t.id].tokens,
            };
        }
        // TPS first (resource-aligned); refund if RPS then fails.
        let tps_bucket = self.tps.get_mut(&t.id).unwrap();
        if !tps_bucket.try_consume(cost, now) {
            let retry = tps_bucket.deficit_seconds(cost, now);
            let tps_tokens = tps_bucket.tokens;
            return Admission {
                allowed: false,
                reason: Some("tps"),
                retry_after: retry,
                remaining_rps: self.rps[&t.id].tokens,
                remaining_tps: tps_tokens,
            };
        }
        let rps_bucket = self.rps.get_mut(&t.id).unwrap();
        if !rps_bucket.try_consume(1.0, now) {
            let retry = rps_bucket.deficit_seconds(1.0, now);
            let rps_tokens = rps_bucket.tokens;
            // refund TPS (give back the reservation)
            self.tps.get_mut(&t.id).unwrap().adjust(cost);
            let tps_tokens = self.tps[&t.id].tokens;
            return Admission {
                allowed: false,
                reason: Some("rps"),
                retry_after: retry,
                remaining_rps: rps_tokens,
                remaining_tps: tps_tokens,
            };
        }
        *self.inflight.entry(t.id.clone()).or_insert(0) += 1;
        Admission {
            allowed: true,
            reason: None,
            retry_after: 0.0,
            remaining_rps: self.rps[&t.id].tokens,
            remaining_tps: self.tps[&t.id].tokens,
        }
    }

    pub fn release(&mut self, tenant_id: &str, reserved_output: f64, actual_output: f64) {
        let c = self.inflight.entry(tenant_id.to_string()).or_insert(0);
        if *c > 0 { *c -= 1; }
        if let Some(tps) = self.tps.get_mut(tenant_id) {
            tps.adjust(reserved_output - actual_output);
        }
    }
}

/// Hash-chain seed that namespaces a tenant's prefix cache (isolation).
pub fn tenant_seed(tenant_id: &str) -> u64 {
    stable_seed(tenant_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_consume_then_empty() {
        let mut b = TokenBucket::new(10.0, 5.0);
        assert!(b.try_consume(10.0, 0.0));
        assert!(!b.try_consume(1.0, 0.0));
    }

    #[test]
    fn bucket_refills_over_time() {
        let mut b = TokenBucket::new(10.0, 5.0);
        b.try_consume(10.0, 0.0);
        assert!(b.try_consume(5.0, 1.0));
        assert!(!b.try_consume(1.0, 1.0));
    }

    #[test]
    fn bucket_caps_at_capacity() {
        let mut b = TokenBucket::new(10.0, 5.0);
        b.try_consume(10.0, 0.0);
        assert!(b.try_consume(10.0, 100.0));
        assert!(!b.try_consume(1.0, 100.0));
    }

    #[test]
    fn bucket_deficit_seconds() {
        let mut b = TokenBucket::new(10.0, 5.0);
        b.try_consume(10.0, 0.0);
        assert!((b.deficit_seconds(5.0, 0.0) - 1.0).abs() < 1e-9);
        assert_eq!(b.deficit_seconds(0.0, 0.0), 0.0);
    }

    #[test]
    fn resolve_bearer_and_anonymous_fallback() {
        let reg = TenantRegistry::new("sk-acme=acme:gold,sk-beta=beta:silver");
        assert_eq!(reg.resolve(Some("Bearer sk-acme")).id, "acme");
        assert_eq!(reg.resolve(Some("bearer sk-beta")).id, "beta");
        assert_eq!(reg.resolve(None).id, "anonymous");
        assert_eq!(reg.resolve(Some("Bearer nope")).id, "anonymous");
    }

    #[test]
    fn tier_quotas_applied() {
        let reg = TenantRegistry::new("sk-acme=acme:gold");
        let t = reg.resolve(Some("Bearer sk-acme"));
        let gold = default_tiers()["gold"];
        assert_eq!((t.rps, t.tps), (gold.rps, gold.tps));
    }

    #[test]
    fn rps_limit_then_throttle() {
        let mut rl = RateLimiter::new();
        let t = Tenant { id: "t".into(), rps: 2.0, tps: 1_000_000.0, max_inflight: 100 };
        assert!(rl.admit(&t, 1.0, 0.0).allowed);
        assert!(rl.admit(&t, 1.0, 0.0).allowed);
        let adm = rl.admit(&t, 1.0, 0.0);
        assert!(!adm.allowed && adm.reason == Some("rps"));
    }

    #[test]
    fn tps_limit_then_throttle() {
        let mut rl = RateLimiter::new();
        let t = Tenant { id: "t".into(), rps: 1000.0, tps: 100.0, max_inflight: 100 };
        let adm = rl.admit(&t, 1000.0, 0.0);
        assert!(!adm.allowed && adm.reason == Some("tps"));
    }

    #[test]
    fn tps_refunded_when_rps_fails() {
        let mut rl = RateLimiter::new();
        let t = Tenant { id: "t".into(), rps: 1.0, tps: 1000.0, max_inflight: 100 };
        assert!(rl.admit(&t, 100.0, 0.0).allowed); // uses 1 rps + 100 tps -> tps 900
        let adm = rl.admit(&t, 100.0, 0.0);
        assert!(!adm.allowed && adm.reason == Some("rps"));
        assert!((adm.remaining_tps - 900.0).abs() < 1e-6); // refunded (not 800)
    }

    #[test]
    fn inflight_cap() {
        let mut rl = RateLimiter::new();
        let t = Tenant { id: "t".into(), rps: 1000.0, tps: 1_000_000.0, max_inflight: 2 };
        assert!(rl.admit(&t, 1.0, 0.0).allowed);
        assert!(rl.admit(&t, 1.0, 0.0).allowed);
        let adm = rl.admit(&t, 1.0, 0.0);
        assert!(!adm.allowed && adm.reason == Some("inflight"));
        rl.release("t", 0.0, 0.0);
        assert!(rl.admit(&t, 1.0, 0.0).allowed);
    }

    #[test]
    fn release_reconciles_overestimate() {
        let mut rl = RateLimiter::new();
        let t = Tenant { id: "t".into(), rps: 1000.0, tps: 1000.0, max_inflight: 10 };
        rl.admit(&t, 300.0, 0.0); // reserve 300 -> tps 700
        rl.release("t", 250.0, 50.0); // used 200 fewer -> refund 200
        assert!((rl.tps["t"].tokens - 900.0).abs() < 1e-6);
    }

    #[test]
    fn tenant_seed_namespaces_hashes() {
        let sa = tenant_seed("acme");
        let sb = tenant_seed("beta");
        assert_eq!(sa, tenant_seed("acme"));
        assert_ne!(sa, sb);
    }
}
