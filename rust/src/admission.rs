//! SLO-aware admission control + priority load shedding (mirrors
//! `gateway/admission/policy.py`).
//!
//! Autoscaling fixes capacity over minutes; admission control protects the SLO
//! in the moment. Before dispatch we decide admit / queue / reject from three
//! signals:
//!   1. TTFT budget   -- if the best achievable TTFT already blows the deadline,
//!      reject early ("fail fast") instead of burning prefill on a doomed request.
//!   2. Fleet pressure -- a [0,1] saturation signal (mean KV usage vs the fraction
//!      of saturated backends).
//!   3. Priority shedding -- above `shed_pressure` the admit floor rises with
//!      pressure, so bronze sheds before silver before gold. Shed requests queue
//!      (backpressure) when possible, else reject with a Retry-After hint.
//!
//! Pure logic, no I/O -- a clean parity port of the Python policy.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct AdmissionConfig {
    pub enabled: bool,
    pub ttft_slo_ms: f64,
    pub max_inflight_per_backend: u32,
    pub shed_pressure: f64,
    pub queue_enabled: bool,
    pub max_queue_depth: u32,
    pub retry_after_base_ms: f64,
    pub retry_after_max_ms: f64,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        AdmissionConfig {
            enabled: false,
            ttft_slo_ms: 500.0,
            max_inflight_per_backend: 64,
            shed_pressure: 0.75,
            queue_enabled: true,
            max_queue_depth: 256,
            retry_after_base_ms: 200.0,
            retry_after_max_ms: 10_000.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Admit,
    Queue,
    Reject,
}

/// tier -> priority; higher survives longer under shedding.
pub fn tier_priority(tier: &str) -> i32 {
    match tier.to_ascii_lowercase().as_str() {
        "gold" => 3,
        "silver" => 2,
        "bronze" => 1,
        _ => 0,
    }
}

#[derive(Default)]
pub struct FleetState {
    pub backend_inflight: HashMap<String, u32>,
    pub backend_kv_usage: HashMap<String, f64>,
    pub queue_depth: u32,
}

pub struct AdmissionRequest<'a> {
    pub est_ttft_ms: f64,
    pub tier: &'a str,
    pub est_output_tokens: u32,
}

#[derive(Clone, Debug)]
pub struct AdmissionDecision {
    pub action: Action,
    pub reason: String,
    pub retry_after_ms: f64,
    pub pressure: f64,
    pub admit_floor: f64,
}

/// A [0,1] saturation signal: the worse of mean KV usage and the fraction of
/// backends at/over the per-backend in-flight cap.
pub fn fleet_pressure(fleet: &FleetState, cfg: &AdmissionConfig) -> f64 {
    let mean_kv = if fleet.backend_kv_usage.is_empty() {
        0.0
    } else {
        fleet.backend_kv_usage.values().sum::<f64>() / fleet.backend_kv_usage.len() as f64
    };
    let sat_frac = if fleet.backend_inflight.is_empty() {
        0.0
    } else {
        let saturated = fleet
            .backend_inflight
            .values()
            .filter(|&&n| n >= cfg.max_inflight_per_backend)
            .count();
        saturated as f64 / fleet.backend_inflight.len() as f64
    };
    mean_kv.max(sat_frac).clamp(0.0, 1.0)
}

/// Priority floor as pressure climbs from `shed_pressure` (floor 1: shed only
/// anonymous) to 1.0 (floor 3: gold only).
fn admit_floor(pressure: f64, cfg: &AdmissionConfig) -> f64 {
    let span = (1.0 - cfg.shed_pressure).max(1e-9);
    let t = ((pressure - cfg.shed_pressure) / span).clamp(0.0, 1.0);
    1.0 + 2.0 * t
}

fn retry_after(pressure: f64, cfg: &AdmissionConfig, overshoot_ms: f64) -> f64 {
    let ms = (cfg.retry_after_base_ms * (1.0 + pressure)).max(overshoot_ms);
    ms.min(cfg.retry_after_max_ms)
}

pub fn decide(
    req: &AdmissionRequest,
    fleet: &FleetState,
    cfg: &AdmissionConfig,
) -> AdmissionDecision {
    let pressure = fleet_pressure(fleet, cfg);

    // 1. TTFT budget: doomed requests fail fast (don't waste prefill).
    if req.est_ttft_ms > cfg.ttft_slo_ms {
        let overshoot = req.est_ttft_ms - cfg.ttft_slo_ms;
        return AdmissionDecision {
            action: Action::Reject,
            reason: format!(
                "deadline infeasible: est TTFT {:.0}ms > SLO {:.0}ms",
                req.est_ttft_ms, cfg.ttft_slo_ms
            ),
            retry_after_ms: retry_after(pressure, cfg, overshoot),
            pressure,
            admit_floor: 0.0,
        };
    }

    // 2. Below the shed threshold: admit everything that meets its deadline.
    if pressure < cfg.shed_pressure {
        return AdmissionDecision {
            action: Action::Admit,
            reason: "within SLO, fleet below shed threshold".to_string(),
            retry_after_ms: 0.0,
            pressure,
            admit_floor: 0.0,
        };
    }

    // 3. Priority shedding: keep high tiers, shed (or queue) low tiers.
    let floor = admit_floor(pressure, cfg);
    if (tier_priority(req.tier) as f64) >= floor {
        return AdmissionDecision {
            action: Action::Admit,
            reason: format!(
                "tier {} clears admit floor {:.2} under pressure {:.2}",
                req.tier, floor, pressure
            ),
            retry_after_ms: 0.0,
            pressure,
            admit_floor: floor,
        };
    }

    if cfg.queue_enabled && fleet.queue_depth < cfg.max_queue_depth {
        return AdmissionDecision {
            action: Action::Queue,
            reason: format!("tier {} below floor {:.2}; queued for backpressure", req.tier, floor),
            retry_after_ms: retry_after(pressure, cfg, 0.0),
            pressure,
            admit_floor: floor,
        };
    }

    AdmissionDecision {
        action: Action::Reject,
        reason: format!("tier {} below floor {:.2} and queue full; shed", req.tier, floor),
        retry_after_ms: retry_after(pressure, cfg, 0.0),
        pressure,
        admit_floor: floor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(ttft: f64, tier: &str) -> AdmissionRequest {
        AdmissionRequest { est_ttft_ms: ttft, tier, est_output_tokens: 256 }
    }

    fn fleet_with_inflight(per_backend: &[u32]) -> FleetState {
        let mut f = FleetState::default();
        for (i, &n) in per_backend.iter().enumerate() {
            f.backend_inflight.insert(format!("b{i}"), n);
        }
        f
    }

    #[test]
    fn tier_priority_ordering() {
        assert!(tier_priority("gold") > tier_priority("silver"));
        assert!(tier_priority("silver") > tier_priority("bronze"));
        assert!(tier_priority("bronze") > tier_priority("anonymous"));
        assert_eq!(tier_priority("GOLD"), 3); // case-insensitive
        assert_eq!(tier_priority("nonsense"), 0);
    }

    #[test]
    fn fail_fast_when_ttft_exceeds_slo() {
        let cfg = AdmissionConfig::default();
        let d = decide(&req(10_000.0, "gold"), &FleetState::default(), &cfg);
        assert_eq!(d.action, Action::Reject); // even gold can't beat physics
        assert!(d.retry_after_ms > 0.0);
    }

    #[test]
    fn admits_everything_below_shed_pressure() {
        let cfg = AdmissionConfig::default();
        // empty fleet -> pressure 0 -> admit any tier within deadline
        let d = decide(&req(50.0, "bronze"), &FleetState::default(), &cfg);
        assert_eq!(d.action, Action::Admit);
        assert_eq!(d.admit_floor, 0.0);
    }

    #[test]
    fn fleet_pressure_is_max_of_kv_and_saturated_fraction() {
        let cfg = AdmissionConfig::default();
        let mut f = fleet_with_inflight(&[cfg.max_inflight_per_backend, 0, 0]); // 1/3 saturated
        // mean kv 0.10 (< 0.333 sat frac) -> pressure follows sat frac
        f.backend_kv_usage.insert("b0".into(), 0.10);
        let p = fleet_pressure(&f, &cfg);
        assert!((p - 1.0 / 3.0).abs() < 1e-9, "got {p}");
    }

    #[test]
    fn sheds_low_tier_protects_gold_at_full_pressure() {
        let cfg = AdmissionConfig::default();
        let cap = cfg.max_inflight_per_backend;
        let fleet = fleet_with_inflight(&[cap, cap, cap]); // pressure 1.0 -> floor 3.0
        let gold = decide(&req(50.0, "gold"), &fleet, &cfg);
        let bronze = decide(&req(50.0, "bronze"), &fleet, &cfg);
        assert_eq!(gold.action, Action::Admit); // gold (3) clears floor 3.0
        assert_eq!(bronze.action, Action::Queue); // bronze (1) < 3.0 -> queued (queue on)
        assert!((gold.pressure - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_when_queue_disabled_and_below_floor() {
        let mut cfg = AdmissionConfig::default();
        cfg.queue_enabled = false;
        let cap = cfg.max_inflight_per_backend;
        let fleet = fleet_with_inflight(&[cap, cap, cap]);
        let bronze = decide(&req(50.0, "bronze"), &fleet, &cfg);
        assert_eq!(bronze.action, Action::Reject);
        assert!(bronze.retry_after_ms > 0.0);
    }

    #[test]
    fn rejects_when_queue_full() {
        let cfg = AdmissionConfig::default();
        let cap = cfg.max_inflight_per_backend;
        let mut fleet = fleet_with_inflight(&[cap, cap, cap]);
        fleet.queue_depth = cfg.max_queue_depth; // full
        let bronze = decide(&req(50.0, "bronze"), &fleet, &cfg);
        assert_eq!(bronze.action, Action::Reject);
    }

    #[test]
    fn retry_after_capped_at_max() {
        let cfg = AdmissionConfig::default();
        // a huge overshoot must still be clamped to retry_after_max_ms
        let d = decide(&req(1_000_000.0, "bronze"), &FleetState::default(), &cfg);
        assert_eq!(d.action, Action::Reject);
        assert!(d.retry_after_ms <= cfg.retry_after_max_ms);
    }
}
