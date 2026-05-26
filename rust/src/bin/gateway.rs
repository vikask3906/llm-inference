//! Gateway binary entrypoint: `cargo run --bin gateway`.
//!
//! Env:
//!   GW_ADDR                bind address (default 0.0.0.0:8000)
//!   GW_BACKENDS            "id=url,..." (default 3 localhost mock backends)
//!   GW_STRATEGY            round_robin | consistent_hash | prefix_tree (default prefix_tree)
//!   GW_RATE_LIMIT_ENABLED  true|false (default false; tenancy is opt-in)
//!   GW_TENANTS             "key=tenant:tier,..."  tier in {gold,silver,bronze}
//!   GW_PREFIX_ISOLATION    tenant | global (default tenant)

use std::env;

use gateway_core::config::{Config, Strategy};
use gateway_core::server::{app, build_state, Backend};

fn env_bool(name: &str) -> Option<bool> {
    env::var(name).ok().map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

#[tokio::main]
async fn main() {
    let mut cfg = Config::default();
    if let Ok(s) = env::var("GW_STRATEGY") {
        cfg.strategy = match s.as_str() {
            "round_robin" => Strategy::RoundRobin,
            "consistent_hash" => Strategy::ConsistentHash,
            _ => Strategy::PrefixTree,
        };
    }
    if let Some(b) = env_bool("GW_RATE_LIMIT_ENABLED") {
        cfg.rate_limit_enabled = b;
    }
    if let Ok(v) = env::var("GW_TENANTS") {
        cfg.tenants = v;
    }
    if let Ok(v) = env::var("GW_PREFIX_ISOLATION") {
        cfg.prefix_isolation = v;
    }
    if let Ok(v) = env::var("GW_LOG_LEVEL") {
        cfg.log_level = v;
    }

    let spec = env::var("GW_BACKENDS").unwrap_or_else(|_| {
        "b0=http://localhost:9001,b1=http://localhost:9002,b2=http://localhost:9003".to_string()
    });
    let backends: Vec<Backend> = spec
        .split(',')
        .filter_map(|p| {
            let p = p.trim();
            let (id, url) = p.split_once('=')?;
            Some(Backend { id: id.trim().to_string(), url: url.trim().to_string() })
        })
        .collect();

    let addr = env::var("GW_ADDR").unwrap_or_else(|_| "0.0.0.0:8000".to_string());
    let state = build_state(cfg, backends);
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    println!("gateway listening on {addr}");
    axum::serve(listener, app(state)).await.expect("serve");
}
