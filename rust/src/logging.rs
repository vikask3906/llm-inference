//! Structured JSON request logging (mirrors `gateway/logging_setup.py`).
//!
//! One JSON line per request to stdout, carrying `request_id` (correlatable
//! with the `x-request-id` response header) + duration + outcome fields.
//! `log_level = "WARN"` suppresses per-request lines for the benchmark parity
//! comparison (Python does the same).

use chrono::{SecondsFormat, Utc};
use serde_json::{json, Value};

pub fn now_iso() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Micros, true)
}

pub fn new_request_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Per-request log line. Fields with `None` are omitted.
pub struct RequestLog<'a> {
    pub level: &'a str,           // "INFO"
    pub request_id: &'a str,
    pub tenant: Option<&'a str>,
    pub model: Option<&'a str>,
    pub strategy: Option<&'a str>,
    pub backend: Option<&'a str>,
    pub match_blocks: Option<usize>,
    pub cache_hit: Option<bool>,
    pub retries: Option<u32>,
    pub status: u16,
    pub reason: Option<&'a str>,
    pub duration_ms: f64,
    pub output_tokens: Option<u64>,
}

impl<'a> RequestLog<'a> {
    pub fn emit(&self) {
        let mut obj = serde_json::Map::new();
        obj.insert("ts".into(), Value::String(now_iso()));
        obj.insert("level".into(), Value::String(self.level.to_string()));
        obj.insert("logger".into(), Value::String("gateway".to_string()));
        obj.insert("msg".into(), Value::String("request".to_string()));
        obj.insert("request_id".into(), Value::String(self.request_id.to_string()));
        if let Some(v) = self.tenant {
            obj.insert("tenant".into(), Value::String(v.to_string()));
        }
        if let Some(v) = self.model {
            obj.insert("model".into(), Value::String(v.to_string()));
        }
        if let Some(v) = self.strategy {
            obj.insert("strategy".into(), Value::String(v.to_string()));
        }
        if let Some(v) = self.backend {
            obj.insert("backend".into(), Value::String(v.to_string()));
        }
        if let Some(v) = self.match_blocks {
            obj.insert("match_blocks".into(), json!(v));
        }
        if let Some(v) = self.cache_hit {
            obj.insert("cache_hit".into(), Value::Bool(v));
        }
        if let Some(v) = self.retries {
            obj.insert("retries".into(), json!(v));
        }
        obj.insert("status".into(), json!(self.status));
        if let Some(v) = self.reason {
            obj.insert("reason".into(), Value::String(v.to_string()));
        }
        // round to 2 decimals for readability, matching Python
        let dur = (self.duration_ms * 100.0).round() / 100.0;
        obj.insert("duration_ms".into(), json!(dur));
        if let Some(v) = self.output_tokens {
            obj.insert("output_tokens".into(), json!(v));
        }
        let line = serde_json::to_string(&Value::Object(obj)).unwrap();
        // single println acquires the stdout lock — fine at moderate rates;
        // benchmark parity runs use log_level=WARN to suppress.
        println!("{}", line);
    }
}

/// True iff `log_level` is INFO-or-lower (i.e., per-request lines should be emitted).
pub fn info_enabled(log_level: &str) -> bool {
    matches!(log_level.to_ascii_uppercase().as_str(), "DEBUG" | "INFO")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_iso_format() {
        let s = now_iso();
        assert!(s.contains('T') && s.ends_with('Z'));
    }

    #[test]
    fn new_request_id_is_hex_32() {
        let id = new_request_id();
        assert_eq!(id.len(), 32);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn info_level_gating() {
        assert!(info_enabled("INFO"));
        assert!(info_enabled("info"));
        assert!(!info_enabled("WARN"));
        assert!(!info_enabled("ERROR"));
    }
}
