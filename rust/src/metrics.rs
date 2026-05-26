//! Prometheus text-exposition metrics. Mirrors `gateway/metrics.py`.
//!
//! Instance-based (not a global registry). Counters/gauges/histograms with
//! cumulative `le` buckets. Same metric names as the Python version so the
//! Grafana dashboard works against either gateway.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::Mutex;

pub const LATENCY_BUCKETS: &[f64] =
    &[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25];
pub const BLOCK_BUCKETS: &[f64] =
    &[0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0];

type LabelKey = Vec<(String, String)>;
type SeriesKey = (String, LabelKey);

struct Hist {
    buckets: Vec<f64>,
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

#[derive(Default)]
struct Inner {
    counters: BTreeMap<SeriesKey, f64>,
    gauges: BTreeMap<SeriesKey, f64>,
    hists: BTreeMap<SeriesKey, Hist>,
    meta: BTreeMap<String, (&'static str, String)>, // name -> (type, help)
}

pub struct MetricsCollector {
    inner: Mutex<Inner>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        MetricsCollector { inner: Mutex::new(Inner::default()) }
    }
}

fn key(name: &str, labels: &[(&str, &str)]) -> SeriesKey {
    let mut kv: Vec<(String, String)> = labels
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    kv.sort_by(|a, b| a.0.cmp(&b.0));
    (name.to_string(), kv)
}

fn fmt_labels(labels: &LabelKey, extra: &[(&str, &str)]) -> String {
    if labels.is_empty() && extra.is_empty() {
        return String::new();
    }
    let mut all: Vec<(String, String)> = labels.clone();
    for (k, v) in extra {
        all.push(((*k).to_string(), (*v).to_string()));
    }
    let inner = all
        .iter()
        .map(|(k, v)| format!("{k}=\"{v}\""))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{}}}", inner)
}

fn fmt_le(ub: f64) -> String {
    if ub.is_infinite() {
        "+Inf".to_string()
    } else {
        format!("{}", ub)
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inc_counter(&self, name: &str, value: f64, help: &str, labels: &[(&str, &str)]) {
        let mut inner = self.inner.lock().unwrap();
        inner
            .meta
            .entry(name.to_string())
            .or_insert(("counter", help.to_string()));
        let k = key(name, labels);
        *inner.counters.entry(k).or_insert(0.0) += value;
    }

    pub fn set_gauge(&self, name: &str, value: f64, help: &str, labels: &[(&str, &str)]) {
        let mut inner = self.inner.lock().unwrap();
        inner
            .meta
            .entry(name.to_string())
            .or_insert(("gauge", help.to_string()));
        let k = key(name, labels);
        inner.gauges.insert(k, value);
    }

    pub fn observe(
        &self,
        name: &str,
        value: f64,
        buckets: &[f64],
        help: &str,
        labels: &[(&str, &str)],
    ) {
        let mut inner = self.inner.lock().unwrap();
        inner
            .meta
            .entry(name.to_string())
            .or_insert(("histogram", help.to_string()));
        let k = key(name, labels);
        let entry = inner.hists.entry(k).or_insert_with(|| Hist {
            buckets: buckets.to_vec(),
            counts: vec![0; buckets.len()],
            sum: 0.0,
            count: 0,
        });
        for (i, ub) in entry.buckets.iter().enumerate() {
            if value <= *ub {
                entry.counts[i] += 1;
            }
        }
        entry.sum += value;
        entry.count += 1;
    }

    pub fn render(&self) -> String {
        let inner = self.inner.lock().unwrap();
        let mut out = String::new();
        let mut emitted: BTreeSet<String> = BTreeSet::new();

        let mut header = |out: &mut String, name: &str, emitted: &mut BTreeSet<String>| {
            if emitted.contains(name) {
                return;
            }
            emitted.insert(name.to_string());
            if let Some((typ, help)) = inner.meta.get(name) {
                if !help.is_empty() {
                    let _ = writeln!(out, "# HELP {name} {help}");
                }
                let _ = writeln!(out, "# TYPE {name} {typ}");
            }
        };

        for ((name, labels), val) in inner.counters.iter() {
            header(&mut out, name, &mut emitted);
            let _ = writeln!(out, "{}{} {}", name, fmt_labels(labels, &[]), val);
        }
        for ((name, labels), val) in inner.gauges.iter() {
            header(&mut out, name, &mut emitted);
            let _ = writeln!(out, "{}{} {}", name, fmt_labels(labels, &[]), val);
        }
        for ((name, labels), h) in inner.hists.iter() {
            header(&mut out, name, &mut emitted);
            for (i, ub) in h.buckets.iter().enumerate() {
                let le = fmt_le(*ub);
                let _ = writeln!(
                    out,
                    "{}_bucket{} {}",
                    name,
                    fmt_labels(labels, &[("le", &le)]),
                    h.counts[i]
                );
            }
            let _ = writeln!(
                out,
                "{}_bucket{} {}",
                name,
                fmt_labels(labels, &[("le", "+Inf")]),
                h.count
            );
            let _ = writeln!(out, "{}_sum{} {}", name, fmt_labels(labels, &[]), h.sum);
            let _ = writeln!(out, "{}_count{} {}", name, fmt_labels(labels, &[]), h.count);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_accumulates_per_label_set() {
        let m = MetricsCollector::new();
        m.inc_counter("reqs_total", 1.0, "", &[("backend", "b0")]);
        m.inc_counter("reqs_total", 1.0, "", &[("backend", "b0")]);
        m.inc_counter("reqs_total", 1.0, "", &[("backend", "b1")]);
        let out = m.render();
        assert!(out.contains("reqs_total{backend=\"b0\"} 2"));
        assert!(out.contains("reqs_total{backend=\"b1\"} 1"));
    }

    #[test]
    fn gauge_overwrites() {
        let m = MetricsCollector::new();
        m.set_gauge("inflight", 5.0, "", &[("backend", "b0")]);
        m.set_gauge("inflight", 2.0, "", &[("backend", "b0")]);
        assert!(m.render().contains("inflight{backend=\"b0\"} 2"));
    }

    #[test]
    fn histogram_buckets_cumulative_sum_count() {
        let m = MetricsCollector::new();
        m.observe("lat_seconds", 0.0003, LATENCY_BUCKETS, "", &[]);
        m.observe("lat_seconds", 0.002, LATENCY_BUCKETS, "", &[]);
        m.observe("lat_seconds", 0.02, LATENCY_BUCKETS, "", &[]);
        let out = m.render();
        assert!(out.contains("lat_seconds_bucket{le=\"0.001\"} 1"));
        assert!(out.contains("lat_seconds_bucket{le=\"0.0025\"} 2"));
        assert!(out.contains("lat_seconds_bucket{le=\"+Inf\"} 3"));
        assert!(out.contains("lat_seconds_count 3"));
    }

    #[test]
    fn render_has_type_headers() {
        let m = MetricsCollector::new();
        m.inc_counter("c_total", 1.0, "a counter", &[]);
        m.set_gauge("g", 1.0, "", &[]);
        m.observe("h", 0.01, LATENCY_BUCKETS, "", &[]);
        let out = m.render();
        assert!(out.contains("# TYPE c_total counter"));
        assert!(out.contains("# HELP c_total a counter"));
        assert!(out.contains("# TYPE g gauge"));
        assert!(out.contains("# TYPE h histogram"));
    }
}
