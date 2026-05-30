//! Fast Rust load generator. The Python `httpx` benchmark client caps at
//! ~370 req/s on a single host (GIL + per-request streaming overhead) — too
//! slow to differentiate the gateways' true throughput ceilings. This driver
//! uses tokio + reqwest to push real concurrency.
//!
//! Usage:
//!   loadgen --url http://127.0.0.1:8000 --n 20000 --concurrency 128 --label rust

use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn parse_args() -> (String, usize, usize, String) {
    let mut url = String::new();
    let mut n: usize = 10_000;
    let mut concurrency: usize = 64;
    let mut label = String::from("loadgen");
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--url" => url = args.next().unwrap_or_default(),
            "--n" => n = args.next().and_then(|s| s.parse().ok()).unwrap_or(n),
            "--concurrency" | "-c" => {
                concurrency = args.next().and_then(|s| s.parse().ok()).unwrap_or(concurrency)
            }
            "--label" => label = args.next().unwrap_or(label),
            _ => {}
        }
    }
    if url.is_empty() {
        eprintln!("error: --url required");
        std::process::exit(2);
    }
    (url, n, concurrency, label)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((q * sorted.len() as f64) as usize).min(sorted.len() - 1);
    sorted[i]
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let (url, n, concurrency, label) = parse_args();
    let endpoint = format!("{}/v1/chat/completions", url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(concurrency * 2)
        .build()
        .expect("client");

    let body = serde_json::json!({
        "model": "mock-model",
        "messages": [
            {"role": "system", "content": format!("You are a helpful assistant. {}", "context ".repeat(40))},
            {"role": "user", "content": "hi"}
        ],
        "stream": true
    });
    let body_bytes = Arc::new(serde_json::to_vec(&body).unwrap());

    let per_worker = n / concurrency.max(1);
    let lats: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));
    let errors = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();
    let mut handles = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let client = client.clone();
        let endpoint = endpoint.clone();
        let body_bytes = body_bytes.clone();
        let lats = lats.clone();
        let errors = errors.clone();
        handles.push(tokio::spawn(async move {
            let mut local = Vec::<f64>::with_capacity(per_worker);
            for _ in 0..per_worker {
                let t0 = Instant::now();
                let req = client
                    .post(&endpoint)
                    .header("content-type", "application/json")
                    .body((*body_bytes).clone())
                    .send()
                    .await;
                match req {
                    Ok(resp) => match resp.bytes().await {
                        Ok(_) => local.push(t0.elapsed().as_secs_f64() * 1000.0),
                        Err(_) => {
                            errors.fetch_add(1, Ordering::Relaxed);
                        }
                    },
                    Err(_) => {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            lats.lock().unwrap().extend(local);
        }));
    }
    for h in handles {
        let _ = h.await;
    }
    let wall = start.elapsed().as_secs_f64();

    let mut v = Arc::try_unwrap(lats).expect("arc").into_inner().unwrap();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thru = v.len() as f64 / wall.max(1e-9);
    let errs = errors.load(Ordering::Relaxed);

    println!(
        "{label:<10} n={:<6} conc={:<4} errors={:<3} throughput={:>7.0} req/s  \
         p50={:>6.2}  p95={:>6.2}  p99={:>6.2}  mean={:>6.2}  (ms)",
        v.len(),
        concurrency,
        errs,
        thru,
        percentile(&v, 0.50),
        percentile(&v, 0.95),
        percentile(&v, 0.99),
        v.iter().sum::<f64>() / v.len().max(1) as f64,
    );
}
