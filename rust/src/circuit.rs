//! Per-backend circuit breaker (mirrors `gateway/circuit.py`).
//!
//! Three states: closed → (threshold consecutive failures) → open → (cooldown
//! elapsed) → half-open → (probe success) closed / (probe failure) reopen.
//! Methods take an explicit `now: f64` for testability; the server passes a
//! monotonic timestamp.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Closed,
    HalfOpen,
    Open,
}

pub struct CircuitBreaker {
    pub fail_threshold: u32,
    pub cooldown_s: f64,
    fails: HashMap<String, u32>,
    opened_at: HashMap<String, f64>,
}

impl CircuitBreaker {
    pub fn new(fail_threshold: u32, cooldown_s: f64) -> Self {
        CircuitBreaker {
            fail_threshold,
            cooldown_s,
            fails: HashMap::new(),
            opened_at: HashMap::new(),
        }
    }

    pub fn state(&self, backend: &str, now: f64) -> State {
        match self.opened_at.get(backend) {
            None => State::Closed,
            Some(&t) => {
                if (now - t) >= self.cooldown_s {
                    State::HalfOpen
                } else {
                    State::Open
                }
            }
        }
    }

    pub fn state_code(&self, backend: &str, now: f64) -> u32 {
        match self.state(backend, now) {
            State::Closed => 0,
            State::HalfOpen => 1,
            State::Open => 2,
        }
    }

    pub fn allow(&self, backend: &str, now: f64) -> bool {
        self.state(backend, now) != State::Open
    }

    pub fn record_success(&mut self, backend: &str) {
        self.fails.insert(backend.to_string(), 0);
        self.opened_at.remove(backend);
    }

    pub fn record_failure(&mut self, backend: &str, now: f64) {
        if self.state(backend, now) == State::HalfOpen {
            // probe failed -> reopen + restart cooldown
            self.opened_at.insert(backend.to_string(), now);
            self.fails.insert(backend.to_string(), self.fail_threshold);
            return;
        }
        let f = self.fails.entry(backend.to_string()).or_insert(0);
        *f += 1;
        if *f >= self.fail_threshold {
            self.opened_at.insert(backend.to_string(), now);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_after_threshold() {
        let mut cb = CircuitBreaker::new(3, 5.0);
        assert!(cb.allow("b", 0.0));
        cb.record_failure("b", 0.0);
        cb.record_failure("b", 0.0);
        assert_eq!(cb.state("b", 0.0), State::Closed);
        cb.record_failure("b", 0.0);
        assert_eq!(cb.state("b", 0.0), State::Open);
        assert!(!cb.allow("b", 0.0));
    }

    #[test]
    fn half_open_after_cooldown() {
        let mut cb = CircuitBreaker::new(3, 5.0);
        for _ in 0..3 {
            cb.record_failure("b", 0.0);
        }
        assert_eq!(cb.state("b", 4.9), State::Open);
        assert_eq!(cb.state("b", 5.0), State::HalfOpen);
        assert!(cb.allow("b", 5.0));
    }

    #[test]
    fn success_closes_circuit() {
        let mut cb = CircuitBreaker::new(3, 5.0);
        for _ in 0..3 {
            cb.record_failure("b", 0.0);
        }
        cb.record_success("b");
        assert_eq!(cb.state("b", 0.0), State::Closed);
    }

    #[test]
    fn half_open_failure_reopens() {
        let mut cb = CircuitBreaker::new(3, 5.0);
        for _ in 0..3 {
            cb.record_failure("b", 0.0);
        }
        assert_eq!(cb.state("b", 5.0), State::HalfOpen);
        cb.record_failure("b", 5.0);
        assert_eq!(cb.state("b", 5.0), State::Open);
        assert_eq!(cb.state("b", 9.9), State::Open);
        assert_eq!(cb.state("b", 10.0), State::HalfOpen);
    }

    #[test]
    fn isolated_per_backend() {
        let mut cb = CircuitBreaker::new(2, 5.0);
        cb.record_failure("a", 0.0);
        cb.record_failure("a", 0.0);
        assert!(!cb.allow("a", 0.0));
        assert!(cb.allow("b", 0.0));
    }

    #[test]
    fn state_code_mapping() {
        let mut cb = CircuitBreaker::new(1, 5.0);
        assert_eq!(cb.state_code("b", 0.0), 0);
        cb.record_failure("b", 0.0);
        assert_eq!(cb.state_code("b", 0.0), 2);
        assert_eq!(cb.state_code("b", 5.0), 1);
    }
}
