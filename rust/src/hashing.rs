//! Chained block hashing that mirrors vLLM's automatic prefix caching.
//!
//! Each block's hash folds in the previous block's hash, so block `i` matches
//! only if the entire prefix up to `i` is identical (prefix-exact). A non-zero
//! `seed` namespaces the chain (per-tenant isolation). Byte-for-byte identical
//! to the Python `gateway/hashing.py`.

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

fn fnv1a(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Deterministic, process-independent seed from a string (e.g. a tenant id).
pub fn stable_seed(s: &str) -> u64 {
    fnv1a(s.as_bytes())
}

/// Chained hash of each FULL block, truncated to `cutoff_blocks`.
/// Partial trailing block is dropped (not a stable cache key).
pub fn block_hashes(prompt: &str, block_chars: usize, cutoff_blocks: usize, seed: u64) -> Vec<u64> {
    let raw = prompt.as_bytes();
    let n_full = raw.len() / block_chars;
    let n = n_full.min(cutoff_blocks);
    let mut out = Vec::with_capacity(n);
    let mut prev = seed;
    for i in 0..n {
        let chunk = &raw[i * block_chars..(i + 1) * block_chars];
        let mut buf = Vec::with_capacity(8 + chunk.len());
        buf.extend_from_slice(&prev.to_le_bytes());
        buf.extend_from_slice(chunk);
        prev = fnv1a(&buf);
        out.push(prev);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    const BC: usize = 64;

    #[test]
    fn deterministic() {
        let p = "X".repeat(200);
        assert_eq!(block_hashes(&p, BC, 100, 0), block_hashes(&p, BC, 100, 0));
    }

    #[test]
    fn full_blocks_only() {
        let p = "a".repeat(BC * 3 + 30); // 3 full blocks + partial
        assert_eq!(block_hashes(&p, BC, 100, 0).len(), 3);
    }

    #[test]
    fn cutoff_truncates() {
        let p = "a".repeat(BC * 10);
        assert_eq!(block_hashes(&p, BC, 4, 0).len(), 4);
    }

    #[test]
    fn chaining_is_prefix_exact() {
        let a = format!("{}{}", "X".repeat(BC), "A".repeat(BC));
        let b = format!("{}{}", "X".repeat(BC), "B".repeat(BC));
        let ha = block_hashes(&a, BC, 100, 0);
        let hb = block_hashes(&b, BC, 100, 0);
        assert_eq!(ha[0], hb[0]); // shared prefix block
        assert_ne!(ha[1], hb[1]); // divergent block
    }

    #[test]
    fn seed_namespaces() {
        let p = "X".repeat(200);
        let sa = stable_seed("acme");
        let sb = stable_seed("beta");
        assert_eq!(sa, stable_seed("acme"));
        assert_ne!(sa, sb);
        assert_ne!(block_hashes(&p, BC, 100, sa), block_hashes(&p, BC, 100, sb));
    }

    #[test]
    fn empty_prompt() {
        assert!(block_hashes("", BC, 100, 0).is_empty());
    }
}
