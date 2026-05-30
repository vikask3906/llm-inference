//! BPE token-ID block hashing (parity with Python `gateway.extensions.bpe_hashing`).
//!
//! Hashes blocks of real BPE token IDs instead of raw character bytes, so
//! prefixes that tokenize identically but encode to slightly different bytes
//! hit the same cache key as vLLM uses. Same chained FNV-1a scheme as
//! `hashing::block_hashes`, just operating on token IDs.
//!
//! Compiled only with `--features bpe`. When the feature is off, the public
//! surface here is a no-op stub so the rest of the crate compiles unchanged.

#[cfg(feature = "bpe")]
mod inner {
    use std::sync::{Mutex, OnceLock};

    use tokenizers::Tokenizer;

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

    // Tokenizer is heavy to load; cache by path + reuse across requests.
    // OnceLock + Mutex<Option<_>> rather than RwLock so swapping the
    // tokenizer (e.g. for tests) is straightforward.
    fn cache() -> &'static Mutex<Option<(String, Tokenizer)>> {
        static C: OnceLock<Mutex<Option<(String, Tokenizer)>>> = OnceLock::new();
        C.get_or_init(|| Mutex::new(None))
    }

    /// Load a tokenizer.json from disk. Subsequent calls with the same path
    /// reuse the cached instance.
    pub fn load_tokenizer(path: &str) -> Result<(), String> {
        let mut slot = cache().lock().map_err(|e| e.to_string())?;
        if let Some((cached_path, _)) = slot.as_ref() {
            if cached_path == path {
                return Ok(());
            }
        }
        let tok = Tokenizer::from_file(path).map_err(|e| e.to_string())?;
        *slot = Some((path.to_string(), tok));
        Ok(())
    }

    /// Encode `prompt` with the previously loaded tokenizer. Returns the
    /// token ID sequence, or `Err` if no tokenizer is loaded / encoding fails.
    pub fn tokenize(prompt: &str) -> Result<Vec<u32>, String> {
        let slot = cache().lock().map_err(|e| e.to_string())?;
        let (_, tok) = slot.as_ref().ok_or("no tokenizer loaded")?;
        let enc = tok.encode(prompt, false).map_err(|e| e.to_string())?;
        Ok(enc.get_ids().to_vec())
    }

    fn hash_block(prev: u64, ids: &[u32]) -> u64 {
        let mut buf = Vec::with_capacity(8 + ids.len() * 4);
        buf.extend_from_slice(&prev.to_le_bytes());
        for &id in ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        fnv1a(&buf)
    }

    /// Chained hash of each FULL token-block of length `block_tokens`,
    /// truncated to `cutoff_blocks`. Parity with the Python implementation.
    pub fn bpe_block_hashes(
        prompt: &str,
        block_tokens: usize,
        cutoff_blocks: usize,
        seed: u64,
    ) -> Result<Vec<u64>, String> {
        let ids = tokenize(prompt)?;
        let n_full = ids.len() / block_tokens;
        let n = n_full.min(cutoff_blocks);
        let mut out = Vec::with_capacity(n);
        let mut prev = seed;
        for i in 0..n {
            let chunk = &ids[i * block_tokens..(i + 1) * block_tokens];
            prev = hash_block(prev, chunk);
            out.push(prev);
        }
        Ok(out)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // These tests require a tokenizer.json on disk. CI provides one via
        // `GW_TOKENIZER_FILE`; if absent, the test is a no-op so the suite
        // stays green in environments without a downloaded tokenizer.
        fn tokenizer_path() -> Option<String> {
            std::env::var("GW_TOKENIZER_FILE").ok()
        }

        #[test]
        fn loads_and_tokenizes_if_available() {
            let Some(p) = tokenizer_path() else { return };
            load_tokenizer(&p).expect("load tokenizer");
            let ids = tokenize("hello world").expect("encode");
            assert!(!ids.is_empty());
        }

        #[test]
        fn deterministic_chain_if_available() {
            let Some(p) = tokenizer_path() else { return };
            load_tokenizer(&p).expect("load tokenizer");
            let a = "the quick brown fox jumps over the lazy dog ".repeat(20);
            let h1 = bpe_block_hashes(&a, 16, 100, 0).unwrap();
            let h2 = bpe_block_hashes(&a, 16, 100, 0).unwrap();
            assert_eq!(h1, h2);
        }

        #[test]
        fn cutoff_truncates_if_available() {
            let Some(p) = tokenizer_path() else { return };
            load_tokenizer(&p).expect("load tokenizer");
            let a = "x ".repeat(2000);
            let h = bpe_block_hashes(&a, 16, 5, 0).unwrap();
            assert_eq!(h.len(), 5);
        }

        #[test]
        fn errors_without_tokenizer_loaded() {
            // Force the cache empty by loading a bogus path first (will fail)
            // then verifying tokenize() returns Err. Note: this test is best-
            // effort; if another test loaded a tokenizer earlier in the run,
            // the cache may still be populated. So we only assert when we know
            // it's empty.
            let mut slot = cache().lock().unwrap();
            *slot = None;
            drop(slot);
            assert!(tokenize("anything").is_err());
        }
    }
}

#[cfg(feature = "bpe")]
pub use inner::{bpe_block_hashes, load_tokenizer, tokenize};

#[cfg(not(feature = "bpe"))]
pub fn bpe_block_hashes(
    _prompt: &str,
    _block_tokens: usize,
    _cutoff_blocks: usize,
    _seed: u64,
) -> Result<Vec<u64>, String> {
    Err("rust gateway built without --features bpe".to_string())
}

#[cfg(not(feature = "bpe"))]
pub fn load_tokenizer(_path: &str) -> Result<(), String> {
    Err("rust gateway built without --features bpe".to_string())
}
