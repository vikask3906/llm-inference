from __future__ import annotations

"""Routing strategies.

- round_robin     : naive load balancer, cache-blind (the baseline to beat).
- consistent_hash : deterministic affinity on the FIRST block only. No
                    longest-prefix match, no eviction awareness -> collapses
                    families that share a system prompt onto one backend.
- prefix_tree     : longest-prefix match via the radix tree, scored by estimated
                    TTFT (affinity + load) with saturation cutoff + hysteresis.
"""

from .config import Config
from .hashing import block_hashes
from .load_tracker import LoadTracker
from .radix_tree import RadixTree


class RouteResult:
    __slots__ = ("backend_id", "hashes", "tokens", "match_blocks")

    def __init__(self, backend_id, hashes, tokens, match_blocks):
        self.backend_id = backend_id
        self.hashes = hashes
        self.tokens = tokens
        self.match_blocks = match_blocks


class Router:
    def __init__(self, cfg: Config, tree: RadixTree, load: LoadTracker):
        self.cfg = cfg
        self.tree = tree
        self.load = load
        self._rr = 0

    def _candidates(self, backends: list[str]) -> list[str]:
        return backends

    def choose(self, prompt: str, backends: list[str], strategy: str | None = None,
               seed: int = 0) -> RouteResult:
        strategy = strategy or self.cfg.strategy
        hashes = block_hashes(prompt, self.cfg.block_chars, self.cfg.hash_cutoff_blocks, seed=seed)
        tokens = len(hashes) * self.cfg.block_tokens
        cands = self._candidates(backends)

        if strategy == "round_robin":
            b = cands[self._rr % len(cands)]
            self._rr += 1
            return RouteResult(b, hashes, tokens, 0)

        if strategy == "consistent_hash":
            key = hashes[0] if hashes else 0
            b = cands[key % len(cands)]
            return RouteResult(b, hashes, tokens, 0)

        # prefix_tree: minimize estimated TTFT, then load-balance within hysteresis
        match = self.tree.match(hashes)
        scored = []                            # (ttft, backend, match_blocks)
        saturated_fallback = None
        for b in cands:
            # --- guardrails: hard saturation cutoff (affinity yields to load) ---
            if self.load.kv_usage[b] > self.cfg.kv_pressure_cutoff or \
               self.load.inflight[b] > self.cfg.max_inflight:
                if saturated_fallback is None or \
                   self.load.inflight[b] < self.load.inflight[saturated_fallback]:
                    saturated_fallback = b
                continue
            m_blocks = match.get(b, 0)
            scored.append((self._est_ttft(tokens, m_blocks, b), b, m_blocks))

        if not scored:                         # everything saturated
            b = saturated_fallback or cands[0]
            return RouteResult(b, hashes, tokens, match.get(b, 0))

        # Backends whose TTFT is within hysteresis of the best are "equivalent":
        # spread across them by least-load (+ round-robin) so a tiny shared prefix
        # (e.g. a common system prompt) doesn't pin all traffic to one node, while
        # a large real affinity still keeps a single backend the sole winner.
        min_ttft = min(s[0] for s in scored)
        good = [(b, m) for (ttft, b, m) in scored if ttft <= min_ttft + self.cfg.hysteresis_ms]
        min_load = min(self.load.inflight[b] for b, _ in good)
        tied = [(b, m) for (b, m) in good if self.load.inflight[b] == min_load]
        b, m_blocks = tied[self._rr % len(tied)]
        self._rr += 1
        return RouteResult(b, hashes, tokens, m_blocks)

    def _est_ttft(self, tokens: int, match_blocks: int, backend_id: str) -> float:
        cached_tokens = match_blocks * self.cfg.block_tokens
        uncached = max(0, tokens - cached_tokens)
        # MVP: linear prefill. Phase 2: a*(N-m)*N + b*(N-m) fitted from real timings.
        prefill_ms = self.cfg.prefill_ms_per_token * uncached
        queue_ms = self.load.inflight[backend_id] * self.cfg.service_ms_per_request
        return prefill_ms + queue_ms
