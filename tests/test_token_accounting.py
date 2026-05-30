"""Token-accurate cost accounting (gateway/extensions/bpe_hashing counters +
server estimate_prompt_tokens wiring).

Tokenizer loads are network/cache dependent, so the real-tokenizer path is
exercised by monkeypatching `tokenize_ids` (deterministic, offline); the
char-heuristic and fallback paths are tested directly.
"""

import gateway.extensions.bpe_hashing as bpe
import gateway.server as gw
from gateway.extensions.bpe_hashing import TokenizerUnavailable, count_tokens_with_fallback


# --- char heuristic path ----------------------------------------------------

def test_char_heuristic_when_tokenizer_disabled():
    n, mode = count_tokens_with_fallback("a" * 40, chars_per_token=4,
                                         tokenizer_name="gpt2", use_tokenizer=False)
    assert (n, mode) == (10, "char")


def test_floor_is_one():
    n, mode = count_tokens_with_fallback("", chars_per_token=4,
                                         tokenizer_name="gpt2", use_tokenizer=False)
    assert n == 1 and mode == "char"


# --- real tokenizer path (monkeypatched, offline) ---------------------------

def test_tokenizer_path_counts_real_tokens(monkeypatch):
    # A code-ish string the char heuristic would badly misjudge; pretend the
    # tokenizer splits it into 7 tokens.
    monkeypatch.setattr(bpe, "tokenize_ids", lambda p, name: [1, 2, 3, 4, 5, 6, 7])
    n, mode = count_tokens_with_fallback("def f(x): return x*x",
                                         chars_per_token=4, tokenizer_name="gpt2",
                                         use_tokenizer=True)
    assert (n, mode) == (7, "bpe")


def test_falls_back_to_char_when_tokenizer_unavailable(monkeypatch):
    def boom(p, name):
        raise TokenizerUnavailable("no tokenizer here")
    monkeypatch.setattr(bpe, "tokenize_ids", boom)
    n, mode = count_tokens_with_fallback("x" * 20, chars_per_token=4,
                                         tokenizer_name="bogus", use_tokenizer=True)
    assert (n, mode) == (5, "char")            # gracefully degraded, never crashes


# --- server wiring ----------------------------------------------------------

def test_server_estimate_honors_flag(monkeypatch):
    prompt = "x" * 400                          # char heuristic -> 100 tokens
    # default (flag off): char heuristic
    monkeypatch.setattr(gw.cfg, "token_accurate_accounting", False)
    assert gw.estimate_prompt_tokens(prompt) == 100

    # flag on + a tokenizer that "returns" 250 tokens -> accurate count used
    monkeypatch.setattr(gw.cfg, "token_accurate_accounting", True)
    monkeypatch.setattr(bpe, "tokenize_ids", lambda p, name: list(range(250)))
    assert gw.estimate_prompt_tokens(prompt) == 250
