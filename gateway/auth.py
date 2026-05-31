from __future__ import annotations

"""Optional API-key authentication (default OFF).

A production inference gateway must gate access. When `require` is on, a request
must carry `Authorization: Bearer <key>` with a key in the valid set (the
configured tenant keys plus any extra `GW_API_KEYS`); otherwise it is rejected
with 401. This reuses the tenancy bearer-key scheme, so an authenticated key
still resolves to its tenant/tier for quota + prefix isolation.

Kept standalone from the hot path: when `require` is False, `check()` is a
constant no-op, so an unconfigured gateway behaves exactly as before.
"""

import hashlib
import hmac

MISSING = "missing_api_key"
INVALID = "invalid_api_key"


def sha256_hex(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def extract_bearer(headers) -> str | None:
    """Pull the bearer token from an Authorization header, or None."""
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


def parse_api_keys(spec: str) -> set[str]:
    return {k.strip() for k in spec.split(",") if k.strip()}


class Authenticator:
    def __init__(self, valid_keys: set[str], require: bool,
                 valid_hashes: set[str] | None = None) -> None:
        self._keys = set(valid_keys)
        # Pre-hashed keys (hex sha256). Lets an operator configure ONLY hashes,
        # so a leaked config/env never exposes a usable key (defense in depth).
        self._hashes = {h.strip().lower() for h in (valid_hashes or set()) if h.strip()}
        self.require = require

    def is_valid(self, key: str | None) -> bool:
        if not key:
            return False
        # Compare against every known key with a constant-time primitive so a
        # rejected request's timing doesn't reveal a near-match.
        ok = False
        for k in self._keys:
            if hmac.compare_digest(key, k):
                ok = True
        if self._hashes:
            h = sha256_hex(key)
            for vh in self._hashes:
                if hmac.compare_digest(h, vh):
                    ok = True
        return ok

    def check(self, headers) -> tuple[bool, str | None]:
        """Return (authorized, reason). Always authorized when `require` is off."""
        if not self.require:
            return True, None
        key = extract_bearer(headers)
        if key is None:
            return False, MISSING
        if not self.is_valid(key):
            return False, INVALID
        return True, None
