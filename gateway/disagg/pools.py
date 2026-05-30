from __future__ import annotations

"""Backend pool roles for disaggregated routing.

A backend can be a prefill node, a decode node, or both. The spec string maps
backend ids to roles, e.g. "b0:prefill;b1:decode;b2:prefill,decode". Anything
not listed is assumed to serve BOTH phases (the safe default, identical to a
non-disaggregated fleet) so an unconfigured deployment keeps working.
"""

ROLE_PREFILL = "prefill"
ROLE_DECODE = "decode"
_VALID_ROLES = {ROLE_PREFILL, ROLE_DECODE}


def parse_pool_config(spec: str) -> dict[str, set[str]]:
    """Parse "b0:prefill;b1:decode;b2:prefill,decode" into {backend_id: {roles}}.

    Whitespace is tolerated; unknown roles and malformed segments are dropped.
    A backend listed with no valid role is omitted (callers treat it as 'both').
    """
    out: dict[str, set[str]] = {}
    if not spec:
        return out
    for seg in spec.split(";"):
        seg = seg.strip()
        if not seg or ":" not in seg:
            continue
        bid, _, roles = seg.partition(":")
        bid = bid.strip()
        if not bid:
            continue
        role_set = {r.strip().lower() for r in roles.split(",")}
        role_set &= _VALID_ROLES
        if role_set:
            out[bid] = role_set
    return out


class PoolRegistry:
    """Resolves which backends can serve each phase.

    Constructed from a list of backend ids and a parsed role map. A backend
    absent from the role map (or present with no valid role) defaults to BOTH
    phases, so the registry degrades to a homogeneous fleet when unconfigured.
    """

    def __init__(self, backend_ids: list[str], roles: dict[str, set[str]]):
        self._roles: dict[str, set[str]] = {}
        for bid in backend_ids:
            r = roles.get(bid)
            self._roles[bid] = set(r) if r else {ROLE_PREFILL, ROLE_DECODE}

    @classmethod
    def from_spec(cls, backend_ids: list[str], spec: str) -> "PoolRegistry":
        return cls(backend_ids, parse_pool_config(spec))

    def roles(self, backend_id: str) -> set[str]:
        return self._roles.get(backend_id, set())

    def is_prefill(self, backend_id: str) -> bool:
        return ROLE_PREFILL in self._roles.get(backend_id, set())

    def is_decode(self, backend_id: str) -> bool:
        return ROLE_DECODE in self._roles.get(backend_id, set())

    def prefill_pool(self) -> list[str]:
        return [b for b, r in self._roles.items() if ROLE_PREFILL in r]

    def decode_pool(self) -> list[str]:
        return [b for b, r in self._roles.items() if ROLE_DECODE in r]

    def colocatable(self) -> list[str]:
        """Backends that can serve a whole request alone (both phases)."""
        return [b for b, r in self._roles.items() if _VALID_ROLES <= r]
