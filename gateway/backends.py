from __future__ import annotations

"""Backend registry. Parses "id=url" pairs and tracks per-backend health.

Routing is two-stage: first filter to backends serving the request's model, then
apply prefix-affinity + load within that pool. MVP assumes a homogeneous fleet
(all serve default_model); heterogeneous fleets are a Phase-2 extension.
"""

from .config import Config


class Backend:
    __slots__ = ("id", "url", "model", "healthy", "draining", "adapters")

    def __init__(self, id: str, url: str, model: str,
                 adapters: set[str] | None = None) -> None:
        self.id = id
        self.url = url
        self.model = model
        self.healthy = True
        # Operator-controlled maintenance flag: a draining backend stays alive and
        # finishes its in-flight work but receives NO new requests (excluded from
        # routing). Distinct from `healthy`, which the health scrape owns.
        self.draining = False
        # LoRA adapters loaded on this backend. Empty set = base only.
        self.adapters: set[str] = set(adapters) if adapters else set()


class BackendRegistry:
    def __init__(self, cfg: Config) -> None:
        self._by_id: dict[str, Backend] = {}
        self._default_model = cfg.default_model
        for pair in cfg.backends.split(","):
            pair = pair.strip()
            if not pair:
                continue
            bid, _, url = pair.partition("=")
            self._by_id[bid.strip()] = Backend(bid.strip(), url.strip(), cfg.default_model)

    def all(self) -> list[Backend]:
        return list(self._by_id.values())

    def add(self, backend_id: str, url: str, model: str | None = None) -> bool:
        """Add (or update the URL of) a backend at runtime. Returns True if it was
        newly created. A new backend starts unhealthy until the next health scrape."""
        new = backend_id not in self._by_id
        if new:
            b = Backend(backend_id, url, model or self._default_model)
            b.healthy = False                      # let the scrape confirm it's up
            self._by_id[backend_id] = b
        else:
            self._by_id[backend_id].url = url      # URL update is idempotent
        return new

    def remove(self, backend_id: str) -> bool:
        """Remove a backend from the fleet. Returns False if it didn't exist."""
        return self._by_id.pop(backend_id, None) is not None

    def url(self, backend_id: str) -> str:
        return self._by_id[backend_id].url

    def set_health(self, backend_id: str, healthy: bool) -> None:
        b = self._by_id.get(backend_id)
        if b:
            b.healthy = healthy

    def get(self, backend_id: str) -> Backend | None:
        return self._by_id.get(backend_id)

    def set_draining(self, backend_id: str, draining: bool) -> bool:
        """Mark a backend for maintenance. Returns False if it doesn't exist."""
        b = self._by_id.get(backend_id)
        if not b:
            return False
        b.draining = draining
        return True

    def ids_for(self, model: str | None) -> list[str]:
        # Exclude draining backends so new requests never route to a node under
        # maintenance; in-flight requests already dispatched there finish normally.
        return [
            b.id for b in self._by_id.values()
            if b.healthy and not b.draining and (model is None or model == b.model)
        ]
