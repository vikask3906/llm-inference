from __future__ import annotations

"""Minimal, instance-based metrics collector that renders the Prometheus text
exposition format.

Instance-based (not a global registry) so it composes with the rest of the
gateway's per-instance state and is trivially testable. In production this would
be swapped for prometheus_client; the exposition format here is wire-compatible.
"""

import math
import threading

LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
BLOCK_BUCKETS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def _fmt_labels(labels: tuple, **extra) -> str:
    items = list(labels) + list(extra.items())
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in items) + "}"


def _le(ub: float) -> str:
    return "+Inf" if math.isinf(ub) else f"{ub:g}"


class MetricsCollector:
    def __init__(self) -> None:
        self._counters: dict[tuple, float] = {}
        self._gauges: dict[tuple, float] = {}
        self._hist: dict[tuple, dict] = {}
        self._meta: dict[str, tuple[str, str]] = {}   # name -> (type, help)
        self._lock = threading.Lock()

    @staticmethod
    def _key(name: str, labels: dict) -> tuple:
        return (name, tuple(sorted(labels.items())))

    def inc_counter(self, name: str, value: float = 1.0, help: str = "", **labels) -> None:
        with self._lock:
            self._meta[name] = ("counter", help)
            k = self._key(name, labels)
            self._counters[k] = self._counters.get(k, 0.0) + value

    def set_gauge(self, name: str, value: float, help: str = "", **labels) -> None:
        with self._lock:
            self._meta[name] = ("gauge", help)
            self._gauges[self._key(name, labels)] = float(value)

    def observe(self, name: str, value: float, buckets=None, help: str = "", **labels) -> None:
        with self._lock:
            self._meta[name] = ("histogram", help)
            k = self._key(name, labels)
            h = self._hist.get(k)
            if h is None:
                b = list(buckets or LATENCY_BUCKETS)
                h = {"buckets": b, "counts": [0] * len(b), "sum": 0.0, "count": 0}
                self._hist[k] = h
            for i, ub in enumerate(h["buckets"]):
                if value <= ub:
                    h["counts"][i] += 1       # cumulative "le" semantics
            h["sum"] += value
            h["count"] += 1

    def render(self) -> str:
        lines: list[str] = []
        emitted: set[str] = set()

        def header(name: str) -> None:
            if name in emitted:
                return
            emitted.add(name)
            typ, hlp = self._meta.get(name, ("untyped", ""))
            if hlp:
                lines.append(f"# HELP {name} {hlp}")
            lines.append(f"# TYPE {name} {typ}")

        with self._lock:
            for (name, labels), val in sorted(self._counters.items()):
                header(name)
                lines.append(f"{name}{_fmt_labels(labels)} {val}")
            for (name, labels), val in sorted(self._gauges.items()):
                header(name)
                lines.append(f"{name}{_fmt_labels(labels)} {val}")
            for (name, labels), h in sorted(self._hist.items()):
                header(name)
                for ub, c in zip(h["buckets"], h["counts"]):
                    lines.append(f'{name}_bucket{_fmt_labels(labels, le=_le(ub))} {c}')
                lines.append(f'{name}_bucket{_fmt_labels(labels, le="+Inf")} {h["count"]}')
                lines.append(f'{name}_sum{_fmt_labels(labels)} {h["sum"]}')
                lines.append(f'{name}_count{_fmt_labels(labels)} {h["count"]}')

        return "\n".join(lines) + "\n"
