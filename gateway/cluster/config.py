from __future__ import annotations

"""Configuration for the multi-replica prefix-state cluster.

Standalone from the gateway's main Config so the replication subsystem can be
tuned without touching the benchmarked hot path. `from_env()` reads GW_CLUSTER_*
mirroring Config.from_env(). Disabled by default -- a single-replica gateway
behaves exactly as before.
"""

import os
import socket
from dataclasses import dataclass


def _default_replica_id() -> str:
    # Stable-ish within a process; unique across replicas (host + pid).
    return f"{socket.gethostname()}-{os.getpid()}"


@dataclass
class ClusterConfig:
    enabled: bool = False

    # Identity of THIS replica (must be unique across the fleet). Auto-derived
    # from hostname+pid if left blank.
    replica_id: str = ""

    # Transport for the mutation stream:
    #   memory -- in-process broker (single process; for tests / 1-process demos)
    #   redis  -- Redis pub/sub (a central broker; simple ops)
    #   gossip -- peer-to-peer HTTP: replicas push events to each other directly,
    #             NO broker. Set `peers` to the other replicas' base URLs.
    transport: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    channel: str = "gw:prefix"
    # Comma-separated base URLs of the OTHER replicas (gossip transport only),
    # e.g. "http://gw2:8000,http://gw3:8000". Self is excluded.
    peers: str = ""
    # Shared secret authenticating peer gossip. When set, outbound pushes carry it
    # (X-Cluster-Secret) and POST /cluster/gossip rejects requests without it, so a
    # rogue host can't inject routing state. Empty = trust the network (set it in prod).
    secret: str = ""

    # Background drain cadence (ms). The hot path only *buffers* a publish;
    # this loop flushes outbound + applies inbound, off the request path.
    sync_interval_ms: int = 250

    @classmethod
    def from_env(cls) -> "ClusterConfig":
        cfg = cls()
        for f in cls.__dataclass_fields__:
            env = os.environ.get(f"GW_CLUSTER_{f.upper()}")
            if env is None:
                continue
            cur = getattr(cfg, f)
            if isinstance(cur, bool):
                setattr(cfg, f, env.lower() in ("1", "true", "yes"))
            elif isinstance(cur, int):
                setattr(cfg, f, int(env))
            elif isinstance(cur, float):
                setattr(cfg, f, float(env))
            else:
                setattr(cfg, f, env)
        if not cfg.replica_id:
            cfg.replica_id = _default_replica_id()
        return cfg
