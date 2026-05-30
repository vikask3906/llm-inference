"""Multi-replica prefix-state replication (standalone, default OFF).

Replicates radix-tree mutations across gateway replicas so prefix-aware routing
works horizontally: every replica's local tree converges on which backend holds
which prefix, while reads (longest-prefix match) stay local and fast.

Wire-up: construct a ClusterCoordinator(tree, bus, replica_id), call
publish_insert/publish_remove on the write path, and sync() from a background
loop. Disabled unless GW_CLUSTER_ENABLED=1.
"""

from .bus import InMemoryBroker, InMemoryBus, RedisBus, ReplicationBus, make_bus
from .config import ClusterConfig
from .coordinator import ClusterCoordinator
from .events import INSERT, LOAD, REMOVE_BACKEND, LoadEvent, PrefixEvent, decode_event
from .fleet import FleetLoadView

__all__ = [
    "ClusterConfig",
    "ClusterCoordinator",
    "FleetLoadView",
    "ReplicationBus",
    "InMemoryBus",
    "InMemoryBroker",
    "RedisBus",
    "make_bus",
    "PrefixEvent",
    "LoadEvent",
    "decode_event",
    "INSERT",
    "REMOVE_BACKEND",
    "LOAD",
]
