"""Cache-locality-aware DAG scheduler (standalone, not on the hot path).

The live router optimizes one request at a time. Real LLM workloads are often
DAGs of requests -- map-reduce summarization, plan/fan-out/aggregate, tool-use
chains -- whose nodes share large prefixes (a common system prompt, retrieved
context, a parent's continuation). Routing each node independently scatters that
shared work across backends and throws away the cache.

This package schedules the WHOLE DAG with foreknowledge:

  1. graph     -- model the workflow as a DAG and layer it topologically
     (Kahn), so each layer is a dependency barrier of independent nodes.
  2. scheduler -- greedy list-scheduling per layer that co-locates cache-sharing
     nodes, scoring affinity with the same RadixTree longest-prefix match the
     live router uses, traded off against load + a parent-affinity bonus.

Isolated from server.py like gateway/disagg and gateway/rag: the benchmarked
single-request hot path is untouched.
"""

from .config import DagSchedConfig
from .graph import DagError, DagNode, RequestDag
from .scheduler import NodePlacement, Schedule, schedule

__all__ = [
    "DagSchedConfig",
    "DagError",
    "DagNode",
    "RequestDag",
    "NodePlacement",
    "Schedule",
    "schedule",
]
