from __future__ import annotations

"""Request DAG and topological layering.

An agentic / multi-step LLM workflow is a DAG of requests: map-reduce
summarization (N parallel "summarize chunk" calls -> one "combine" call), a
plan -> fan-out -> aggregate loop, a tool-use chain, etc. Each node is one LLM
call carrying its own prompt; an edge u -> v means v cannot start until u
finishes (v's prompt typically embeds u's output and/or shared upstream
context).

`layers()` returns a topological *layering* (Kahn's algorithm): each layer is a
set of mutually independent nodes that may run concurrently, and all of a
node's parents live in strictly earlier layers. The scheduler walks layers in
order, using each layer as a dependency barrier.
"""

import dataclasses
from collections import deque


@dataclasses.dataclass(frozen=True)
class DagNode:
    id: str
    prompt: str
    parents: tuple[str, ...] = ()


class DagError(ValueError):
    pass


class RequestDag:
    def __init__(self, nodes: list[DagNode] | None = None):
        self.nodes: dict[str, DagNode] = {}
        for n in nodes or []:
            self.add(n)

    def add(self, node: DagNode) -> None:
        if node.id in self.nodes:
            raise DagError(f"duplicate node id: {node.id!r}")
        self.nodes[node.id] = node

    def validate(self) -> None:
        """Raise DagError if any parent is missing or the graph has a cycle."""
        for n in self.nodes.values():
            for p in n.parents:
                if p not in self.nodes:
                    raise DagError(f"node {n.id!r} references unknown parent {p!r}")
        # cycle check via the same Kahn pass used for layering
        self._kahn()

    def children(self) -> dict[str, list[str]]:
        kids: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for n in self.nodes.values():
            for p in n.parents:
                kids[p].append(n.id)
        return kids

    def layers(self) -> list[list[str]]:
        """Topological layers; ids within a layer are sorted for determinism."""
        return self._kahn()

    def _kahn(self) -> list[list[str]]:
        indeg = {nid: len(self.nodes[nid].parents) for nid in self.nodes}
        kids = self.children()
        frontier = deque(sorted(nid for nid, d in indeg.items() if d == 0))
        layers: list[list[str]] = []
        seen = 0
        while frontier:
            layer = sorted(frontier)
            frontier.clear()
            for nid in layer:
                seen += 1
                for c in kids[nid]:
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        frontier.append(c)
            layers.append(layer)
        if seen != len(self.nodes):
            raise DagError("DAG contains a cycle")
        return layers
