#!/usr/bin/env python
"""Worked example of RAG-aware caching + chunk-affinity routing.

Two things to see here, both without a GPU:

  1. Canonicalization: two queries that retrieve the SAME documents in a
     DIFFERENT relevance order render as byte-identical prefixes, so a stock
     prefix cache reuses the chunk KVs. We print the chunk-id sequence for two
     reorderings and show they match.

  2. Chunk-affinity routing: once a backend has served a chunk set, a later
     query over an overlapping set routes back to it (the set-overlap analogue
     of longest-prefix match), traded off against load.

Usage:
    python scripts/rag_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gateway.load_tracker import LoadTracker
from gateway.rag import (
    ChunkAffinityIndex,
    RagConfig,
    canonicalize_chunks,
    choose_rag_backend,
    chunk_ids,
)


def _short(ids: list[int]) -> str:
    return "[" + ", ".join(f"{i & 0xFFFF:04x}" for i in ids) + "]"


def demo_canonicalization() -> None:
    docs = ["solar output peaks at noon", "battery storage smooths demand",
            "grid frequency must stay at 50Hz"]
    order_a = [docs[0], docs[1], docs[2]]
    order_b = [docs[2], docs[0], docs[1]]   # retriever returned a different order

    ids_a = chunk_ids(canonicalize_chunks(order_a))
    ids_b = chunk_ids(canonicalize_chunks(order_b))

    print("== canonicalization (order-independent prefix) ==")
    print(f"  retrieval order A -> canonical ids {_short(ids_a)}")
    print(f"  retrieval order B -> canonical ids {_short(ids_b)}")
    print(f"  identical prefix? {ids_a == ids_b}  "
          f"(stock prefix cache reuses chunk KVs either way)\n")


def demo_affinity_routing() -> None:
    cfg = RagConfig()
    idx = ChunkAffinityIndex()
    load = LoadTracker()

    # b0 has previously served a 4-doc set; b1 has served nothing.
    served = canonicalize_chunks(["doc-w", "doc-x", "doc-y", "doc-z"])
    idx.record("b0", chunk_ids(served))

    print("== chunk-affinity routing ==")
    print(f"  b0 has cached {len(served)} chunks; b1 is cold\n")
    print(f"  {'query overlaps b0':>20} {'b0 load':>8} {'chosen':>7} "
          f"{'overlap':>8} {'est_ms':>8}")
    print("  " + "-" * 56)

    # A query that reuses 3 of b0's 4 chunks plus one new one.
    query = canonicalize_chunks(["doc-w", "doc-x", "doc-y", "doc-NEW"])
    qids = chunk_ids(query)

    for b0_inflight in (0, 4, cfg.max_inflight + 1):
        load.inflight["b0"] = b0_inflight
        load.inflight["b1"] = 0
        res = choose_rag_backend(chunk_ids=qids, backends=["b0", "b1"],
                                 index=idx, load=load, cfg=cfg)
        note = "(b0 saturated)" if b0_inflight > cfg.max_inflight else ""
        print(f"  {'3 of 4':>20} {b0_inflight:>8} {res.backend_id:>7} "
              f"{res.overlap:>8} {res.est_ms:>8.1f}  {note}")
    print("\n  -> warm backend wins on cache affinity until it saturates, "
          "then we shed to the cold one.")


def main() -> None:
    demo_canonicalization()
    demo_affinity_routing()


if __name__ == "__main__":
    main()
