from __future__ import annotations

"""RAG prompt structuring.

A RAG prompt is [system] + [retrieved chunks] + [user query]. The chunks are
the bulk of the tokens and are shared across many different queries (the same
documents get retrieved again), but the query is short and unique. Stock vLLM
prefix-caches byte-identical prefixes, so two queries that retrieve the same
documents only share cache if those documents render as the SAME prefix.

Retrievers return chunks in a query-dependent relevance order, which breaks
that sharing. This module:
  * hashes each chunk to a stable, process-independent id,
  * canonicalizes the chunk list (dedupe + sort by id) so any two requests
    over the same chunk SET produce the same prefix,
  * assembles messages with the (stable) chunks in the system prefix and the
    (variable) query as the tail, so only the short query is ever uncached.
"""

import dataclasses

from ..hashing import stable_seed


def chunk_id(text: str) -> int:
    """Stable, process-independent id for a retrieved chunk."""
    return stable_seed(text)


@dataclasses.dataclass
class RagPrompt:
    system: str
    chunks: list[str]
    query: str


def parse_rag_request(body: dict, request_field: str = "rag") -> RagPrompt | None:
    """Extract a RagPrompt from a request body, or None if it isn't a RAG
    request (so the caller falls back to normal routing)."""
    payload = body.get(request_field)
    if not isinstance(payload, dict):
        return None
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return None
    clean = [c for c in chunks if isinstance(c, str) and c]
    if not clean:
        return None
    return RagPrompt(
        system=str(payload.get("system", "")),
        chunks=clean,
        query=str(payload.get("query", "")),
    )


def canonicalize_chunks(chunks: list[str]) -> list[str]:
    """Dedupe and stable-sort chunks by id so identical chunk SETS yield an
    identical prefix regardless of retrieval order."""
    seen: dict[int, str] = {}
    for c in chunks:
        seen.setdefault(chunk_id(c), c)   # first occurrence wins on collision
    return [seen[k] for k in sorted(seen.keys())]


def chunk_ids(chunks: list[str]) -> list[int]:
    return [chunk_id(c) for c in chunks]


def build_messages(rag: RagPrompt, ordered_chunks: list[str]) -> list[dict]:
    """Assemble OpenAI-style messages: chunks live in the system prefix
    (cacheable), the query is the user tail (uncached)."""
    context = "\n\n".join(ordered_chunks)
    system = f"{rag.system}\n\n{context}" if rag.system else context
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": rag.query},
    ]
