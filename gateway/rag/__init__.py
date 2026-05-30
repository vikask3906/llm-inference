"""RAG-aware caching and routing (standalone, not imported by the hot path).

Prefix caching only catches byte-identical prefixes; in RAG the shared content
is the set of retrieved document chunks, which a retriever returns in a
query-dependent order. This package makes that sharing cacheable:

  1. structure  -- hash chunks to stable ids, canonicalize (dedupe + sort) so
     identical chunk SETS render as identical PREFIXES, and put the variable
     query in the tail.
  2. index      -- track which chunks each backend has cached (LRU set).
  3. router     -- route to the backend with the best chunk-set overlap,
     traded off against load.

Like gateway/disagg, this is isolated from server.py so the benchmarked
single-pool hot path is untouched.
"""

from .config import RagConfig
from .index import ChunkAffinityIndex
from .router import RagRouteResult, choose_rag_backend
from .structure import (
    RagPrompt,
    build_messages,
    canonicalize_chunks,
    chunk_id,
    chunk_ids,
    parse_rag_request,
)

__all__ = [
    "RagConfig",
    "ChunkAffinityIndex",
    "RagRouteResult",
    "choose_rag_backend",
    "RagPrompt",
    "build_messages",
    "canonicalize_chunks",
    "chunk_id",
    "chunk_ids",
    "parse_rag_request",
]
