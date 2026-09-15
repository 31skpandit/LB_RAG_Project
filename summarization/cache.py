"""Content-hash-keyed cache for indexing-time summaries (text/table/image).

Without this, restarting the pipeline on unchanged documents -- during
development, or a repeated ingestion run -- re-summarizes every single chunk
and image from scratch, burning LLM tokens on identical content every time.
This caches each summary keyed by a hash of its content plus which
provider/model produced it, so:
- Re-running on the same documents is free (cache hit, no LLM call).
- Switching SUMMARY_PROVIDER (or its model) correctly misses the cache and
  re-summarizes with the new one, rather than silently reusing an older
  provider's summary.
- Genuinely new/changed content still gets summarized normally.

Backed by Redis (already a required service for this project's docstore --
see retrieval/doc_store.py), in a separate key prefix so the two don't collide.
"""
import hashlib
from typing import Optional

from langchain_community.utilities.redis import get_client

from config import settings

_CACHE_PREFIX = "summary_cache:"


def _client():
    return get_client(settings.REDIS_URL)


def _cache_key(content: str, provider: str, model: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{_CACHE_PREFIX}{provider}:{model}:{digest}"


def get_cached_summary(content: str, provider: str, model: str) -> Optional[str]:
    """Return the cached summary for this exact content + provider/model, or None on a miss."""
    value = _client().get(_cache_key(content, provider, model))
    return value.decode("utf-8") if value is not None else None


def set_cached_summary(content: str, provider: str, model: str, summary: str) -> None:
    """Cache `summary` for this exact content + provider/model."""
    _client().set(_cache_key(content, provider, model), summary)
