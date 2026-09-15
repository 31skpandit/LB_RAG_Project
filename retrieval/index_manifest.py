"""Tracks which vectorstore/docstore chunk IDs belong to which source file.

This is what makes the pipeline actually skip re-processing unchanged files,
and keeps deleted/changed files from leaving orphaned data behind:

1. Unchanged files (same content, same chunking-relevant settings) skip the
   entire partition/summarize/embed pipeline on the next run -- not for API
   cost (that's already handled by the per-chunk caches in
   summarization/cache.py and retrieval/multi_vector_retriever.py) but for
   wall-clock time: re-parsing a document via Unstructured on every restart
   is pure waste when nothing changed.
2. A file whose *content* changed gets its old, now-stale chunk IDs deleted
   before the new ones are written, instead of leaving orphaned vectors/raw
   content behind from the previous version.
3. A file that's deleted from disk entirely gets pruned the same way, the
   next time the pipeline runs in auto-discovery mode.

Backed by Redis, same pattern as the other caches in this project.
"""
import hashlib
import json
from typing import List

from langchain_community.utilities.redis import get_client

from config import settings

_MANIFEST_PREFIX = "indexed_manifest:"


def _client():
    return get_client(settings.REDIS_URL)


def _manifest_key(file_path: str) -> str:
    return _MANIFEST_PREFIX + hashlib.sha256(file_path.encode("utf-8")).hexdigest()


def _content_fingerprint(file_path: str) -> str:
    """Hash the file's bytes plus every setting that affects how it gets
    partitioned/chunked, so changing UNSTRUCTURED_STRATEGY/CHUNKING_STRATEGY/
    MAX_CHARACTERS/NEW_AFTER_N_CHARS/COMBINE_TEXT_UNDER_N_CHARS correctly
    forces re-processing instead of silently keeping stale chunks built
    under different settings.
    """
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    config_fingerprint = "|".join(
        [
            settings.UNSTRUCTURED_STRATEGY,
            settings.CHUNKING_STRATEGY,
            str(settings.MAX_CHARACTERS),
            str(settings.NEW_AFTER_N_CHARS),
            str(settings.COMBINE_TEXT_UNDER_N_CHARS),
        ]
    )
    return hashlib.sha256(f"{file_hash}:{config_fingerprint}".encode("utf-8")).hexdigest()


def _load(file_path: str):
    raw = _client().get(_manifest_key(file_path))
    return json.loads(raw) if raw is not None else None


def is_already_indexed(file_path: str) -> bool:
    """True if this exact file content, under the current chunking settings,
    was already fully partitioned/summarized/embedded in a prior run."""
    manifest = _load(file_path)
    return manifest is not None and manifest.get("fingerprint") == _content_fingerprint(file_path)


def record_indexed(file_path: str, chunk_ids: List[str]) -> None:
    """Record that `file_path`, at its current content+settings fingerprint,
    now maps to exactly `chunk_ids`. Call only after summarization/embedding
    for this file's chunks has actually succeeded.
    """
    manifest = {
        "file_path": file_path,
        "fingerprint": _content_fingerprint(file_path),
        "chunk_ids": chunk_ids,
    }
    _client().set(_manifest_key(file_path), json.dumps(manifest))


def forget_file(file_path: str) -> List[str]:
    """Remove `file_path`'s manifest entry entirely, returning the chunk IDs
    it had (empty if it was never indexed) -- the caller is responsible for
    actually deleting those IDs from the vectorstore/docstore.
    """
    key = _manifest_key(file_path)
    raw = _client().get(key)
    _client().delete(key)
    return json.loads(raw).get("chunk_ids", []) if raw is not None else []


def all_manifest_file_paths() -> List[str]:
    """Every file path currently tracked in the manifest -- used to detect
    deletions: anything here no longer present on disk needs pruning."""
    client = _client()
    paths = []
    for key in client.scan_iter(match=f"{_MANIFEST_PREFIX}*"):
        raw = client.get(key)
        if raw:
            path = json.loads(raw).get("file_path")
            if path:
                paths.append(path)
    return paths
