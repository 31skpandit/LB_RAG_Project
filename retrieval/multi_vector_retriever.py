"""Multi-vector retriever: indexes summaries but returns raw text/table/image content."""
import hashlib
from typing import List

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document

from config import settings


def content_id(content: str) -> str:
    """Deterministic ID derived from raw content (not a random UUID).

    This is what actually fixes "why does it re-embed on every restart":
    with a persisted Chroma store (see config.settings.CHROMA_PERSIST_DIR)
    and IDs derived from content instead of random per-run UUIDs, the same
    unchanged chunk always maps to the same vectorstore ID across restarts,
    so _add_documents below can check "is this ID already embedded" and skip
    the (billed) embedding call entirely for content that hasn't changed.

    Public (not module-private) because main.py needs to compute the exact
    same IDs to record which chunks belong to which source file -- see
    retrieval/index_manifest.py -- so pruning a deleted/changed file deletes
    precisely the right vectorstore/docstore entries.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _add_documents(retriever: MultiVectorRetriever, doc_summaries: List[str], doc_contents: List, id_key: str):
    if not doc_summaries:
        return

    doc_ids = [content_id(content) for content in doc_contents]

    # Skip re-embedding (and re-paying for) content whose vector is already
    # in the persisted store -- this is the actual embedding-cost fix.
    already_indexed = set(retriever.vectorstore.get(ids=doc_ids)["ids"])
    new_indices = [i for i, doc_id in enumerate(doc_ids) if doc_id not in already_indexed]

    if new_indices:
        new_summary_docs = [
            Document(page_content=doc_summaries[i], metadata={id_key: doc_ids[i]}) for i in new_indices
        ]
        retriever.vectorstore.add_documents(new_summary_docs, ids=[doc_ids[i] for i in new_indices])

    skipped = len(doc_ids) - len(new_indices)
    if skipped:
        print(f"  {skipped} embeddings reused (already indexed), {len(new_indices)} newly embedded")

    # Docstore writes are local, not billed -- always refresh so raw content
    # stays in sync even on a run that only newly-embeds some entries.
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def create_multi_vector_retriever(
    docstore,
    vectorstore,
    text_summaries: List[str],
    texts: List,
    table_summaries: List[str],
    tables: List,
    image_summaries: List[str],
    images: List,
    id_key: str = None,
) -> MultiVectorRetriever:
    """Create a retriever that indexes summaries but returns raw text/table/image content."""
    id_key = id_key or settings.RETRIEVER_ID_KEY

    # RETRIEVAL_K controls how many documents get pulled into context per
    # question -- directly controls the input-token cost of every final-answer
    # call (the cost that scales with ongoing usage, not just indexing).
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        search_kwargs={"k": settings.RETRIEVAL_K},
    )

    _add_documents(retriever, text_summaries, texts, id_key)
    _add_documents(retriever, table_summaries, tables, id_key)
    _add_documents(retriever, image_summaries, images, id_key)

    return retriever
