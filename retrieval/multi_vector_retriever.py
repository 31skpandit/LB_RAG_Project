"""Multi-vector retriever: indexes summaries but returns raw text/table/image content."""
import uuid
from typing import List

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document

from config import settings


def _add_documents(retriever: MultiVectorRetriever, doc_summaries: List[str], doc_contents: List, id_key: str):
    if not doc_summaries:
        return
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(doc_summaries)]
    retriever.vectorstore.add_documents(summary_docs)
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
