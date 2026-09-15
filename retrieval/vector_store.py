"""Chroma vector store setup — stores summary embeddings for retrieval."""
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from config import settings
from config.llm_factory import get_embedding_model as _get_embedding_model


def get_embedding_model() -> Embeddings:
    return _get_embedding_model()


def get_vector_store(collection_name: str = None, embedding_function=None) -> Chroma:
    return Chroma(
        collection_name=collection_name or settings.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function or get_embedding_model(),
        collection_metadata={"hnsw:space": "cosine"},
        # Without this, Chroma is in-memory-only and the whole vector index
        # is lost on every process exit -- see CHROMA_PERSIST_DIR in
        # config/settings.py for why this matters.
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
