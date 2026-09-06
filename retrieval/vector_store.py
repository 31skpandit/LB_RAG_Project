"""Chroma vector store setup — stores summary embeddings for retrieval."""
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import settings


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)


def get_vector_store(collection_name: str = None, embedding_function=None) -> Chroma:
    return Chroma(
        collection_name=collection_name or settings.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function or get_embedding_model(),
        collection_metadata={"hnsw:space": "cosine"},
    )
