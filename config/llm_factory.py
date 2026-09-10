"""Builds chat/embedding models for the configured LLM_PROVIDER (openai or gemini).

Every module that needs an LLM or embedding model should go through these
factories instead of instantiating a provider client directly, so switching
`LLM_PROVIDER` in `.env` swaps the whole pipeline without touching call sites.
"""
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from config import settings


def get_chat_model(temperature: float = None) -> BaseChatModel:
    """Return a chat model for the configured provider (supports text + vision)."""
    temperature = settings.LLM_TEMPERATURE if temperature is None else temperature

    if settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=settings.GEMINI_CHAT_MODEL, temperature=temperature)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.CHATGPT_MODEL, temperature=temperature)


def get_embedding_model() -> Embeddings:
    """Return an embedding model for the configured provider."""
    if settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=settings.GEMINI_EMBEDDING_MODEL)

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
