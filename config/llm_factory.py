"""Builds chat/embedding models for the configured LLM_PROVIDER (openai, gemini, ollama, or qwen).

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

    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.OLLAMA_CHAT_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
            num_ctx=settings.OLLAMA_NUM_CTX,
        )

    from langchain_openai import ChatOpenAI

    if settings.LLM_PROVIDER == "qwen":
        return ChatOpenAI(
            model=settings.QWEN_CHAT_MODEL,
            temperature=temperature,
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
        )

    return ChatOpenAI(model=settings.CHATGPT_MODEL, temperature=temperature)


def get_embedding_model() -> Embeddings:
    """Return an embedding model for the configured provider."""
    if settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=settings.GEMINI_EMBEDDING_MODEL)

    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.OLLAMA_EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)

    from langchain_openai import OpenAIEmbeddings

    if settings.LLM_PROVIDER == "qwen":
        return OpenAIEmbeddings(
            model=settings.QWEN_EMBEDDING_MODEL,
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
        )

    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
