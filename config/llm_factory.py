"""Builds chat/embedding models for the configured providers.

get_chat_model() uses LLM_PROVIDER (openai, gemini, ollama, qwen, groq, or
deepseek) -- used ONLY for final answer synthesis (rag_chain/pipeline.py),
where quality matters most.

get_summary_model() uses SUMMARY_PROVIDER (openai or gemini only) -- used for
indexing-time summarization (text/table/image, summarization/*.py), a
mechanical task that doesn't need the same model as the final answer. Kept
separate and restricted to two cheap, vision-capable providers so a
cost-conscious default here doesn't force the same tradeoff onto what the
user actually reads.

get_embedding_model() uses EMBEDDING_PROVIDER (same list as LLM_PROVIDER minus
groq/deepseek, neither of which has an embeddings API) -- it defaults to
LLM_PROVIDER, so setting only LLM_PROVIDER is enough unless you're using groq
or deepseek, which need EMBEDDING_PROVIDER set to something else (e.g. gemini)
for embeddings.

Every module that needs a model should go through these factories instead of
instantiating a provider client directly, so switching providers in `.env`
swaps the whole pipeline without touching call sites.
"""
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from config import settings


def get_chat_model(temperature: float = None) -> BaseChatModel:
    """Return a chat model for the configured provider (supports text + vision).

    Used only for final-answer synthesis -- see module docstring.
    LLM_MAX_TOKENS caps output length on every provider, but each provider's
    LangChain class uses a different field name for this (confirmed by
    reading each package's source): ChatOpenAI (openai/qwen/groq/deepseek)
    uses `max_tokens`, ChatGoogleGenerativeAI (gemini) uses
    `max_output_tokens`, ChatOllama uses `num_predict`.
    """
    temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
    max_tokens = settings.LLM_MAX_TOKENS

    if settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_CHAT_MODEL, temperature=temperature, max_output_tokens=max_tokens
        )

    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.OLLAMA_CHAT_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
            num_ctx=settings.OLLAMA_NUM_CTX,
            num_predict=max_tokens,
        )

    from langchain_openai import ChatOpenAI

    if settings.LLM_PROVIDER == "qwen":
        return ChatOpenAI(
            model=settings.QWEN_CHAT_MODEL,
            temperature=temperature,
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
            max_tokens=max_tokens,
        )

    if settings.LLM_PROVIDER == "groq":
        return ChatOpenAI(
            model=settings.GROQ_CHAT_MODEL,
            temperature=temperature,
            api_key=settings.GROQ_API_KEY,
            base_url=settings.GROQ_BASE_URL,
            max_tokens=max_tokens,
        )

    if settings.LLM_PROVIDER == "deepseek":
        return ChatOpenAI(
            model=settings.DEEPSEEK_CHAT_MODEL,
            temperature=temperature,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            max_tokens=max_tokens,
        )

    return ChatOpenAI(model=settings.CHATGPT_MODEL, temperature=temperature, max_tokens=max_tokens)


def get_summary_model(temperature: float = None) -> BaseChatModel:
    """Return the cheap, vision-capable model used for indexing-time summarization.

    Keyed on SUMMARY_PROVIDER (openai or gemini only), independent of
    LLM_PROVIDER -- see module docstring for why. Also capped by
    LLM_MAX_TOKENS, same as get_chat_model().
    """
    temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
    max_tokens = settings.LLM_MAX_TOKENS

    if settings.SUMMARY_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.SUMMARY_GEMINI_MODEL, temperature=temperature, max_output_tokens=max_tokens
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.SUMMARY_OPENAI_MODEL, temperature=temperature, max_tokens=max_tokens)


def get_embedding_model() -> Embeddings:
    """Return an embedding model for the configured provider.

    Keyed on EMBEDDING_PROVIDER, not LLM_PROVIDER -- they're independent so a
    chat-only provider like Groq (no embeddings API) can still be used for
    chat while another provider handles embeddings. EMBEDDING_PROVIDER
    defaults to LLM_PROVIDER, so this is a no-op for single-provider setups.
    """
    if settings.EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=settings.GEMINI_EMBEDDING_MODEL)

    if settings.EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.OLLAMA_EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)

    from langchain_openai import OpenAIEmbeddings

    if settings.EMBEDDING_PROVIDER == "qwen":
        return OpenAIEmbeddings(
            model=settings.QWEN_EMBEDDING_MODEL,
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
        )

    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
