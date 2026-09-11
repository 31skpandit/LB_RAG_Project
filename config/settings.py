"""Central configuration for the Multimodal RAG pipeline.

Every other module reads its settings from here so that experiments (swapping
models, paths, or backends) only require editing one place or setting env vars.
"""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM / embeddings
    # Toggle provider to test with the free-tier Gemini API instead of OpenAI.
    # One of: openai, gemini
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHATGPT_MODEL: str = os.getenv("CHATGPT_MODEL", "gpt-4o")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # Data sources
    PDF_URL: str = os.getenv("PDF_URL", "https://sgp.fas.org/crs/misc/IF10244.pdf")
    PDF_PATH: str = os.getenv("PDF_PATH", "./data/IF10244.pdf")
    FIGURES_DIR: str = os.getenv("FIGURES_DIR", "./data/figures")

    # Unstructured partitioning
    # Toggle to compare chunking behavior: "by_title"/"basic" run natively inside
    # Unstructured; "recursive"/"sentence"/"paragraph" are applied afterwards
    # by processing/chunking.py. See CHUNKING_STRATEGIES for all valid values.
    CHUNKING_STRATEGIES = ("by_title", "basic", "recursive", "sentence", "paragraph")
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "by_title")
    MAX_CHARACTERS: int = int(os.getenv("MAX_CHARACTERS", "4000"))
    NEW_AFTER_N_CHARS: int = int(os.getenv("NEW_AFTER_N_CHARS", "4000"))
    COMBINE_TEXT_UNDER_N_CHARS: int = int(os.getenv("COMBINE_TEXT_UNDER_N_CHARS", "2000"))

    # Retrieval backends
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "mm_rag")
    RETRIEVER_ID_KEY: str = "doc_id"

    @classmethod
    def ensure_openai_key(cls) -> str:
        """Return the OpenAI key, prompting interactively if not set via env/.env."""
        if not cls.OPENAI_API_KEY:
            from getpass import getpass

            cls.OPENAI_API_KEY = getpass("Enter OpenAI API Key: ")
            os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        else:
            os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return cls.OPENAI_API_KEY

    @classmethod
    def ensure_google_key(cls) -> str:
        """Return the Google (Gemini) key, prompting interactively if not set via env/.env."""
        if not cls.GOOGLE_API_KEY:
            from getpass import getpass

            cls.GOOGLE_API_KEY = getpass("Enter Google (Gemini) API Key: ")
            os.environ["GOOGLE_API_KEY"] = cls.GOOGLE_API_KEY
        else:
            os.environ["GOOGLE_API_KEY"] = cls.GOOGLE_API_KEY
        return cls.GOOGLE_API_KEY

    @classmethod
    def ensure_llm_key(cls) -> str:
        """Ensure the API key for the configured LLM_PROVIDER is set, prompting if needed."""
        if cls.LLM_PROVIDER == "gemini":
            return cls.ensure_google_key()
        return cls.ensure_openai_key()


settings = Settings()

if settings.LLM_PROVIDER not in ("openai", "gemini"):
    raise ValueError(f"Invalid LLM_PROVIDER {settings.LLM_PROVIDER!r}. Expected 'openai' or 'gemini'.")

if settings.CHUNKING_STRATEGY not in settings.CHUNKING_STRATEGIES:
    raise ValueError(
        f"Invalid CHUNKING_STRATEGY {settings.CHUNKING_STRATEGY!r}. "
        f"Expected one of {settings.CHUNKING_STRATEGIES}."
    )


