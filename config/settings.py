"""Central configuration for the Multimodal RAG pipeline.

Every other module reads its settings from here so that experiments (swapping
models, paths, or backends) only require editing one place or setting env vars.
"""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM / embeddings
    # Toggle provider: swap the whole pipeline (chat + vision + embeddings) by
    # changing this one value. No code changes needed elsewhere.
    # One of: openai, gemini, ollama, qwen
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    PROVIDERS = ("openai", "gemini", "ollama", "qwen")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHATGPT_MODEL: str = os.getenv("CHATGPT_MODEL", "gpt-4o")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

    # Ollama runs models locally, so no API key is needed. OLLAMA_CHAT_MODEL must
    # be a vision-capable model (e.g. "qwen3-vl:2b", "llava") since it's also
    # used to summarize images. Default picks the Qwen3 family: qwen3-vl:2b for
    # chat+vision (small enough to fit in ~4GB of VRAM on modest/laptop GPUs --
    # bump to qwen3-vl:4b/8b on stronger hardware for better summary quality),
    # qwen3-embedding:0.6b for embeddings (`ollama pull` both first).
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3-vl:2b")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    # Ollama's own default context window (4096) -- kept configurable in case a
    # larger corpus/retrieval-k needs more, but NOTE: raising this to 8192 was
    # tried as a fix for an empty-answer bug on this hardware and did NOT fix
    # it, while roughly tripling generation time (bigger KV-cache -> less of
    # the model fits in VRAM -> more CPU fallback). The empty-answer issue has
    # a different root cause, still under investigation -- don't raise this
    # again without solid evidence it's actually the fix.
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

    # Qwen (Alibaba) via DashScope's OpenAI-compatible endpoint, has a free tier.
    # Get a key at https://bailian.console.alibabacloud.com/ (or the intl. console).
    # QWEN_CHAT_MODEL must be a vision-capable model (e.g. "qwen-vl-plus") since
    # it's also used to summarize images.
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    QWEN_CHAT_MODEL: str = os.getenv("QWEN_CHAT_MODEL", "qwen-vl-plus")
    QWEN_EMBEDDING_MODEL: str = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

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
    def ensure_qwen_key(cls) -> str:
        """Return the Qwen (DashScope) key, prompting interactively if not set via env/.env."""
        if not cls.QWEN_API_KEY:
            from getpass import getpass

            cls.QWEN_API_KEY = getpass("Enter Qwen (DashScope) API Key: ")
            os.environ["QWEN_API_KEY"] = cls.QWEN_API_KEY
        else:
            os.environ["QWEN_API_KEY"] = cls.QWEN_API_KEY
        return cls.QWEN_API_KEY

    @classmethod
    def ensure_llm_key(cls) -> str:
        """Ensure the API key for the configured LLM_PROVIDER is set, prompting if needed.

        Ollama runs locally and needs no API key, so it's a no-op.
        """
        if cls.LLM_PROVIDER == "gemini":
            return cls.ensure_google_key()
        if cls.LLM_PROVIDER == "qwen":
            return cls.ensure_qwen_key()
        if cls.LLM_PROVIDER == "ollama":
            return ""
        return cls.ensure_openai_key()


settings = Settings()

if settings.LLM_PROVIDER not in settings.PROVIDERS:
    raise ValueError(f"Invalid LLM_PROVIDER {settings.LLM_PROVIDER!r}. Expected one of {settings.PROVIDERS}.")

if settings.CHUNKING_STRATEGY not in settings.CHUNKING_STRATEGIES:
    raise ValueError(
        f"Invalid CHUNKING_STRATEGY {settings.CHUNKING_STRATEGY!r}. "
        f"Expected one of {settings.CHUNKING_STRATEGIES}."
    )


