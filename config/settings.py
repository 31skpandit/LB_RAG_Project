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
    # One of: openai, gemini, ollama, qwen, groq, deepseek
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    PROVIDERS = ("openai", "gemini", "ollama", "qwen", "groq", "deepseek")

    # Providers with no embeddings API at all: Groq (confirmed live via its
    # /models list, despite some third-party docs/blogs claiming otherwise)
    # and DeepSeek (its hosted API only ever exposed chat completions, no
    # embeddings endpoint). EMBEDDING_PROVIDER selects the embedding-model
    # provider independently of LLM_PROVIDER, defaulting to match it so
    # nothing changes for single-provider setups (openai/gemini/ollama/qwen
    # all support both). Override it when LLM_PROVIDER=groq or deepseek --
    # e.g. EMBEDDING_PROVIDER=gemini -- to mix a chat-only provider with
    # another provider's embeddings.
    NO_EMBEDDINGS_PROVIDERS = ("groq", "deepseek")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", LLM_PROVIDER)
    # NOTE: can't reference NO_EMBEDDINGS_PROVIDERS by name inside this
    # comprehension -- comprehensions in a class body run in their own nested
    # scope that can only see the outermost iterable (PROVIDERS here), not
    # other class-body names, so the tuple is inlined literally instead.
    EMBEDDING_PROVIDERS = tuple(p for p in PROVIDERS if p not in ("groq", "deepseek"))

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
    # To run DeepSeek locally instead: this needs NO code changes, just
    # `ollama pull deepseek-r1:1.5b` (fits ~4GB VRAM; bump to :7b/:8b/:14b on
    # stronger hardware) and set OLLAMA_CHAT_MODEL=deepseek-r1:1.5b below --
    # BUT deepseek-r1 is text-only (confirmed: no vision variant exists on
    # Ollama), so it will fail if it ever needs to summarize an actual image.
    # Safe with UNSTRUCTURED_STRATEGY=fast (extracts no images anyway); keep
    # qwen3-vl or llava as OLLAMA_CHAT_MODEL if you switch to hi_res and need
    # working image summaries.
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

    # Groq: extremely fast free-tier inference via its OpenAI-compatible endpoint.
    # Get a free key at https://console.groq.com/keys (no country restriction).
    # GROQ_CHAT_MODEL must be vision-capable since it's also used for image
    # summaries -- Groq's vision models are currently Qwen-based and marked
    # "preview" (can be deprecated/rotated without much notice); check
    # https://console.groq.com/docs/vision if the default below stops working.
    # Groq has no embeddings API for this account -- confirmed live, not just
    # from docs (nomic-embed-text-v1_5 404s, and no embedding model appears in
    # the account's /v1/models list at all). So there's no GROQ_EMBEDDING_MODEL
    # here; see EMBEDDING_PROVIDER above for how embeddings are handled when
    # LLM_PROVIDER=groq.
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_CHAT_MODEL: str = os.getenv("GROQ_CHAT_MODEL", "qwen/qwen3.6-27b")
    # Groq's free tier enforces output-tokens-per-minute (OTPM), separate from
    # (and tighter than) the general tokens-per-minute limit -- 1000 OTPM for
    # qwen/qwen3.6-27b as of writing. Without an explicit cap, the client lets
    # the model request as many output tokens as its max context allows, which
    # trips a 429 immediately. 500 leaves headroom under 1000 for a 2nd request
    # in the same minute; lower it further if you still see 429s.
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "500"))

    # DeepSeek's official API, OpenAI-compatible. Get a key at
    # https://platform.deepseek.com/api_keys (pay-as-you-go, no free tier, but
    # rate limiting is dynamic/concurrency-based rather than a harsh fixed RPM
    # like Groq/Gemini's free tiers -- shouldn't need GROQ_MAX_TOKENS-style
    # workarounds). DEEPSEEK_CHAT_MODEL must be vision-capable since it's also
    # used for image summaries -- "deepseek-flash" supports image input;
    # "deepseek-v4-pro" (higher quality, no vision, pricier) is the
    # alternative if you don't need image summarization.
    # No embeddings endpoint (same gap as Groq) -- see EMBEDDING_PROVIDER above.
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_CHAT_MODEL: str = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-flash")

    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # Data sources
    # When no --pdf-path/--pdf-url is given, main.py auto-discovers and
    # processes every supported document (PDF/DOCX/PPTX/TXT/HTML/image)
    # directly inside DATA_DIR (see data_ingestion.discover_documents) --
    # the pipeline only ever runs on documents actually placed there; an
    # empty DATA_DIR is a hard error, not a silent download. PDF_URL/PDF_PATH
    # are only used for the explicit --pdf-url override (data_ingestion.download_pdf).
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    PDF_URL: str = os.getenv("PDF_URL", "https://sgp.fas.org/crs/misc/IF10244.pdf")
    PDF_PATH: str = os.getenv("PDF_PATH", "./data/IF10244.pdf")
    FIGURES_DIR: str = os.getenv("FIGURES_DIR", "./data/figures")

    # Unstructured partitioning
    # "hi_res" runs a deep-learning layout/OCR model over a rendered image of
    # every page -- accurate for scanned/complex-layout PDFs, but commonly
    # 1-3+ seconds PER PAGE on CPU (no GPU here), so it doesn't scale to
    # real, many-page, digitally-native documents (confirmed: 126 pages was
    # the actual bottleneck, not a bug). "fast" reads the PDF's own text layer
    # directly, skipping the layout model entirely -- typically 10-50x faster
    # for documents that aren't scanned images. Table-structure quality can be
    # lower with "fast" (processing/table_converter.py already falls back
    # gracefully if text_as_html is missing). Switch back to "hi_res" if you
    # need scanned-document support or notice table extraction quality drop.
    UNSTRUCTURED_STRATEGIES = ("fast", "hi_res", "ocr_only", "auto")
    UNSTRUCTURED_STRATEGY: str = os.getenv("UNSTRUCTURED_STRATEGY", "fast")

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
    def ensure_groq_key(cls) -> str:
        """Return the Groq key, prompting interactively if not set via env/.env."""
        if not cls.GROQ_API_KEY:
            from getpass import getpass

            cls.GROQ_API_KEY = getpass("Enter Groq API Key: ")
            os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        else:
            os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        return cls.GROQ_API_KEY

    @classmethod
    def ensure_deepseek_key(cls) -> str:
        """Return the DeepSeek key, prompting interactively if not set via env/.env."""
        if not cls.DEEPSEEK_API_KEY:
            from getpass import getpass

            cls.DEEPSEEK_API_KEY = getpass("Enter DeepSeek API Key: ")
            os.environ["DEEPSEEK_API_KEY"] = cls.DEEPSEEK_API_KEY
        else:
            os.environ["DEEPSEEK_API_KEY"] = cls.DEEPSEEK_API_KEY
        return cls.DEEPSEEK_API_KEY

    @classmethod
    def ensure_llm_key(cls) -> str:
        """Ensure the API key for the configured LLM_PROVIDER is set, prompting if needed.

        Ollama runs locally and needs no API key, so it's a no-op.
        """
        if cls.LLM_PROVIDER == "gemini":
            return cls.ensure_google_key()
        if cls.LLM_PROVIDER == "qwen":
            return cls.ensure_qwen_key()
        if cls.LLM_PROVIDER == "groq":
            return cls.ensure_groq_key()
        if cls.LLM_PROVIDER == "deepseek":
            return cls.ensure_deepseek_key()
        if cls.LLM_PROVIDER == "ollama":
            return ""
        return cls.ensure_openai_key()


settings = Settings()

if settings.LLM_PROVIDER not in settings.PROVIDERS:
    raise ValueError(f"Invalid LLM_PROVIDER {settings.LLM_PROVIDER!r}. Expected one of {settings.PROVIDERS}.")

if settings.EMBEDDING_PROVIDER not in settings.EMBEDDING_PROVIDERS:
    raise ValueError(
        f"Invalid EMBEDDING_PROVIDER {settings.EMBEDDING_PROVIDER!r}. "
        f"{settings.NO_EMBEDDINGS_PROVIDERS} have no embeddings API, so they can't be used here -- "
        f"expected one of {settings.EMBEDDING_PROVIDERS}. "
        f"(If LLM_PROVIDER is one of those, set EMBEDDING_PROVIDER explicitly, e.g. to 'gemini'.)"
    )

if settings.CHUNKING_STRATEGY not in settings.CHUNKING_STRATEGIES:
    raise ValueError(
        f"Invalid CHUNKING_STRATEGY {settings.CHUNKING_STRATEGY!r}. "
        f"Expected one of {settings.CHUNKING_STRATEGIES}."
    )

if settings.UNSTRUCTURED_STRATEGY not in settings.UNSTRUCTURED_STRATEGIES:
    raise ValueError(
        f"Invalid UNSTRUCTURED_STRATEGY {settings.UNSTRUCTURED_STRATEGY!r}. "
        f"Expected one of {settings.UNSTRUCTURED_STRATEGIES}."
    )


