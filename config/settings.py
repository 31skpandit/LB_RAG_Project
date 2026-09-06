"""Central configuration for the Multimodal RAG pipeline.

Every other module reads its settings from here so that experiments (swapping
models, paths, or backends) only require editing one place or setting env vars.
"""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM / embeddings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHATGPT_MODEL: str = os.getenv("CHATGPT_MODEL", "gpt-4o")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # Data sources
    PDF_URL: str = os.getenv("PDF_URL", "https://sgp.fas.org/crs/misc/IF10244.pdf")
    PDF_PATH: str = os.getenv("PDF_PATH", "./data/IF10244.pdf")
    FIGURES_DIR: str = os.getenv("FIGURES_DIR", "./data/figures")

    # Unstructured partitioning
    CHUNKING_STRATEGY: str = "by_title"
    MAX_CHARACTERS: int = 4000
    NEW_AFTER_N_CHARS: int = 4000
    COMBINE_TEXT_UNDER_N_CHARS: int = 2000

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


settings = Settings()
