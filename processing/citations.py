"""Attach source citation metadata (filename + page) to raw content before it's
stored in the docstore, and recover it at query time.

Citations are assembled deterministically from this metadata rather than left
to the LLM to remember to include -- the free-tier models exercised in this
project have already shown they can't be relied on for that consistently.
"""
import json
from typing import Optional, Tuple

from langchain_core.documents import Document


def pack_content(doc: Document) -> str:
    """JSON-encode a Document's content + source metadata for docstore storage."""
    source = doc.metadata.get("filename") or doc.metadata.get("source") or "unknown"
    page = doc.metadata.get("page_number")
    return json.dumps({"content": doc.page_content, "source": source, "page": page})


def unpack_content(raw: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Decode a packed content string back into (content, source, page).

    Falls back to treating `raw` as plain, un-packed content -- e.g. images,
    which are stored as bare base64 strings and were never packed -- when it
    isn't valid JSON in the expected shape.
    """
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "content" in data:
            return data["content"], data.get("source"), data.get("page")
    except (json.JSONDecodeError, TypeError):
        pass
    return raw, None, None


def format_citation(source: str, page: Optional[int]) -> str:
    """Render a citation tag, degrading gracefully when no page number exists.

    Page numbers are reliable for PDF/PPTX, unreliable for DOCX (only set when
    the source file has explicit hard page-breaks), and never set for TXT/HTML.
    """
    if page is not None:
        return f"{source}, p.{page}"
    return source
