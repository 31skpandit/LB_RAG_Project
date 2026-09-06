"""Element categorization helpers.

Two responsibilities:
1. Split raw Unstructured elements into text-chunk docs vs table docs.
2. Split retriever *results* (raw base64 images + raw text/tables) into buckets
   so the RAG prompt can treat images and text separately.
"""
import base64
import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document


def split_elements_by_category(data: List[Document]) -> Tuple[List[Document], List[Document]]:
    """Separate loaded elements into (text docs, table docs)."""
    docs, tables = [], []
    for doc in data:
        category = doc.metadata.get("category")
        if category == "Table":
            tables.append(doc)
        elif category == "CompositeElement":
            docs.append(doc)
    return docs, tables


def looks_like_base64(sb: str) -> bool:
    """Check if the string looks like base64."""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data: str) -> bool:
    """Check if the base64 data is an image by inspecting its magic bytes."""
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        return any(header.startswith(sig) for sig in image_signatures)
    except Exception:
        return False


def split_image_text_types(docs: List) -> Dict[str, List[str]]:
    """Split a list of retrieved raw docstore values into base64 images vs text/tables."""
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            content = doc.page_content.decode("utf-8")
        else:
            content = doc.decode("utf-8")
        if looks_like_base64(content) and is_image_data(content):
            b64_images.append(content)
        else:
            texts.append(content)
    return {"images": b64_images, "texts": texts}
