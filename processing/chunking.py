"""Custom text chunking strategies, used for experimenting with pipeline behavior.

Unstructured natively handles "by_title" and "basic" chunking during PDF
partitioning (see data_ingestion/pdf_loader.py). The strategies here
("recursive", "sentence", "paragraph") are applied afterwards on raw,
unchunked text elements so you can compare RAG quality/behavior across
different chunking approaches by only toggling `settings.CHUNKING_STRATEGY`.
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

NATIVE_STRATEGIES = {"by_title", "basic"}
CUSTOM_STRATEGIES = {"recursive", "sentence", "paragraph"}
SUPPORTED_STRATEGIES = NATIVE_STRATEGIES | CUSTOM_STRATEGIES


def _make_composite_docs(chunks: List[str]) -> List[Document]:
    """Wrap raw text chunks as Documents matching Unstructured's CompositeElement shape."""
    return [
        Document(page_content=chunk.strip(), metadata={"category": "CompositeElement"})
        for chunk in chunks
        if chunk.strip()
    ]


def chunk_recursive(text: str, new_after_n_chars: int) -> List[Document]:
    """Split text recursively by paragraph -> line -> sentence -> word as needed."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=new_after_n_chars,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return _make_composite_docs(splitter.split_text(text))


def chunk_by_sentence(text: str, max_characters: int) -> List[Document]:
    """Group whole sentences together until the combined length nears max_characters."""
    from nltk.tokenize import sent_tokenize

    chunks, current = [], ""
    for sentence in sent_tokenize(text):
        if current and len(current) + len(sentence) + 1 > max_characters:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return _make_composite_docs(chunks)


def chunk_by_paragraph(text: str, max_characters: int) -> List[Document]:
    """Group whole paragraphs (blank-line separated) together until max_characters is reached."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_characters:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return _make_composite_docs(chunks)


def apply_custom_chunking(
    strategy: str,
    raw_documents: List[Document],
    max_characters: int,
    new_after_n_chars: int,
) -> List[Document]:
    """Dispatch to the requested custom chunking strategy over raw (unchunked) text elements."""
    text_elements = [d for d in raw_documents if d.metadata.get("category") != "Table"]
    text = "\n\n".join(doc.page_content for doc in text_elements if doc.page_content.strip())

    if strategy == "recursive":
        return chunk_recursive(text, new_after_n_chars)
    if strategy == "sentence":
        return chunk_by_sentence(text, max_characters)
    if strategy == "paragraph":
        return chunk_by_paragraph(text, max_characters)

    raise ValueError(
        f"Unsupported custom chunking strategy: {strategy!r}. "
        f"Expected one of {sorted(CUSTOM_STRATEGIES)}."
    )
