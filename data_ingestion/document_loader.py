"""Partition a document (PDF, DOCX, PPTX, TXT, HTML, or image) into text/table
elements using Unstructured's auto-detecting `partition()`.

Partitions ONCE per document, not twice. An earlier version of this module
(and the original PDF-only code it replaced) called LangChain's
`UnstructuredFileLoader` separately for a "tables pass" and a "chunked-text
pass" -- each invocation reruns partition() from scratch, including the
expensive `strategy="hi_res"` layout/OCR model pass over every page. For a
126-page real document that's roughly double the cost for no reason: Table
elements and chunked text can both be derived from ONE raw partition result.
Unstructured's own chunking functions (chunk_by_title/chunk_elements) are
lightweight, in-memory post-processing over already-partitioned elements --
they don't re-run OCR -- so we call them directly instead of re-partitioning.

The underlying reason a two-pass split is needed AT ALL (tables from raw,
unchunked elements specifically) is that Unstructured's chunking drops Table
elements -- confirmed to affect every format equally (one shared chunking
implementation applied to all partitioners). See:
https://github.com/Unstructured-IO/unstructured/issues/3827
"""
import os
from typing import Any, List

from langchain_core.documents import Document
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from config import settings
from processing.chunking import apply_custom_chunking

# Only PDF and standalone image files support Unstructured's
# extract_images_in_pdf/image_output_dir_path kwargs -- docx/pptx/txt/html have
# no equivalent mechanism (docx needs custom Python-level picture-partitioner
# registration, pptx has no output-dir hook, html/txt have no image concept at
# all). Passing these kwargs for other formats is a silent no-op per
# Unstructured's own routing code, but we only pass them where they'll
# actually do something, to keep the code's intent honest.
IMAGE_EXTRACTABLE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def _image_kwargs(file_path: str, figures_dir: str) -> dict:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTRACTABLE_EXTENSIONS:
        return {"extract_images_in_pdf": True, "image_output_dir_path": figures_dir}
    return {}


def _elements_to_documents(elements: List[Any], file_path: str) -> List[Document]:
    """Convert native Unstructured Elements to LangChain Documents.

    Mirrors langchain_community.document_loaders.unstructured.UnstructuredFileLoader's
    own mode="elements" conversion exactly (base {"source": file_path} metadata,
    then element.metadata.to_dict(), then category) so downstream code -- which
    expects doc.metadata["category"]/["text_as_html"]/["filename"]/["page_number"]
    -- sees exactly the same shape it always has.
    """
    docs = []
    for element in elements:
        metadata: dict = {"source": file_path}
        if hasattr(element, "metadata"):
            metadata.update(element.metadata.to_dict())
        if hasattr(element, "category"):
            metadata["category"] = element.category
        docs.append(Document(page_content=str(element), metadata=metadata))
    return docs


def _partition_raw(file_path: str, figures_dir: str) -> List[Any]:
    """Partition once, returning raw (unchunked) native Elements -- Table
    elements included, since it's chunking that drops them, not partitioning."""
    return partition(
        filename=file_path,
        strategy=settings.UNSTRUCTURED_STRATEGY,
        # NOT infer_table_structure=True: auto.partition() computes that itself
        # from skip_infer_table_types and passes it internally to e.g.
        # partition_pdf() -- also passing it explicitly collides ("got multiple
        # values for keyword argument 'infer_table_structure'"). An empty skip
        # list means "don't skip table inference for any type" (pdf/jpg/png/heic
        # default to skipped; everything else already infers tables by default).
        skip_infer_table_types=[],
        **_image_kwargs(file_path, figures_dir),
    )


def _extract_tables(raw_elements: List[Any], file_path: str) -> List[Document]:
    tables = [e for e in raw_elements if getattr(e, "category", None) == "Table"]
    return _elements_to_documents(tables, file_path)


def _chunk_text(raw_elements: List[Any], file_path: str) -> List[Document]:
    """Chunk raw elements using the configured CHUNKING_STRATEGY.

    "by_title"/"basic" call Unstructured's own chunking functions directly
    (cheap, in-memory -- no re-partitioning). "recursive"/"sentence"/
    "paragraph" are applied afterwards (see processing/chunking.py) so
    different chunking approaches can be toggled to compare pipeline behavior.
    """
    strategy = settings.CHUNKING_STRATEGY

    if strategy == "by_title":
        chunked = chunk_by_title(
            raw_elements,
            max_characters=settings.MAX_CHARACTERS,
            new_after_n_chars=settings.NEW_AFTER_N_CHARS,
            combine_text_under_n_chars=settings.COMBINE_TEXT_UNDER_N_CHARS,
        )
        return _elements_to_documents(chunked, file_path)

    if strategy == "basic":
        chunked = chunk_elements(
            raw_elements,
            max_characters=settings.MAX_CHARACTERS,
            new_after_n_chars=settings.NEW_AFTER_N_CHARS,
        )
        return _elements_to_documents(chunked, file_path)

    # Custom strategy (recursive/sentence/paragraph): chunk the raw text ourselves.
    raw_docs = _elements_to_documents(raw_elements, file_path)
    return apply_custom_chunking(
        strategy,
        raw_docs,
        max_characters=settings.MAX_CHARACTERS,
        new_after_n_chars=settings.NEW_AFTER_N_CHARS,
    )


def load_tables(file_path: str, figures_dir: str = None) -> List[Document]:
    """Partition the document and keep only Table elements."""
    figures_dir = figures_dir or settings.FIGURES_DIR
    return _extract_tables(_partition_raw(file_path, figures_dir), file_path)


def load_chunked_text(file_path: str, figures_dir: str = None) -> List[Document]:
    """Partition the document and chunk its text using the configured CHUNKING_STRATEGY."""
    figures_dir = figures_dir or settings.FIGURES_DIR
    return _chunk_text(_partition_raw(file_path, figures_dir), file_path)


def partition_document(file_path: str, figures_dir: str = None) -> List[Document]:
    """Return combined text + table elements extracted from the document.

    Partitions exactly once and derives both tables and chunked text from the
    same raw result -- see module docstring for why this matters.
    """
    figures_dir = figures_dir or settings.FIGURES_DIR
    raw_elements = _partition_raw(file_path, figures_dir)
    return _chunk_text(raw_elements, file_path) + _extract_tables(raw_elements, file_path)
