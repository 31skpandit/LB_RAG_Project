"""Partition a PDF into text, table, and image elements using Unstructured.

Run twice (tables pass + chunked-text pass) because chunking currently drops
table elements in the Unstructured library. See:
https://github.com/Unstructured-IO/unstructured/issues/3827
"""
from typing import List

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document

from config import settings
from processing.chunking import NATIVE_STRATEGIES, apply_custom_chunking


def load_tables(pdf_path: str, figures_dir: str = None) -> List[Document]:
    """Load the PDF once with default settings and keep only Table elements."""
    figures_dir = figures_dir or settings.FIGURES_DIR
    loader = UnstructuredPDFLoader(
        file_path=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        mode="elements",
        image_output_dir_path=figures_dir,
    )
    data = loader.load()
    return [doc for doc in data if doc.metadata["category"] == "Table"]


def load_chunked_text(pdf_path: str, figures_dir: str = None) -> List[Document]:
    """Load the PDF and chunk its text using the configured CHUNKING_STRATEGY.

    "by_title"/"basic" are chunked natively by Unstructured during partitioning.
    "recursive"/"sentence"/"paragraph" are applied afterwards (see processing/chunking.py)
    so different chunking approaches can be toggled to compare pipeline behavior.
    """
    figures_dir = figures_dir or settings.FIGURES_DIR
    strategy = settings.CHUNKING_STRATEGY
    loader_kwargs = dict(
        file_path=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        mode="elements",
        image_output_dir_path=figures_dir,
    )

    if strategy in NATIVE_STRATEGIES:
        loader = UnstructuredPDFLoader(
            chunking_strategy=strategy,
            max_characters=settings.MAX_CHARACTERS,
            new_after_n_chars=settings.NEW_AFTER_N_CHARS,
            combine_text_under_n_chars=settings.COMBINE_TEXT_UNDER_N_CHARS,
            **loader_kwargs,
        )
        return loader.load()

    # Custom strategy: load raw (unchunked) elements, then chunk them ourselves.
    raw_elements = UnstructuredPDFLoader(**loader_kwargs).load()
    return apply_custom_chunking(
        strategy,
        raw_elements,
        max_characters=settings.MAX_CHARACTERS,
        new_after_n_chars=settings.NEW_AFTER_N_CHARS,
    )


def partition_pdf(pdf_path: str, figures_dir: str = None) -> List[Document]:
    """Return combined text + table elements extracted from the PDF."""
    tables = load_tables(pdf_path, figures_dir)
    texts = load_chunked_text(pdf_path, figures_dir)
    return texts + tables
