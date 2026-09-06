"""Partition a PDF into text, table, and image elements using Unstructured.

Run twice (tables pass + chunked-text pass) because chunking currently drops
table elements in the Unstructured library. See:
https://github.com/Unstructured-IO/unstructured/issues/3827
"""
from typing import List

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document

from config import settings


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
    """Load the PDF with title-based chunking and return CompositeElement text chunks."""
    figures_dir = figures_dir or settings.FIGURES_DIR
    loader = UnstructuredPDFLoader(
        file_path=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy=settings.CHUNKING_STRATEGY,
        max_characters=settings.MAX_CHARACTERS,
        new_after_n_chars=settings.NEW_AFTER_N_CHARS,
        combine_text_under_n_chars=settings.COMBINE_TEXT_UNDER_N_CHARS,
        mode="elements",
        image_output_dir_path=figures_dir,
    )
    return loader.load()


def partition_pdf(pdf_path: str, figures_dir: str = None) -> List[Document]:
    """Return combined text + table elements extracted from the PDF."""
    tables = load_tables(pdf_path, figures_dir)
    texts = load_chunked_text(pdf_path, figures_dir)
    return texts + tables
