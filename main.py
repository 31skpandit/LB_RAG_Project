"""End-to-end orchestration of the Multimodal RAG pipeline.

Usage:
    python main.py                          # indexes every supported document in DATA_DIR (./data)
    python main.py --pdf-url https://sgp.fas.org/crs/misc/IF10244.pdf
    python main.py --pdf-path ./data/IF10244.pdf --query "Summarize the wildfire trend"
"""
import argparse
import os
import sys

from config import settings
from data_ingestion import discover_documents, download_pdf, partition_document
from processing import convert_tables_to_markdown, pack_content, split_elements_by_category
from rag_chain import build_multimodal_rag_chain, multimodal_rag_qa
from retrieval import create_multi_vector_retriever, get_doc_store, get_vector_store
from summarization import generate_img_summaries, summarize_texts_and_tables


def _resolve_document_paths(pdf_path: str = None, pdf_url: str = None, data_dir: str = None) -> list:
    """Pick which document(s) to process.

    Explicit --pdf-path/--pdf-url always wins (single file, any supported
    format despite the flag name). Otherwise, auto-discover every supported
    document in `data_dir` and process all of them. If none are found there,
    fail with a clear message rather than silently downloading a sample --
    this pipeline is meant to run only on documents you've placed in data/.
    """
    if pdf_path:
        return [pdf_path]
    if pdf_url:
        return [download_pdf(url=pdf_url)]

    discovered = discover_documents(data_dir)
    if not discovered:
        raise RuntimeError(
            f"No documents found in {data_dir or settings.DATA_DIR!r}. "
            "Add a PDF/DOCX/PPTX/TXT/HTML/image file there and re-run "
            "(or pass --pdf-path/--pdf-url to point at one file explicitly)."
        )
    return discovered


def build_pipeline(pdf_path: str = None, pdf_url: str = None, figures_dir: str = None, data_dir: str = None):
    """Run ingestion, summarization, and indexing. Returns a ready-to-query RAG chain."""
    settings.ensure_llm_key()

    figures_dir = figures_dir or settings.FIGURES_DIR
    doc_paths = _resolve_document_paths(pdf_path, pdf_url, data_dir)

    text_docs, table_docs = [], []
    for path in doc_paths:
        # Separate figures subdirectory per document so extracted image
        # filenames (e.g. "figure-1-1.jpg", named by Unstructured's internal
        # index) can't collide across different documents sharing one
        # figures_dir. Only PDFs/images actually populate this -- see
        # data_ingestion/document_loader.py's IMAGE_EXTRACTABLE_EXTENSIONS.
        doc_figures_dir = os.path.join(figures_dir, os.path.splitext(os.path.basename(path))[0])
        print(f"Partitioning document: {path}")
        elements = partition_document(path, doc_figures_dir)
        texts, tables = split_elements_by_category(elements)
        text_docs.extend(texts)
        table_docs.extend(tables)

    convert_tables_to_markdown(table_docs)
    print(f"Extracted {len(text_docs)} text chunks and {len(table_docs)} tables from {len(doc_paths)} document(s).")

    # Plain content for summarization (the LLM doesn't need the citation wrapper).
    text_contents = [d.page_content for d in text_docs]
    table_contents = [t.page_content for t in table_docs]

    print("Summarizing text and tables...")
    text_summaries, table_summaries = summarize_texts_and_tables(text_contents, table_contents)

    # Content actually stored in the docstore, packed with source/page metadata
    # so answers can always cite where they came from (see processing/citations.py
    # and processing/element_splitter.py's split_image_text_types).
    packed_texts = [pack_content(d) for d in text_docs]
    packed_tables = [pack_content(t) for t in table_docs]

    print(f"Summarizing images in: {figures_dir}")
    imgs_base64, image_summaries = generate_img_summaries(figures_dir)
    print(f"Summarized {len(imgs_base64)} images.")

    print("Building multi-vector retriever (Chroma + Redis)...")
    vectorstore = get_vector_store()
    docstore = get_doc_store()
    retriever = create_multi_vector_retriever(
        docstore,
        vectorstore,
        text_summaries,
        packed_texts,
        table_summaries,
        packed_tables,
        image_summaries,
        imgs_base64,
    )

    return build_multimodal_rag_chain(retriever)


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG pipeline")
    parser.add_argument("--pdf-url", default=None, help="URL of a document to download and index")
    parser.add_argument("--pdf-path", default=None, help="Local path to a single document to index")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory to auto-discover documents from when --pdf-path/--pdf-url are omitted (default: DATA_DIR, ./data)",
    )
    parser.add_argument("--figures-dir", default=None, help="Directory to extract images into")
    parser.add_argument("--query", default=None, help="Run a single query non-interactively and exit")
    args = parser.parse_args()

    chain = build_pipeline(
        pdf_path=args.pdf_path, pdf_url=args.pdf_url, figures_dir=args.figures_dir, data_dir=args.data_dir
    )

    if args.query:
        multimodal_rag_qa(chain, args.query)
        return

    print("\nPipeline ready. Type a question (or 'exit' to quit).")
    while True:
        query = input("\nQuery> ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break
        multimodal_rag_qa(chain, query)


if __name__ == "__main__":
    sys.exit(main())
