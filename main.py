"""End-to-end orchestration of the Multimodal RAG pipeline.

Usage:
    python main.py --pdf-url https://sgp.fas.org/crs/misc/IF10244.pdf
    python main.py --pdf-path ./data/IF10244.pdf --query "Summarize the wildfire trend"
"""
import argparse
import sys

from config import settings
from data_ingestion import download_pdf, partition_pdf
from processing import convert_tables_to_markdown, split_elements_by_category
from rag_chain import build_multimodal_rag_chain, multimodal_rag_qa
from retrieval import create_multi_vector_retriever, get_doc_store, get_vector_store
from summarization import generate_img_summaries, summarize_texts_and_tables


def build_pipeline(pdf_path: str = None, pdf_url: str = None, figures_dir: str = None):
    """Run ingestion, summarization, and indexing. Returns a ready-to-query RAG chain."""
    settings.ensure_llm_key()

    figures_dir = figures_dir or settings.FIGURES_DIR
    pdf_path = pdf_path or download_pdf(url=pdf_url)

    print(f"Partitioning PDF: {pdf_path}")
    elements = partition_pdf(pdf_path, figures_dir)
    text_docs, table_docs = split_elements_by_category(elements)
    convert_tables_to_markdown(table_docs)
    print(f"Extracted {len(text_docs)} text chunks and {len(table_docs)} tables.")

    text_contents = [d.page_content for d in text_docs]
    table_contents = [t.page_content for t in table_docs]

    print("Summarizing text and tables...")
    text_summaries, table_summaries = summarize_texts_and_tables(text_contents, table_contents)

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
        text_contents,
        table_summaries,
        table_contents,
        image_summaries,
        imgs_base64,
    )

    return build_multimodal_rag_chain(retriever)


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG pipeline (GPT-4o)")
    parser.add_argument("--pdf-url", default=None, help="URL of the PDF to download and index")
    parser.add_argument("--pdf-path", default=None, help="Local path to an already-downloaded PDF")
    parser.add_argument("--figures-dir", default=None, help="Directory to extract images into")
    parser.add_argument("--query", default=None, help="Run a single query non-interactively and exit")
    args = parser.parse_args()

    chain = build_pipeline(pdf_path=args.pdf_path, pdf_url=args.pdf_url, figures_dir=args.figures_dir)

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
