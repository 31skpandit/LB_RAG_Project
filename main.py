"""End-to-end orchestration of the Multimodal RAG pipeline.

Usage:
    python main.py                          # indexes every supported document in DATA_DIR (./data)
    python main.py --pdf-url https://sgp.fas.org/crs/misc/IF10244.pdf
    python main.py --pdf-path ./data/IF10244.pdf --query "Summarize the wildfire trend"
"""
import argparse
import base64
import glob
import os
import sys

from config import settings
from data_ingestion import discover_documents, download_pdf, partition_document
from processing import convert_tables_to_markdown, pack_content, split_elements_by_category
from rag_chain import build_multimodal_rag_chain, multimodal_rag_qa
from retrieval import content_id, create_multi_vector_retriever, get_doc_store, get_vector_store, index_manifest
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


def _prune_deleted_documents(current_paths: list, docstore, vectorstore) -> None:
    """Delete vectorstore/docstore entries for any previously-indexed file
    that's no longer among the currently discovered documents (e.g. removed
    from data/) -- without this, deleted files' chunks/embeddings stay in
    Chroma/Redis forever.
    """
    current_set = {os.path.abspath(p) for p in current_paths}
    removed = 0
    for tracked_path in index_manifest.all_manifest_file_paths():
        if os.path.abspath(tracked_path) not in current_set:
            stale_ids = index_manifest.forget_file(tracked_path)
            if stale_ids:
                vectorstore.delete(ids=stale_ids)
                docstore.mdelete(stale_ids)
            removed += 1
            print(f"Pruned deleted document from the index: {tracked_path}")
    if removed:
        print(f"Pruned {removed} deleted document(s) total.")


def _record_indexed_files(per_file_docs: dict) -> None:
    """After a successful build, record each processed file's resulting
    chunk/image IDs -- this is what lets future runs skip unchanged files
    and correctly prune deleted/changed ones (see retrieval/index_manifest.py).
    """
    for path, info in per_file_docs.items():
        ids = [content_id(pack_content(d)) for d in info["texts"]]
        ids += [content_id(pack_content(t)) for t in info["tables"]]
        for img_path in sorted(glob.glob(os.path.join(info["figures_dir"], "**", "*.jpg"), recursive=True)):
            with open(img_path, "rb") as f:
                ids.append(content_id(base64.b64encode(f.read()).decode("utf-8")))
        index_manifest.record_indexed(path, ids)


def build_pipeline(pdf_path: str = None, pdf_url: str = None, figures_dir: str = None, data_dir: str = None):
    """Run ingestion, summarization, and indexing. Returns a ready-to-query RAG chain."""
    settings.ensure_llm_key()

    figures_dir = figures_dir or settings.FIGURES_DIR
    # Pruning deleted files only makes sense when doc_paths represents "every
    # document currently in data/" -- an explicit --pdf-path/--pdf-url is a
    # narrower, single-file view and shouldn't be treated as "everything else
    # was deleted."
    is_auto_mode = pdf_path is None and pdf_url is None
    doc_paths = _resolve_document_paths(pdf_path, pdf_url, data_dir)

    vectorstore = get_vector_store()
    docstore = get_doc_store()

    if is_auto_mode:
        _prune_deleted_documents(doc_paths, docstore, vectorstore)

    # Skip unchanged files entirely (not for API cost -- that's already
    # handled by the per-chunk caches -- but for the wall-clock time of
    # re-parsing a document via Unstructured on every run for nothing).
    to_process = [p for p in doc_paths if not index_manifest.is_already_indexed(p)]
    skipped = len(doc_paths) - len(to_process)
    if skipped:
        print(f"Skipping {skipped} unchanged document(s) already indexed (nothing to re-partition).")

    # A file that changed since it was last indexed still needs its OLD
    # chunk IDs cleaned up first -- otherwise, if the new content produces
    # fewer chunks than before, the extra old ones become orphaned garbage.
    for path in to_process:
        stale_ids = index_manifest.forget_file(path)
        if stale_ids:
            vectorstore.delete(ids=stale_ids)
            docstore.mdelete(stale_ids)

    per_file_docs = {}
    for path in to_process:
        # Separate figures subdirectory per document so extracted image
        # filenames (e.g. "figure-1-1.jpg", named by Unstructured's internal
        # index) can't collide across different documents sharing one
        # figures_dir. Only PDFs/images actually populate this -- see
        # data_ingestion/document_loader.py's IMAGE_EXTRACTABLE_EXTENSIONS.
        doc_figures_dir = os.path.join(figures_dir, os.path.splitext(os.path.basename(path))[0])
        print(f"Partitioning document: {path}")
        elements = partition_document(path, doc_figures_dir)
        texts, tables = split_elements_by_category(elements)
        per_file_docs[path] = {"texts": texts, "tables": tables, "figures_dir": doc_figures_dir}

    text_docs = [d for info in per_file_docs.values() for d in info["texts"]]
    table_docs = [t for info in per_file_docs.values() for t in info["tables"]]

    convert_tables_to_markdown(table_docs)
    print(f"Extracted {len(text_docs)} text chunks and {len(table_docs)} tables from {len(to_process)} document(s).")

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

    # Only reached once summarization/embedding above actually succeeded, so
    # a run that errors out partway through won't wrongly mark a file done.
    if to_process:
        _record_indexed_files(per_file_docs)

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
