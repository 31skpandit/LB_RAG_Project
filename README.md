# Multimodal RAG System (GPT-4o)

Modular re-implementation of `Multimodal_RAG_System_with_GPT_4o.ipynb` as a set of
independent, swappable components. The goal is to make it easy to experiment with
alternate technologies for each stage of the pipeline (different parsers, embedding
models, vector stores, doc stores, or LLMs) without touching the rest of the system.

## Project Structure

```
RAG/
├── config/
│   └── settings.py            # Central configuration (env vars, model names, paths)
├── data_ingestion/
│   ├── downloader.py          # Fetch source documents (e.g. PDFs)
│   └── pdf_loader.py          # Partition PDFs into text/table/image elements (Unstructured)
├── processing/
│   ├── element_splitter.py    # Categorize raw elements, split retrieved base64/text results
│   └── table_converter.py     # HTML table -> Markdown conversion
├── summarization/
│   ├── text_table_summarizer.py  # LLM summaries for text & table chunks
│   └── image_summarizer.py       # LLM summaries for images (base64 + vision prompt)
├── retrieval/
│   ├── vector_store.py         # Chroma vector store (summary embeddings)
│   ├── doc_store.py            # Redis doc store (raw text/table/image content)
│   └── multi_vector_retriever.py  # Wires vector store + doc store together
├── rag_chain/
│   ├── prompts.py               # Prompt templates
│   ├── utils.py                 # base64/image helpers
│   └── pipeline.py              # End-to-end multimodal RAG chain + QA helper
├── scripts/
│   ├── install_system_deps.sh   # tesseract/poppler/redis system deps
│   ├── download_nltk_data.py    # nltk punkt/POS tagger downloads
│   └── run_pipeline.py          # CLI entry point to run ingestion + build retriever
├── main.py                      # Orchestrates the full pipeline end-to-end
├── requirements.txt
└── .env.example
```

## How the pieces map to the original notebook

| Notebook Section                          | Module                                          |
|--------------------------------------------|--------------------------------------------------|
| Install dependencies                       | `requirements.txt`, `scripts/install_system_deps.sh` |
| NLTK downloads                             | `scripts/download_nltk_data.py`                  |
| Download PDF                               | `data_ingestion/downloader.py`                   |
| Partition PDF (tables/text/images)         | `data_ingestion/pdf_loader.py`                   |
| Separate text vs table elements            | `processing/element_splitter.py`                 |
| HTML table -> Markdown                     | `processing/table_converter.py`                  |
| Text & table summaries                     | `summarization/text_table_summarizer.py`         |
| Image summaries                            | `summarization/image_summarizer.py`              |
| Multi-vector retriever (Chroma + Redis)    | `retrieval/vector_store.py`, `retrieval/doc_store.py`, `retrieval/multi_vector_retriever.py` |
| Split retrieved images/text                | `processing/element_splitter.py`                 |
| Multimodal RAG chain + QA                  | `rag_chain/prompts.py`, `rag_chain/utils.py`, `rag_chain/pipeline.py` |

## Quick start

```bash
pip install -r requirements.txt
bash scripts/install_system_deps.sh     # tesseract, poppler, redis-stack-server
python scripts/download_nltk_data.py
cp .env.example .env                    # fill in OPENAI_API_KEY
python main.py --pdf-url https://sgp.fas.org/crs/misc/IF10244.pdf
```

Then ask questions interactively, or import `rag_chain.pipeline.multimodal_rag_qa`
in a notebook/script for exploration.

## Why modular?

Each folder is a seam where you can swap implementations to experiment, e.g.:
- `data_ingestion`: try PyMuPDF, LlamaParse, or Azure Document Intelligence instead of Unstructured.
- `retrieval`: swap Chroma for FAISS/Pinecone, or Redis for SQLite/local disk.
- `summarization`: try different LLMs (Claude, Gemini) or prompt strategies.
- `rag_chain`: experiment with different prompt structures, reranking, or citation formats.

Because each stage only depends on plain Python objects (lists of `Document`s, dicts,
base64 strings), you can replace any single file without cascading changes elsewhere.
