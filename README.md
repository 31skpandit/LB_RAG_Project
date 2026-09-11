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
│   ├── chunking.py             # Custom chunking strategies (recursive/sentence/paragraph)
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
| Text chunking strategy                     | `processing/chunking.py`                         |
| HTML table -> Markdown                     | `processing/table_converter.py`                  |
| Text & table summaries                     | `summarization/text_table_summarizer.py`         |
| Image summaries                            | `summarization/image_summarizer.py`              |
| Multi-vector retriever (Chroma + Redis)    | `retrieval/vector_store.py`, `retrieval/doc_store.py`, `retrieval/multi_vector_retriever.py` |
| Split retrieved images/text                | `processing/element_splitter.py`                 |
| Multimodal RAG chain + QA                  | `rag_chain/prompts.py`, `rag_chain/utils.py`, `rag_chain/pipeline.py` |

## Quick start

Run these steps in order — each one depends on the previous:

```bash
pip install -r requirements.txt         # 1. Python deps
bash scripts/install_system_deps.sh     # 2. tesseract, poppler, redis-stack-server
python scripts/download_nltk_data.py    # 3. NLTK punkt/POS tagger data
cp .env.example .env                    # 4. fill in OPENAI_API_KEY (required, no default)
python main.py --pdf-url https://sgp.fas.org/crs/misc/IF10244.pdf   # 5. run the pipeline
```

Step 4 must happen before step 5: `config/settings.py` loads `.env` on import, and
`main.py` reads `OPENAI_API_KEY` from it immediately. If it's missing, the pipeline
falls back to an interactive `getpass` prompt instead of failing outright.
`CHATGPT_MODEL` (`gpt-4o`) and `EMBEDDING_MODEL` (`text-embedding-3-small`) already
have working defaults in `.env.example`, so no other values are required to get started.

Then ask questions interactively, or import `rag_chain.pipeline.multimodal_rag_qa`
in a notebook/script for exploration.

### Running on Windows

`scripts/install_system_deps.sh` is Bash + `apt-get` based (Debian/Ubuntu only) and
will **not** work in `cmd`/PowerShell, or even under Git Bash (no `sudo`/`apt-get`
there). Run the whole Quick start sequence inside **WSL2 (Ubuntu)** instead:

```cmd
wsl -d Ubuntu
```

Then, from inside the WSL Ubuntu shell, `cd` to this project (Windows drives are
mounted under `/mnt/`, e.g. `cd "/mnt/d/Santosh/Data Science/Gen AI/Projects using Gen AI/RAG"`)
and run the same 5 Quick start steps above (`pip install -r requirements.txt`,
`bash scripts/install_system_deps.sh`, etc.) there. Use a Python venv inside WSL
(`python3 -m venv .venv && source .venv/bin/activate`) rather than reusing a Windows
virtualenv. Step 2 will install tesseract/poppler and start `redis-stack-server`
inside WSL — it's reachable from both WSL and Windows at `localhost:6379`, so
`REDIS_URL=redis://localhost:6379` works unchanged.

If Redis stops being reachable after a reboot (WSL doesn't keep background daemons
running across restarts), just re-run inside WSL:

```bash
redis-stack-server --daemonize yes
redis-cli ping   # should print PONG
```

## Configuration

All settings live in `config/settings.py` and are overridable via `.env` (see
`.env.example`). Notable ones:

| Variable                    | Default        | Purpose                                              |
|------------------------------|----------------|-------------------------------------------------------|
| `OPENAI_API_KEY`              | (required)     | LLM/embedding calls                                   |
| `CHATGPT_MODEL`                | `gpt-4o`       | Chat model for summaries & answers                    |
| `EMBEDDING_MODEL`              | `text-embedding-3-small` | Embedding model for the vector store        |
| `CHUNKING_STRATEGY`            | `by_title`     | Text chunking strategy, see below                     |
| `MAX_CHARACTERS`               | `4000`         | Max chars per chunk                                   |
| `NEW_AFTER_N_CHARS`            | `4000`         | Soft chunk-size target                                |
| `COMBINE_TEXT_UNDER_N_CHARS`   | `2000`         | Merge small elements below this size                  |

### Chunking strategies

`CHUNKING_STRATEGY` toggles how PDF text is chunked, so you can experiment with
how the pipeline behaves under different chunking scenarios without touching code:

| Value        | Behavior                                                              |
|--------------|------------------------------------------------------------------------|
| `by_title`   | Unstructured's native title-based chunking (default)                   |
| `basic`      | Unstructured's native fixed-size chunking                              |
| `recursive`  | LangChain `RecursiveCharacterTextSplitter` over raw extracted text     |
| `sentence`   | Groups whole sentences (NLTK) up to `MAX_CHARACTERS`                   |
| `paragraph`  | Groups whole paragraphs up to `MAX_CHARACTERS`                        |

`by_title`/`basic` are applied natively during PDF partitioning; the other three
are implemented in `processing/chunking.py` and applied afterwards on the raw
unchunked elements. Set it in `.env`, e.g. `CHUNKING_STRATEGY=sentence`, then
rerun `python main.py ...` to compare results.

## Why modular?

Each folder is a seam where you can swap implementations to experiment, e.g.:
- `data_ingestion`: try PyMuPDF, LlamaParse, or Azure Document Intelligence instead of Unstructured.
- `retrieval`: swap Chroma for FAISS/Pinecone, or Redis for SQLite/local disk.
- `summarization`: try different LLMs (Claude, Gemini) or prompt strategies.
- `rag_chain`: experiment with different prompt structures, reranking, or citation formats.

Because each stage only depends on plain Python objects (lists of `Document`s, dicts,
base64 strings), you can replace any single file without cascading changes elsewhere.

## Running through WSL in terminal of VS Code :- 

wsl --install
wsl
cd /mnt/d/Santosh/Data\ Science/Gen\ AI/Projects\ using\ Gen\ AI/RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
