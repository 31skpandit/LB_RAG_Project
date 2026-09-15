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
│   ├── downloader.py          # Discover documents in data/, download-by-URL helper
│   └── document_loader.py     # Partition any supported document into text/table/image elements (Unstructured)
├── processing/
│   ├── element_splitter.py    # Categorize raw elements, split retrieved results into images/texts/citations
│   ├── chunking.py             # Custom chunking strategies (recursive/sentence/paragraph)
│   ├── table_converter.py     # HTML table -> Markdown conversion
│   └── citations.py           # Pack/unpack source (filename + page) metadata for citations
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
│   └── pipeline.py              # End-to-end multimodal RAG chain + QA helper (answer + citations)
├── api/
│   ├── schemas.py              # Pydantic request/response models
│   └── main.py                 # FastAPI app (POST /query, GET /health) + Swagger UI
├── scripts/
│   ├── install_system_deps.sh   # tesseract/poppler/redis system deps
│   ├── download_nltk_data.py    # nltk punkt/POS tagger downloads
│   └── run_pipeline.py          # CLI entry point to run ingestion + build retriever
├── main.py                      # Orchestrates the full pipeline end-to-end (CLI)
├── requirements.txt
└── .env.example
```

## How the pieces map to the original notebook

| Notebook Section                          | Module                                          |
|--------------------------------------------|--------------------------------------------------|
| Install dependencies                       | `requirements.txt`, `scripts/install_system_deps.sh` |
| NLTK downloads                             | `scripts/download_nltk_data.py`                  |
| Discover/download documents                | `data_ingestion/downloader.py`                   |
| Partition documents (tables/text/images)   | `data_ingestion/document_loader.py`              |
| Separate text vs table elements            | `processing/element_splitter.py`                 |
| Text chunking strategy                     | `processing/chunking.py`                         |
| HTML table -> Markdown                     | `processing/table_converter.py`                  |
| Text & table summaries                     | `summarization/text_table_summarizer.py`         |
| Image summaries                            | `summarization/image_summarizer.py`              |
| Multi-vector retriever (Chroma + Redis)    | `retrieval/vector_store.py`, `retrieval/doc_store.py`, `retrieval/multi_vector_retriever.py` |
| Split retrieved images/text, build citations | `processing/element_splitter.py`, `processing/citations.py` |
| Multimodal RAG chain + QA                  | `rag_chain/prompts.py`, `rag_chain/utils.py`, `rag_chain/pipeline.py` |
| Interactive API                            | `api/main.py`, `api/schemas.py`                  |

## Quick start

Run these steps in order — each one depends on the previous:

```bash
pip install -r requirements.txt         # 1. Python deps
bash scripts/install_system_deps.sh     # 2. tesseract, poppler, redis-server
python scripts/download_nltk_data.py    # 3. NLTK punkt/POS tagger data
cp .env.example .env                    # 4. set LLM_PROVIDER + its API key (see Configuration)
cp your-document.pdf data/              # 5. add whatever you want indexed (PDF/DOCX/PPTX/TXT/HTML/image)
python main.py                          # 6. run the pipeline against everything in data/
```

Step 4 must happen before step 6: `config/settings.py` loads `.env` on import, and
`main.py` reads the API key for the configured `LLM_PROVIDER` immediately. If it's
missing (and the provider isn't `ollama`, which needs no key), the pipeline falls
back to an interactive `getpass` prompt instead of failing outright. Each provider's
model names already have working defaults in `.env.example` — see the LLM provider
toggle table below for how to switch providers.

**The pipeline only ever runs on documents you place in `data/`** — it auto-discovers
every supported file there (PDF, DOCX, PPTX, TXT, HTML, or a standalone image) and
indexes all of them into one combined knowledge base. If `data/` has nothing
supported in it, `main.py` fails immediately with a clear message rather than
silently downloading anything. To index just one specific file instead (any
format), use `--pdf-path ./path/to/file.docx`; to download-then-index from a URL,
use `--pdf-url https://...` (the flag names are historical, they accept any
supported format now).

Then ask questions interactively, import `rag_chain.pipeline.multimodal_rag_qa`
in a notebook/script for exploration, or run the [FastAPI service](#interactive-api-fastapi--swagger)
for a browser-based Q&A experience with citations.

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
virtualenv. Step 2 will install tesseract/poppler and start plain `redis-server`
inside WSL — it's reachable from both WSL and Windows at `localhost:6379`, so
`REDIS_URL=redis://localhost:6379` works unchanged. (Plain Redis is enough here:
`retrieval/doc_store.py` only uses it as a key-value store for raw text/table/
image content — vector search is handled by Chroma — so none of Redis Stack's
extra modules are needed.)

If Redis stops being reachable after a reboot (WSL doesn't keep background daemons
running across restarts), just re-run inside WSL:

```bash
redis-server --daemonize yes
redis-cli ping   # should print PONG
```

## Configuration

All settings live in `config/settings.py` and are overridable via `.env` (see
`.env.example`). Notable ones:

| Variable                    | Default        | Purpose                                              |
|------------------------------|----------------|-------------------------------------------------------|
| `LLM_PROVIDER`                 | `gemini`       | `openai`, `gemini`, `ollama`, `qwen`, `groq`, or `deepseek` — see below |
| `EMBEDDING_PROVIDER`           | matches `LLM_PROVIDER` | Same list minus `groq`/`deepseek` (no embeddings API) — see below |
| `OPENAI_API_KEY`              | (required for `openai`) | LLM/embedding calls                          |
| `CHATGPT_MODEL`                | `gpt-4o`       | Chat model for summaries & answers                    |
| `EMBEDDING_MODEL`              | `text-embedding-3-small` | Embedding model for the vector store        |
| `GOOGLE_API_KEY`               | (required for `gemini`) | LLM/embedding calls, free tier             |
| `GEMINI_CHAT_MODEL`            | `gemini-3.6-flash` | Chat model for summaries & answers                |
| `GEMINI_EMBEDDING_MODEL`       | `models/gemini-embedding-001` | Embedding model for the vector store |
| `OLLAMA_BASE_URL`              | `http://localhost:11434` | Local Ollama server URL                     |
| `OLLAMA_CHAT_MODEL`            | `qwen3-vl:2b`  | Chat model for summaries & answers (must support vision) |
| `OLLAMA_EMBEDDING_MODEL`       | `qwen3-embedding:0.6b` | Embedding model for the vector store       |
| `OLLAMA_NUM_CTX`               | `4096`         | Context window passed to Ollama (see note below) |
| `QWEN_API_KEY`                 | (required for `qwen`) | LLM/embedding calls, free tier via DashScope  |
| `QWEN_BASE_URL`                | DashScope intl. compatible endpoint | Qwen/DashScope API base URL        |
| `QWEN_CHAT_MODEL`              | `qwen-vl-plus` | Chat model for summaries & answers (must support vision) |
| `QWEN_EMBEDDING_MODEL`         | `text-embedding-v3` | Embedding model for the vector store              |
| `GROQ_API_KEY`                 | (required for `groq`) | LLM/embedding calls, free tier, no country lock |
| `GROQ_BASE_URL`                | `https://api.groq.com/openai/v1` | Groq's OpenAI-compatible API base URL   |
| `GROQ_CHAT_MODEL`              | `qwen/qwen3.6-27b` | Chat model for summaries & answers (must support vision) |
| `DEEPSEEK_API_KEY`             | (required for `deepseek`) | LLM calls, pay-as-you-go, no free tier    |
| `DEEPSEEK_BASE_URL`            | `https://api.deepseek.com` | DeepSeek's OpenAI-compatible API base URL |
| `DEEPSEEK_CHAT_MODEL`          | `deepseek-flash` | Chat model for summaries & answers (must support vision) |
| `CHUNKING_STRATEGY`            | `by_title`     | Text chunking strategy, see below                     |
| `MAX_CHARACTERS`               | `4000`         | Max chars per chunk                                   |
| `NEW_AFTER_N_CHARS`            | `4000`         | Soft chunk-size target                                |
| `COMBINE_TEXT_UNDER_N_CHARS`   | `2000`         | Merge small elements below this size                  |
| `LLM_MAX_TOKENS`               | `200`          | Caps output tokens on every chat call, every provider (see Cost controls below) |
| `RETRIEVAL_K`                  | `3`            | Documents retrieved per question (see Cost controls below) |
| `SUMMARY_PROVIDER`             | `openai`       | `openai` or `gemini` only — used for indexing-time summarization instead of `LLM_PROVIDER` (see Cost controls below) |
| `SUMMARY_OPENAI_MODEL`         | `gpt-4o-mini`  | Cheap summarization model when `SUMMARY_PROVIDER=openai` |
| `SUMMARY_GEMINI_MODEL`         | `gemini-3.6-flash` | Cheap summarization model when `SUMMARY_PROVIDER=gemini` |

### LLM provider toggle

`config/llm_factory.py` builds the chat/vision model from `LLM_PROVIDER` and
the embedding model from `EMBEDDING_PROVIDER` (defaults to `LLM_PROVIDER`, so
most setups only ever touch one setting). Every pipeline stage (summarization,
image summarization, embeddings, answer synthesis) goes through this factory,
so switching providers is a one- or two-line `.env` change — no code edits.

| Provider | Chat + vision (`LLM_PROVIDER`) | Embeddings (`EMBEDDING_PROVIDER`) | API key needed? |
|----------|--------------------------------|--------------------------------|------------------|
| `openai` | `CHATGPT_MODEL` (`gpt-4o`)      | `EMBEDDING_MODEL`              | `OPENAI_API_KEY` |
| `gemini` | `GEMINI_CHAT_MODEL`             | `GEMINI_EMBEDDING_MODEL`       | `GOOGLE_API_KEY` (free tier) |
| `ollama` | `OLLAMA_CHAT_MODEL`             | `OLLAMA_EMBEDDING_MODEL`       | none (runs locally) |
| `qwen`   | `QWEN_CHAT_MODEL`               | `QWEN_EMBEDDING_MODEL`         | `QWEN_API_KEY` (free tier) |
| `groq`   | `GROQ_CHAT_MODEL`               | *(none — has no embeddings API)* | `GROQ_API_KEY` (free tier) |
| `deepseek` | `DEEPSEEK_CHAT_MODEL`         | *(none — has no embeddings API)* | `DEEPSEEK_API_KEY` (pay-as-you-go) |

**`groq` and `deepseek` are chat/vision-only** — neither has an embeddings API.
For Groq, this doesn't actually exist for at least some accounts — despite
third-party docs/blogs claiming a `nomic-embed-text-v1_5` model, it 404s
("does not exist or you do not have access to it") and doesn't appear at all
in the account's live `GET /openai/v1/models` response. For DeepSeek, its
hosted API has only ever exposed chat completions. So when `LLM_PROVIDER` is
either of these, you must set `EMBEDDING_PROVIDER` to a different provider
(e.g. `gemini`) — startup raises a clear error if you leave it defaulting to one of them.

To use Groq: create a free key at
[console.groq.com/keys](https://console.groq.com/keys) (no country
restriction, unlike DashScope), set `GROQ_API_KEY` and `EMBEDDING_PROVIDER`
(e.g. `gemini`, with its own key configured) in `.env`, then set
`LLM_PROVIDER=groq`. It's routed through Groq's OpenAI-compatible endpoint via
`langchain-openai`, so no extra dependency is needed — same approach as `qwen`.

**Groq free-tier rate limits** (per Groq's own docs, org-wide and subject to
change — check [console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits)
for your account's current limits):
- The default vision chat model (`qwen/qwen3.6-27b`) and other Qwen/GPT-OSS
  models: **30 requests/min, 1,000 requests/day, 8,000 tokens/min, 200,000
  tokens/day** -- but **output tokens are capped separately and tighter, at
  1,000/min (OTPM)**. This is the limit you'll actually hit in practice: with
  no cap set, the client requests as many output tokens as the model's max
  context allows, which trips a 429 on the very first call. `LLM_MAX_TOKENS`
  (default `200`, applies to every provider -- see Cost controls below) caps
  each response so a single call fits comfortably under the OTPM budget.
- Limits apply per organization, not per API key — multiple keys don't add up.
- **Groq's vision models are labeled "preview"** — Alibaba/Groq can rename or
  retire them with little notice. If `GROQ_CHAT_MODEL` ever 404s, check
  [console.groq.com/docs/vision](https://console.groq.com/docs/vision) for
  the current model ID.
- If you exceed limits, requests get HTTP 429'd; the RAG pipeline doesn't
  currently retry/backoff automatically. `summarization/text_table_summarizer.py`
  defaults `max_concurrency=1` (serial calls) to stay well under the 30 RPM /
  1,000 OTPM limits — raise it only if you've confirmed your account can take it.
- **Caution combining a low `LLM_MAX_TOKENS` with a "thinking" model**: models
  like `qwen/qwen3.6-27b` spend part of their output budget on internal
  reasoning before writing the final answer. If `LLM_MAX_TOKENS` is too tight,
  the model can exhaust its budget mid-reasoning and return an empty response
  instead of a 429 — if you see that, raise `LLM_MAX_TOKENS` a bit rather than
  assuming the pipeline is broken.

To use DeepSeek (API): create a key at
[platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
(pay-as-you-go, no free tier), set `DEEPSEEK_API_KEY` and `EMBEDDING_PROVIDER`
(e.g. `gemini`) in `.env`, then set `LLM_PROVIDER=deepseek`. Routed through
DeepSeek's OpenAI-compatible endpoint via `langchain-openai`, so no extra
dependency needed. `deepseek-flash` (default) supports vision, needed for
image summaries; `deepseek-v4-pro` is higher quality but text-only. Rate
limiting is dynamic/concurrency-based rather than a harsh fixed RPM like
Groq/Gemini's free tiers, so it's less likely to need a tight `LLM_MAX_TOKENS`
specifically to dodge rate limits (the default still applies for cost reasons).

To use DeepSeek (local, via Ollama): needs **no code changes at all** — it's
just a model name under the existing `ollama` provider. Run
`ollama pull deepseek-r1:1.5b` (fits ~4GB VRAM laptop GPUs; bump to `:7b`/
`:8b`/`:14b` on stronger hardware) and set `OLLAMA_CHAT_MODEL=deepseek-r1:1.5b`
with `LLM_PROVIDER=ollama`. **`deepseek-r1` is text-only** — no vision variant
exists on Ollama at all — so it will fail if it ever needs to summarize an
actual image. Safe as long as `UNSTRUCTURED_STRATEGY=fast` (extracts no
images anyway); use `qwen3-vl`/`llava` as `OLLAMA_CHAT_MODEL` instead if you
switch to `hi_res` and need working image summaries.

To use Ollama: install it from [ollama.com](https://ollama.com), run
`ollama pull qwen3-vl:2b && ollama pull qwen3-embedding:0.6b` (or whichever
models you set), start the Ollama server, then set `LLM_PROVIDER=ollama` in
`.env`. The chat model must support vision since it's also used to summarize
images — plain `qwen3:4b` is text-only and will fail on that step; its
vision-capable sibling `qwen3-vl` family handles both. This is also the option
to reach for when a cloud provider's signup isn't available in your region
(e.g. Alibaba's DashScope console for `qwen`).

Pick the `qwen3-vl` size based on your GPU's VRAM: `:2b` (~2GB) fits laptop/
budget GPUs (e.g. 4GB VRAM); `:4b`/`:8b` need more VRAM but summarize with
noticeably better quality. Check with `ollama ps` while a run is in progress —
if `PROCESSOR` shows a CPU/GPU split instead of 100% GPU, the model doesn't
fully fit in VRAM and inference will be much slower; drop to a smaller size.

**Known issue**: some queries against the local Ollama setup come back with an
empty answer (the retrieval/sources step still works fine). Raising
`OLLAMA_NUM_CTX` from Ollama's 4096 default was tried as a fix and did NOT
resolve it, while roughly tripling generation time -- so it's left at the
4096 default and the real root cause is still being investigated. If you hit
this, a cloud provider (`openai`/`gemini`) is the reliable fallback for now.

To use Qwen: create a free API key at
[bailian.console.alibabacloud.com](https://bailian.console.alibabacloud.com/)
(or the [international console](https://modelstudio.console.alibabacloud.com/)),
set `QWEN_API_KEY` in `.env`, then set `LLM_PROVIDER=qwen`. It's routed through
DashScope's OpenAI-compatible endpoint (`QWEN_BASE_URL`) via `langchain-openai`,
so no extra dependency is needed. As with the other providers, the chat model
must support vision (`qwen-vl-plus` by default).

### Chunking strategies

`CHUNKING_STRATEGY` toggles how document text is chunked, so you can experiment
with how the pipeline behaves under different chunking scenarios without touching code:

| Value        | Behavior                                                              |
|--------------|------------------------------------------------------------------------|
| `by_title`   | Unstructured's native title-based chunking (default)                   |
| `basic`      | Unstructured's native fixed-size chunking                              |
| `recursive`  | LangChain `RecursiveCharacterTextSplitter` over raw extracted text     |
| `sentence`   | Groups whole sentences (NLTK) up to `MAX_CHARACTERS`                   |
| `paragraph`  | Groups whole paragraphs up to `MAX_CHARACTERS`                        |

`by_title`/`basic` are applied natively during partitioning (confirmed generic
across every supported format, not just PDF -- one shared chunking
implementation in Unstructured); the other three are implemented in
`processing/chunking.py` and applied afterwards on the raw unchunked elements.
Set it in `.env`, e.g. `CHUNKING_STRATEGY=sentence`, then rerun `python main.py`
to compare results.

### Cost controls

Four settings specifically to control LLM token spend:

- **`LLM_MAX_TOKENS`** (default `200`) -- caps output length on every chat
  call, every provider, all three call sites (summarization, image
  summarization, final answer). `config/llm_factory.py` maps this one setting
  to the field name each provider's LangChain class actually uses (confirmed
  by reading each package's source): `max_tokens` for `ChatOpenAI`
  (openai/qwen/groq/deepseek), `max_output_tokens` for
  `ChatGoogleGenerativeAI` (gemini), `num_predict` for `ChatOllama`. A cap
  this tight can cause a "thinking" model to exhaust its budget mid-reasoning
  and return an empty response rather than a full answer -- raise it if that
  happens rather than assuming the pipeline is broken.
- **`RETRIEVAL_K`** (default `3`) -- how many documents the multi-vector
  retriever pulls into context per question. This is the setting that scales
  with *ongoing usage*: every question asked pays for this much context,
  unlike summarization, which is a one-time indexing cost. Lower = cheaper
  and faster, but less context for the model to draw on.
- **`SUMMARY_PROVIDER`** (default `openai`, one of `openai`/`gemini` only) --
  text/table/image summarization at indexing time uses this cheap, restricted
  toggle instead of `LLM_PROVIDER`. Summarization is a mechanical task that
  doesn't need your best/most expensive model; `LLM_PROVIDER` continues to
  control only the final answer synthesis the user actually reads.
  `SUMMARY_OPENAI_MODEL` (`gpt-4o-mini`) / `SUMMARY_GEMINI_MODEL`
  (`gemini-3.6-flash`) pick the actual model for each option. Defaults to
  `openai`, not `gemini`: Gemini's free-tier flash models can have a
  severely restrictive RPM quota (confirmed live: `gemini-3.6-flash`'s free
  tier is 5 requests/minute), which trips immediately past a handful of
  chunks -- fine if you're on a paid Gemini tier or summarizing very few
  chunks, but `openai`/`gpt-4o-mini` is the safer default for a real document.
- **Summary caching** (`summarization/cache.py`, always on, no setting) --
  every text/table/image summary is cached in Redis, keyed by a hash of its
  content plus which `SUMMARY_PROVIDER`/model produced it. Restarting the
  pipeline on unchanged documents reuses cached summaries instead of
  re-summarizing (and re-paying for) identical content -- you'll see
  `"N summaries reused from cache"` printed when this kicks in. Switching
  `SUMMARY_PROVIDER` (or its model) naturally misses the cache and
  re-summarizes with the new one, rather than silently reusing an older
  provider's output.

## Multi-format ingestion

`data_ingestion/document_loader.py` uses Unstructured's auto-detecting
`partition()` (via LangChain's generic `UnstructuredFileLoader`, dispatching by
file extension) instead of a PDF-specific loader, so `data/` can hold any mix
of: **PDF, DOCX, PPTX, TXT, HTML, or standalone images**. Every discovered file
is partitioned into its own `figures_dir/<filename>/` subdirectory (preventing
image-filename collisions between files) and all of their text/table elements
are combined into one knowledge base.

**Known limitation**: only PDFs (and standalone image files) contribute
*extracted* images to the multimodal/vision index. Unstructured has no
equivalent image-extraction mechanism for DOCX (needs custom Python-level
picture-partitioner registration), PPTX (no output-directory hook), or
HTML/TXT (no image concept at all) -- confirmed by reading Unstructured's own
source, not assumed. DOCX/PPTX/TXT/HTML documents still contribute text and
tables normally; they just won't produce chart/figure images for the vision
summarization step.

Table-structure inference and page-number metadata reliability also vary by
format: page numbers are solid for PDF/PPTX, best-effort for DOCX (only set
when the source file has explicit hard page-breaks), and never set for
TXT/HTML -- this is why citations (below) sometimes show just a filename with
no page number.

## Citations

Every answer includes a **citations list assembled deterministically in code**
from retrieved-chunk metadata (`processing/citations.py`,
`processing/element_splitter.py`) -- not left to the LLM to remember to cite,
since the free-tier models this project has been tested against aren't
reliable enough for that. Each retrieved text/table chunk is stored with its
source filename + page number (when available) and shown to the model
prefixed with a `[Source: filename, p.N]` tag, so it can reference sources
naturally in its answer -- but the guaranteed `citations` field on the chain's
output (and the CLI's "Citations:" section, and the API's `citations` response
field) comes from the tracked metadata directly, independent of whether the
model actually mentioned it.

## Interactive API (FastAPI + Swagger)

For a browser-based Q&A experience instead of the CLI, run:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open **http://127.0.0.1:8000/docs** for the interactive Swagger UI. It
builds the RAG pipeline once at startup (indexing is a one-time cost, not
per-request) against whatever's in `data/`, then exposes:

- `POST /query` — `{"question": "..."}` → `{"answer": "...", "citations": [...]}`. Use Swagger's "Try it out" button to ask questions directly in the browser.
- `GET /health` — `{"status": "ok"|"starting", "documents_indexed": N}`.

Since startup runs the full ingestion pipeline, the first request may need to
wait a bit after launching `uvicorn` (watch the terminal log, or poll `/health`
until `status` is `"ok"`) — how long depends on your `LLM_PROVIDER`/how many
documents are in `data/`.

**Don't add `--reload` when running under WSL from a Windows-mounted drive**
(e.g. `/mnt/d/...`, which this project's path is). `--reload`'s file-watcher
recursively scans the whole project directory -- including `.venv`'s tens of
thousands of dependency files -- on every check, and over WSL's slower
9p-mounted-NTFS filesystem access to a Windows drive this can peg CPU/IO badly
enough that the server prints "Application startup complete" but doesn't
actually respond to requests (curl/browser hangs or gets connection-refused,
even though `ps`/`ss` show the process alive and listening). Only use
`--reload` if you're actively editing `api/`'s code and want auto-restart on
save, and even then scope it with `--reload-dir api` rather than watching the
whole project.

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
python3 -m venv .venv - this only when new .venv to create
source .venv/bin/activate
pip install -r requirements.txt
python main.py
