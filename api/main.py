"""FastAPI service exposing the multimodal RAG pipeline for interactive Q&A.

Run with (from the repo root):
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI -- use
"Try it out" on POST /query to ask questions against whatever documents are
in data/, with citations included in every response.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, QueryRequest, QueryResponse
from data_ingestion import discover_documents
from main import build_pipeline

# Holds the built RAG chain + startup stats. Populated once by `lifespan`
# below; indexing is a one-time cost paid at startup, not per request.
_state = {"chain": None, "documents_indexed": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["documents_indexed"] = len(discover_documents())
    _state["chain"] = build_pipeline()
    yield
    _state.clear()


app = FastAPI(
    title="Multimodal RAG API",
    description=(
        "Ask questions against every document in data/ (PDF, DOCX, PPTX, TXT, "
        "HTML, images). Answers always include a deterministically assembled "
        "citations list."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state["chain"] is not None else "starting",
        documents_indexed=_state["documents_indexed"],
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    chain = _state["chain"]
    if chain is None:
        raise HTTPException(status_code=503, detail="Pipeline is still starting up -- try again shortly.")

    response = chain.invoke({"input": request.question})
    return QueryResponse(answer=response["answer"], citations=response.get("citations", []))
