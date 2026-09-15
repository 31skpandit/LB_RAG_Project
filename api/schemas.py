"""Pydantic request/response models for the RAG query API."""
from typing import List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask against the indexed documents.")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The synthesized answer.")
    citations: List[str] = Field(
        default_factory=list,
        description=(
            "Deterministically assembled source citations (e.g. 'report.pdf, p.3'), "
            "built from retrieved-chunk metadata -- always present when the answer "
            "used cited text/table context, independent of whether the model itself "
            "remembered to mention sources."
        ),
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="'starting' while the pipeline is still indexing, 'ok' once ready.")
    documents_indexed: int = Field(..., description="Number of documents discovered in data/ at startup.")
