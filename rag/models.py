from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DocumentStatus(Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


class SupportedMimeType(Enum):
    PDF = "application/pdf"
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"


@dataclass
class Chunk:
    """
    A single unit of indexed text.

    chunk_index and source_document_id together enable re-assembly of
    original documents and proper citation generation.
    """
    chunk_id: str
    source_document_id: str
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    """
    Represents a document as it moves through the ingestion pipeline.

    Separates raw source (file_path) from derived state (status, chunk_count),
    enabling idempotent re-ingestion by document_id.
    """
    document_id: str
    file_path: str
    title: str
    mime_type: str
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    ingested_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class RetrievedChunk:
    """A chunk returned from vector search, augmented with its similarity score."""
    chunk: Chunk
    score: float  # cosine similarity [0.0, 1.0]
    document_title: str


@dataclass
class RAGResponse:
    """
    The complete response from a RAG query.

    Keeping sources separate from the answer lets the client decide how
    to render citations — the server makes no UI decisions.
    """
    answer: str
    sources: list[RetrievedChunk]
    query: str
    model: str
    retrieved_chunk_count: int
