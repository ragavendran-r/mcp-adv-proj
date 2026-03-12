"""
RAG package: Retrieval-Augmented Generation pipeline components.

Import from here rather than submodules directly — this decouples callers
from internal restructuring.
"""

from rag.engine import RAGEngine
from rag.doc_processor import ChunkingConfig, DocumentProcessor
from rag.models import (
    Chunk,
    Document,
    DocumentStatus,
    RAGResponse,
    RetrievedChunk,
    SupportedMimeType,
)

__all__ = [
    "RAGEngine",
    "DocumentProcessor",
    "ChunkingConfig",
    "Chunk",
    "Document",
    "DocumentStatus",
    "RAGResponse",
    "RetrievedChunk",
    "SupportedMimeType",
]
