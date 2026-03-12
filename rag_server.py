"""
RAG MCP Server

Exposes four tools over the MCP stdio transport:
  - ingest_document  : index a file into the vector store
  - query_rag        : retrieve chunks and generate a cited answer
  - list_documents   : catalogue of indexed documents
  - delete_document  : remove a document and all its chunks

The server layer is intentionally thin — zero business logic lives here.
All RAG orchestration is in RAGEngine, keeping the MCP surface testable
independently of the protocol.
"""

import json
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from rag.engine import RAGEngine

load_dotenv()

mcp = FastMCP(name="RAG Server")

# Module-level singleton is correct for stdio transport (one process = one client).
# For SSE transport with concurrent clients, move this into a per-session context.
_engine = RAGEngine(
    persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./.chromadb"),
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)


@mcp.tool()
async def ingest_document(
    file_path: str = Field(
        description="Absolute or relative path to the file (.pdf, .txt, .md)"
    ),
    *,
    ctx: Context,
) -> str:
    """
    Index a document into the RAG knowledge base.

    Chunks the file, generates embeddings, and persists to the vector store.
    Idempotent: re-ingesting the same file updates the existing entry.
    """
    await ctx.info(f"Starting ingestion: {file_path}")
    await ctx.report_progress(0, 100)

    document = await _engine.ingest_document(file_path)

    await ctx.report_progress(100, 100)

    if document.status.value == "failed":
        return json.dumps({
            "status": "error",
            "document_id": document.document_id,
            "error": document.error_message,
        })

    return json.dumps({
        "status": "success",
        "document_id": document.document_id,
        "title": document.title,
        "chunk_count": document.chunk_count,
        "ingested_at": document.ingested_at.isoformat(),
    })


@mcp.tool()
async def query_rag(
    question: str = Field(description="The question to answer using indexed documents"),
    n_results: int = Field(
        default=5,
        description="Number of chunks to retrieve (1–20). More chunks = more context.",
        ge=1,
        le=20,
    ),
    *,
    ctx: Context,
) -> str:
    """
    Answer a question using the RAG pipeline.

    Semantically retrieves the most relevant chunks, then uses Claude to
    synthesise a cited answer grounded in those sources.
    """
    await ctx.info(f"Retrieving context for: {question}")
    await ctx.report_progress(25, 100)

    rag_response = await _engine.query(question, n_results=n_results)

    await ctx.report_progress(100, 100)

    return json.dumps({
        "answer": rag_response.answer,
        "model": rag_response.model,
        "retrieved_chunks": rag_response.retrieved_chunk_count,
        "sources": [
            {
                "document_title": rc.document_title,
                "relevance_score": round(rc.score, 4),
                "excerpt": (
                    rc.chunk.text[:300] + "..."
                    if len(rc.chunk.text) > 300
                    else rc.chunk.text
                ),
            }
            for rc in rag_response.sources
        ],
    })


@mcp.tool()
async def list_documents(ctx: Context) -> str:
    """
    List all documents currently indexed in the RAG knowledge base.

    Returns metadata (title, chunk count, ingestion time) but not chunk content.
    """
    documents = _engine.list_documents()

    return json.dumps({
        "document_count": len(documents),
        "documents": [
            {
                "document_id": doc.document_id,
                "title": doc.title,
                "file_path": doc.file_path,
                "mime_type": doc.mime_type,
                "chunk_count": doc.chunk_count,
                "status": doc.status.value,
                "ingested_at": doc.ingested_at.isoformat() if doc.ingested_at else None,
            }
            for doc in documents
        ],
    })


@mcp.tool()
async def delete_document(
    document_id: str = Field(
        description="The document_id returned by ingest_document"
    ),
    *,
    ctx: Context,
) -> str:
    """
    Remove a document and all its chunks from the RAG index.

    This operation is permanent. Re-ingest the file to restore it.
    """
    await ctx.info(f"Deleting document: {document_id}")
    success = _engine.delete_document(document_id)

    return json.dumps({
        "status": "deleted" if success else "not_found",
        "document_id": document_id,
    })


if __name__ == "__main__":
    mcp.run(transport="stdio")
