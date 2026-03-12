"""Tests for rag_server.py — MCP tool wrappers.

conftest.py sets CHROMA_PERSIST_DIR before this module is imported,
so the module-level RAGEngine construction uses a temp directory.
Then each test replaces _engine with a mock via monkeypatch.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import rag_server
from rag.models import Document, DocumentStatus, RAGResponse, RetrievedChunk, Chunk


class MockContext:
    """Minimal MCP Context mock that satisfies the server's ctx usage."""
    async def info(self, msg: str) -> None:
        pass

    async def report_progress(self, current: int, total: int) -> None:
        pass


@pytest.fixture
def ctx():
    return MockContext()


@pytest.fixture
def mock_engine(monkeypatch):
    """Replace the module-level _engine with a MagicMock for each test."""
    engine = MagicMock()
    monkeypatch.setattr(rag_server, "_engine", engine)
    return engine


def _make_document(status=DocumentStatus.PROCESSED, chunk_count=5, error=None):
    now = datetime.now(timezone.utc)
    return Document(
        document_id="abc123",
        file_path="/some/file.txt",
        title="Sample Doc",
        mime_type="text/plain",
        status=status,
        chunk_count=chunk_count,
        ingested_at=now if status == DocumentStatus.PROCESSED else None,
        error_message=error,
    )


class TestIngestDocument:
    async def test_success(self, mock_engine, ctx):
        doc = _make_document()
        mock_engine.ingest_document = AsyncMock(return_value=doc)

        result = await rag_server.ingest_document(file_path="/some/file.txt", ctx=ctx)
        data = json.loads(result)

        assert data["status"] == "success"
        assert data["document_id"] == "abc123"
        assert data["title"] == "Sample Doc"
        assert data["chunk_count"] == 5

    async def test_failure(self, mock_engine, ctx):
        doc = _make_document(status=DocumentStatus.FAILED, error="Parse error")
        mock_engine.ingest_document = AsyncMock(return_value=doc)

        result = await rag_server.ingest_document(file_path="/bad/file.pdf", ctx=ctx)
        data = json.loads(result)

        assert data["status"] == "error"
        assert data["error"] == "Parse error"

    async def test_calls_engine_with_file_path(self, mock_engine, ctx):
        doc = _make_document()
        mock_engine.ingest_document = AsyncMock(return_value=doc)

        await rag_server.ingest_document(file_path="/my/doc.txt", ctx=ctx)
        mock_engine.ingest_document.assert_awaited_once_with("/my/doc.txt")


class TestQueryRag:
    def _make_rag_response(self, answer="The answer."):
        chunk = Chunk(chunk_id="c1", source_document_id="d1", text="context text", chunk_index=0)
        sources = [RetrievedChunk(chunk=chunk, score=0.91, document_title="My Doc")]
        return RAGResponse(
            answer=answer,
            sources=sources,
            query="test question",
            model="test-model",
            retrieved_chunk_count=1,
        )

    async def test_success(self, mock_engine, ctx):
        mock_engine.query = AsyncMock(return_value=self._make_rag_response())

        result = await rag_server.query_rag(question="What is RAG?", ctx=ctx)
        data = json.loads(result)

        assert data["answer"] == "The answer."
        assert data["retrieved_chunks"] == 1
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_title"] == "My Doc"
        assert data["sources"][0]["relevance_score"] == 0.91

    async def test_source_excerpt_truncated_at_300(self, mock_engine, ctx):
        chunk = Chunk(
            chunk_id="c",
            source_document_id="d",
            text="X" * 500,
            chunk_index=0,
        )
        sources = [RetrievedChunk(chunk=chunk, score=0.8, document_title="Doc")]
        resp = RAGResponse(
            answer="a",
            sources=sources,
            query="q",
            model="m",
            retrieved_chunk_count=1,
        )
        mock_engine.query = AsyncMock(return_value=resp)
        result = await rag_server.query_rag(question="q", ctx=ctx)
        data = json.loads(result)
        assert data["sources"][0]["excerpt"].endswith("...")

    async def test_short_excerpt_not_truncated(self, mock_engine, ctx):
        chunk = Chunk(chunk_id="c", source_document_id="d", text="Short.", chunk_index=0)
        sources = [RetrievedChunk(chunk=chunk, score=0.8, document_title="Doc")]
        resp = RAGResponse(
            answer="a",
            sources=sources,
            query="q",
            model="m",
            retrieved_chunk_count=1,
        )
        mock_engine.query = AsyncMock(return_value=resp)
        result = await rag_server.query_rag(question="q", ctx=ctx)
        data = json.loads(result)
        assert data["sources"][0]["excerpt"] == "Short."

    async def test_uses_n_results(self, mock_engine, ctx):
        mock_engine.query = AsyncMock(return_value=self._make_rag_response())
        await rag_server.query_rag(question="q", n_results=10, ctx=ctx)
        mock_engine.query.assert_awaited_once_with("q", n_results=10)


class TestListDocuments:
    async def test_returns_document_list(self, mock_engine, ctx):
        mock_engine.list_documents.return_value = [_make_document(), _make_document()]

        result = await rag_server.list_documents(ctx=ctx)
        data = json.loads(result)

        assert data["document_count"] == 2
        assert len(data["documents"]) == 2

    async def test_empty_list(self, mock_engine, ctx):
        mock_engine.list_documents.return_value = []

        result = await rag_server.list_documents(ctx=ctx)
        data = json.loads(result)

        assert data["document_count"] == 0
        assert data["documents"] == []

    async def test_document_fields(self, mock_engine, ctx):
        mock_engine.list_documents.return_value = [_make_document()]

        result = await rag_server.list_documents(ctx=ctx)
        data = json.loads(result)
        doc = data["documents"][0]

        assert "document_id" in doc
        assert "title" in doc
        assert "chunk_count" in doc
        assert "mime_type" in doc
        assert "status" in doc


class TestDeleteDocument:
    async def test_delete_existing(self, mock_engine, ctx):
        mock_engine.delete_document.return_value = True

        result = await rag_server.delete_document(document_id="abc123", ctx=ctx)
        data = json.loads(result)

        assert data["status"] == "deleted"
        assert data["document_id"] == "abc123"

    async def test_delete_nonexistent(self, mock_engine, ctx):
        mock_engine.delete_document.return_value = False

        result = await rag_server.delete_document(document_id="unknown", ctx=ctx)
        data = json.loads(result)

        assert data["status"] == "not_found"
