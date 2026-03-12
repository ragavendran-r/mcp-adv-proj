"""Tests for rag/models.py — data contracts."""

from datetime import datetime, timezone

from rag.models import (
    Chunk,
    Document,
    DocumentStatus,
    RAGResponse,
    RetrievedChunk,
    SupportedMimeType,
)


class TestDocumentStatus:
    def test_values(self):
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.PROCESSED.value == "processed"
        assert DocumentStatus.FAILED.value == "failed"

    def test_from_value(self):
        assert DocumentStatus("pending") == DocumentStatus.PENDING
        assert DocumentStatus("processed") == DocumentStatus.PROCESSED
        assert DocumentStatus("failed") == DocumentStatus.FAILED


class TestSupportedMimeType:
    def test_values(self):
        assert SupportedMimeType.PDF.value == "application/pdf"
        assert SupportedMimeType.TEXT.value == "text/plain"
        assert SupportedMimeType.MARKDOWN.value == "text/markdown"


class TestChunk:
    def test_minimal_creation(self):
        chunk = Chunk(
            chunk_id="c1",
            source_document_id="d1",
            text="Hello world",
            chunk_index=0,
        )
        assert chunk.chunk_id == "c1"
        assert chunk.source_document_id == "d1"
        assert chunk.text == "Hello world"
        assert chunk.chunk_index == 0
        assert chunk.metadata == {}

    def test_with_metadata(self):
        chunk = Chunk(
            chunk_id="c2",
            source_document_id="d2",
            text="text",
            chunk_index=1,
            metadata={"start_char": 0, "end_char": 100},
        )
        assert chunk.metadata["start_char"] == 0
        assert chunk.metadata["end_char"] == 100

    def test_metadata_default_is_independent(self):
        """Each Chunk should get its own default dict, not a shared one."""
        c1 = Chunk(chunk_id="a", source_document_id="d", text="t", chunk_index=0)
        c2 = Chunk(chunk_id="b", source_document_id="d", text="t", chunk_index=1)
        c1.metadata["key"] = "value"
        assert "key" not in c2.metadata


class TestDocument:
    def test_defaults(self):
        doc = Document(
            document_id="doc1",
            file_path="/path/file.txt",
            title="My Doc",
            mime_type="text/plain",
        )
        assert doc.status == DocumentStatus.PENDING
        assert doc.chunk_count == 0
        assert doc.ingested_at is None
        assert doc.error_message is None

    def test_full_construction(self):
        now = datetime.now(timezone.utc)
        doc = Document(
            document_id="d1",
            file_path="/f.txt",
            title="T",
            mime_type="text/plain",
            status=DocumentStatus.PROCESSED,
            chunk_count=7,
            ingested_at=now,
            error_message=None,
        )
        assert doc.status == DocumentStatus.PROCESSED
        assert doc.chunk_count == 7
        assert doc.ingested_at == now

    def test_failed_status_with_error_message(self):
        doc = Document(
            document_id="d2",
            file_path="/bad.pdf",
            title="Bad",
            mime_type="application/pdf",
            status=DocumentStatus.FAILED,
            error_message="Could not parse PDF",
        )
        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message == "Could not parse PDF"


class TestRetrievedChunk:
    def test_construction(self):
        chunk = Chunk(chunk_id="c", source_document_id="d", text="text", chunk_index=0)
        rc = RetrievedChunk(chunk=chunk, score=0.92, document_title="My Title")
        assert rc.score == 0.92
        assert rc.document_title == "My Title"
        assert rc.chunk is chunk

    def test_score_range(self):
        chunk = Chunk(chunk_id="c", source_document_id="d", text="t", chunk_index=0)
        for score in [0.0, 0.5, 1.0]:
            rc = RetrievedChunk(chunk=chunk, score=score, document_title="Doc")
            assert rc.score == score


class TestRAGResponse:
    def test_construction(self):
        chunk = Chunk(chunk_id="c", source_document_id="d", text="t", chunk_index=0)
        rc = RetrievedChunk(chunk=chunk, score=0.9, document_title="Doc")
        resp = RAGResponse(
            answer="The answer is 42.",
            sources=[rc],
            query="What is the answer?",
            model="claude-sonnet-4-6",
            retrieved_chunk_count=1,
        )
        assert resp.answer == "The answer is 42."
        assert len(resp.sources) == 1
        assert resp.retrieved_chunk_count == 1
        assert resp.model == "claude-sonnet-4-6"
        assert resp.query == "What is the answer?"

    def test_empty_sources(self):
        resp = RAGResponse(
            answer="No documents indexed.",
            sources=[],
            query="anything",
            model="test",
            retrieved_chunk_count=0,
        )
        assert resp.sources == []
        assert resp.retrieved_chunk_count == 0
