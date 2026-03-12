"""Tests for rag/engine.py — RAGEngine orchestration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.engine import RAGEngine
from rag.models import Chunk, DocumentStatus, RAGResponse, RetrievedChunk


@pytest.fixture
def engine(tmp_path, mock_embed_fn):
    """Real PersistentClient in a temp dir + mock embeddings + mocked Gemini client."""
    with patch("rag.engine.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        e = RAGEngine(
            persist_directory=str(tmp_path / "chroma"),
            model="test-model",
            embedding_function=mock_embed_fn,
        )
    # _gemini is already assigned before patch context exits
    return e


@pytest.fixture
async def engine_with_doc(engine, tmp_text_file):
    await engine.ingest_document(tmp_text_file)
    return engine


def _mock_gemini_response(text: str):
    """Build a mock Gemini response returning the given text."""
    mock_resp = MagicMock()
    mock_resp.text = text
    return mock_resp


def _setup_gemini_mock(engine, text: str):
    """Configure the engine's Gemini mock to return the given text."""
    engine._gemini.aio.models.generate_content = AsyncMock(
        return_value=_mock_gemini_response(text)
    )


class TestIngest:
    async def test_ingest_txt(self, engine, tmp_text_file):
        doc = await engine.ingest_document(tmp_text_file)
        assert doc.status == DocumentStatus.PROCESSED
        assert doc.chunk_count > 0
        assert doc.ingested_at is not None

    async def test_ingest_md(self, engine, tmp_md_file):
        doc = await engine.ingest_document(tmp_md_file)
        assert doc.status == DocumentStatus.PROCESSED
        assert doc.chunk_count > 0

    async def test_ingest_bad_pdf_returns_failed(self, engine, tmp_path):
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"not a valid pdf")
        doc = await engine.ingest_document(str(f))
        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message is not None

    async def test_ingest_is_idempotent(self, engine, tmp_text_file):
        """Re-ingesting same file must not create duplicate documents."""
        doc1 = await engine.ingest_document(tmp_text_file)
        doc2 = await engine.ingest_document(tmp_text_file)
        assert doc1.document_id == doc2.document_id
        assert len(engine.list_documents()) == 1

    async def test_ingest_stores_in_catalogue(self, engine, tmp_text_file, tmp_md_file):
        await engine.ingest_document(tmp_text_file)
        await engine.ingest_document(tmp_md_file)
        docs = engine.list_documents()
        assert len(docs) == 2


class TestRetrieve:
    def test_retrieve_empty_collection_returns_empty(self, engine):
        assert engine._retrieve("anything", n_results=5) == []

    async def test_retrieve_returns_scored_chunks(self, engine, tmp_text_file):
        await engine.ingest_document(tmp_text_file)
        results = engine._retrieve("quick brown fox", n_results=3)
        assert len(results) > 0
        for rc in results:
            assert 0.0 <= rc.score <= 1.0
            assert rc.document_title
            assert rc.chunk.text

    async def test_retrieve_respects_n_results_cap(self, engine, tmp_text_file):
        await engine.ingest_document(tmp_text_file)
        results = engine._retrieve("text", n_results=2)
        assert len(results) <= 2

    async def test_retrieve_n_results_capped_at_collection_size(self, engine, tmp_path):
        """n_results larger than total chunks should not raise."""
        f = tmp_path / "small.txt"
        # Write enough text to clear min_chunk_size=100
        f.write_text("The quick brown fox jumps over the lazy dog. " * 10)
        await engine.ingest_document(str(f))
        results = engine._retrieve("fox", n_results=1000)
        assert len(results) >= 1


class TestQuery:
    async def test_query_no_documents(self, engine):
        resp = await engine.query("What is this?")
        assert isinstance(resp, RAGResponse)
        assert "No documents" in resp.answer
        assert resp.retrieved_chunk_count == 0
        assert resp.sources == []

    async def test_query_with_documents_calls_llm(self, engine, tmp_text_file):
        await engine.ingest_document(tmp_text_file)
        _setup_gemini_mock(engine, "The answer is 42.")
        resp = await engine.query("brown fox", n_results=3)
        assert resp.answer == "The answer is 42."
        assert resp.retrieved_chunk_count > 0
        assert resp.model == "test-model"

    async def test_query_sources_have_titles(self, engine, tmp_text_file):
        await engine.ingest_document(tmp_text_file)
        _setup_gemini_mock(engine, "answer")
        resp = await engine.query("fox", n_results=2)
        for src in resp.sources:
            assert src.document_title

    async def test_query_respects_n_results(self, engine, tmp_text_file):
        await engine.ingest_document(tmp_text_file)
        _setup_gemini_mock(engine, "answer")
        resp = await engine.query("fox", n_results=2)
        assert resp.retrieved_chunk_count <= 2

    async def test_query_response_fields(self, engine, tmp_text_file):
        await engine.ingest_document(tmp_text_file)
        _setup_gemini_mock(engine, "detailed answer")
        resp = await engine.query("my question")
        assert resp.query == "my question"
        assert resp.model == "test-model"


class TestCatalogue:
    def test_list_empty(self, engine):
        assert engine.list_documents() == []

    async def test_list_after_ingest(self, engine, tmp_text_file):
        await engine.ingest_document(tmp_text_file)
        docs = engine.list_documents()
        assert len(docs) == 1
        assert docs[0].status == DocumentStatus.PROCESSED
        assert docs[0].chunk_count > 0
        assert docs[0].ingested_at is not None

    async def test_delete_existing_document(self, engine, tmp_text_file):
        doc = await engine.ingest_document(tmp_text_file)
        assert len(engine.list_documents()) == 1
        result = engine.delete_document(doc.document_id)
        assert result is True
        assert engine.list_documents() == []

    async def test_delete_removes_chunks(self, engine, tmp_text_file):
        """After deletion, chunks should no longer be retrievable."""
        doc = await engine.ingest_document(tmp_text_file)
        assert engine._chunks_col.count() > 0
        engine.delete_document(doc.document_id)
        assert engine._chunks_col.count() == 0

    def test_delete_nonexistent_does_not_raise(self, engine):
        """ChromaDB silently ignores deletes of missing IDs; method should not crash."""
        # ChromaDB's delete() does not raise for nonexistent IDs, so the method
        # returns True (docs_col.delete succeeded). This is acceptable behaviour.
        result = engine.delete_document("does_not_exist")
        assert isinstance(result, bool)


class TestHelpers:
    def test_build_context_block_single_source(self, engine):
        chunk = Chunk(chunk_id="c", source_document_id="d", text="Some content.", chunk_index=0)
        rc = RetrievedChunk(chunk=chunk, score=0.87, document_title="My Document")
        ctx = engine._build_context_block([rc])
        assert "[Source 1]" in ctx
        assert "My Document" in ctx
        assert "Some content." in ctx
        assert "0.87" in ctx

    def test_build_context_block_multiple_sources(self, engine):
        chunks = [
            RetrievedChunk(
                chunk=Chunk(chunk_id=f"c{i}", source_document_id="d", text=f"Text {i}.", chunk_index=i),
                score=0.9 - i * 0.1,
                document_title=f"Doc {i}",
            )
            for i in range(3)
        ]
        ctx = engine._build_context_block(chunks)
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx
        assert "[Source 3]" in ctx
        assert "---" in ctx  # separator between sources

    def test_build_prompt_includes_query_and_context(self, engine):
        prompt = engine._build_prompt("What is RAG?", "Here is some context.")
        assert "What is RAG?" in prompt
        assert "Here is some context." in prompt

    def test_build_prompt_has_citation_instruction(self, engine):
        prompt = engine._build_prompt("Q", "C")
        assert "Source" in prompt

    async def test_get_document_title_missing(self, engine):
        title = engine._get_document_title("nonexistent_id")
        assert title == "Unknown Document"

    async def test_get_document_title_existing(self, engine, tmp_text_file):
        doc = await engine.ingest_document(tmp_text_file)
        title = engine._get_document_title(doc.document_id)
        assert title == doc.title
