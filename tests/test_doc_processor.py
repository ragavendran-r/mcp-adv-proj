"""Tests for rag/doc_processor.py — chunking and document loading."""

import pytest
from pathlib import Path

from rag.doc_processor import ChunkingConfig, DocumentProcessor
from rag.models import DocumentStatus


class TestChunkingConfig:
    def test_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.min_chunk_size == 100

    def test_custom_values(self):
        cfg = ChunkingConfig(chunk_size=500, chunk_overlap=50, min_chunk_size=30)
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 50
        assert cfg.min_chunk_size == 30


class TestComputeDocumentId:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello world")
        id1 = DocumentProcessor.compute_document_id(str(f))
        id2 = DocumentProcessor.compute_document_id(str(f))
        assert id1 == id2

    def test_length(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content")
        doc_id = DocumentProcessor.compute_document_id(str(f))
        assert len(doc_id) == 16

    def test_different_content_gives_different_id(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content a")
        f2.write_text("content b")
        assert DocumentProcessor.compute_document_id(str(f1)) != DocumentProcessor.compute_document_id(str(f2))


class TestDetectMimeType:
    def setup_method(self):
        self.proc = DocumentProcessor()

    def test_txt(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.touch()
        assert self.proc.detect_mime_type(str(f)) == "text/plain"

    def test_md(self, tmp_path):
        f = tmp_path / "doc.md"
        f.touch()
        assert self.proc.detect_mime_type(str(f)) == "text/markdown"

    def test_pdf(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.touch()
        assert self.proc.detect_mime_type(str(f)) == "application/pdf"

    def test_uppercase_extension(self, tmp_path):
        f = tmp_path / "DOC.TXT"
        f.touch()
        assert self.proc.detect_mime_type(str(f)) == "text/plain"

    def test_unsupported_raises(self, tmp_path):
        f = tmp_path / "data.csv"
        f.touch()
        with pytest.raises(ValueError, match="Unsupported file type"):
            self.proc.detect_mime_type(str(f))


class TestCreateDocument:
    def test_creates_document_from_txt(self, tmp_text_file):
        proc = DocumentProcessor()
        doc = proc.create_document(tmp_text_file)
        assert doc.file_path == str(Path(tmp_text_file).resolve())
        assert doc.mime_type == "text/plain"
        assert doc.title == "Sample"
        assert len(doc.document_id) == 16

    def test_creates_document_from_md(self, tmp_md_file):
        proc = DocumentProcessor()
        doc = proc.create_document(tmp_md_file)
        assert doc.mime_type == "text/markdown"
        assert doc.title == "Sample"

    def test_missing_file_raises(self, tmp_path):
        proc = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            proc.create_document(str(tmp_path / "nonexistent.txt"))

    def test_title_normalises_underscores(self, tmp_path):
        f = tmp_path / "my_long_file_name.txt"
        f.write_text("content")
        proc = DocumentProcessor()
        doc = proc.create_document(str(f))
        assert doc.title == "My Long File Name"

    def test_title_normalises_hyphens(self, tmp_path):
        f = tmp_path / "rag-overview.md"
        f.write_text("# content")
        proc = DocumentProcessor()
        doc = proc.create_document(str(f))
        assert doc.title == "Rag Overview"


class TestProcess:
    def test_process_txt_succeeds(self, tmp_text_file):
        proc = DocumentProcessor()
        doc, chunks = proc.process(tmp_text_file)
        assert doc.status == DocumentStatus.PROCESSED
        assert doc.chunk_count > 0
        assert len(chunks) == doc.chunk_count
        assert doc.ingested_at is not None

    def test_process_md_succeeds(self, tmp_md_file):
        proc = DocumentProcessor()
        doc, chunks = proc.process(tmp_md_file)
        assert doc.status == DocumentStatus.PROCESSED
        assert len(chunks) > 0

    def test_chunk_ids_are_unique(self, tmp_text_file):
        proc = DocumentProcessor()
        _, chunks = proc.process(tmp_text_file)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_source_document_id(self, tmp_text_file):
        proc = DocumentProcessor()
        doc, chunks = proc.process(tmp_text_file)
        for chunk in chunks:
            assert chunk.source_document_id == doc.document_id

    def test_chunk_indices_are_sequential(self, tmp_text_file):
        proc = DocumentProcessor()
        _, chunks = proc.process(tmp_text_file)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_metadata_fields(self, tmp_text_file):
        proc = DocumentProcessor()
        doc, chunks = proc.process(tmp_text_file)
        for c in chunks:
            assert "document_id" in c.metadata
            assert "chunk_index" in c.metadata
            assert "start_char" in c.metadata
            assert "end_char" in c.metadata
            assert c.metadata["document_id"] == doc.document_id

    def test_process_bad_pdf_returns_failed(self, tmp_path):
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"this is not a valid pdf binary")
        proc = DocumentProcessor()
        doc, chunks = proc.process(str(bad_pdf))
        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message is not None
        assert chunks == []

    def test_process_missing_file_raises(self, tmp_path):
        proc = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            proc.process(str(tmp_path / "nope.txt"))


class TestChunkTextSliding:
    def test_overlap_start_positions(self, tmp_path):
        """Second chunk should start at chunk_size - chunk_overlap."""
        cfg = ChunkingConfig(chunk_size=200, chunk_overlap=50, min_chunk_size=10)
        proc = DocumentProcessor(chunking_config=cfg)
        f = tmp_path / "big.txt"
        f.write_text("A" * 600)
        _, chunks = proc.process(str(f))
        assert len(chunks) >= 3
        # Chunk 0: start=0, end=200
        # Chunk 1: start=150 (200-50), end=350
        assert chunks[0].metadata["start_char"] == 0
        assert chunks[1].metadata["start_char"] == 150

    def test_min_chunk_size_filters_tiny_trailing_chunks(self, tmp_path):
        """Trailing text below min_chunk_size should be dropped."""
        cfg = ChunkingConfig(chunk_size=100, chunk_overlap=0, min_chunk_size=50)
        proc = DocumentProcessor(chunking_config=cfg)
        # 100 chars + 20 chars trailing (below min=50)
        f = tmp_path / "short.txt"
        f.write_text("X" * 120)
        _, chunks = proc.process(str(f))
        # The 20-char trailing chunk should be dropped
        for c in chunks:
            assert len(c.text.strip()) >= cfg.min_chunk_size

    def test_single_chunk_for_short_text(self, tmp_path):
        cfg = ChunkingConfig(chunk_size=1000, chunk_overlap=100, min_chunk_size=10)
        proc = DocumentProcessor(chunking_config=cfg)
        f = tmp_path / "short.txt"
        f.write_text("Short text only.")
        _, chunks = proc.process(str(f))
        assert len(chunks) == 1
