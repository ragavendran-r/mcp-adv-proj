import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pypdf

from rag.models import Chunk, Document, DocumentStatus, SupportedMimeType


class ChunkingConfig:
    """
    Encapsulates chunking strategy in a single injectable object.

    Centralising these constants here means a future change to chunking
    (e.g. switching to semantic/paragraph-aware splitting) only touches
    this class, not every caller.
    """
    def __init__(
        self,
        chunk_size: int = 1000,    # characters
        chunk_overlap: int = 200,   # overlap prevents context loss at boundaries
        min_chunk_size: int = 100,  # discard near-empty trailing chunks
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size


class DocumentProcessor:
    """
    Stateless document-to-chunks pipeline.

    Each public method is a pure transformation: given a file, return chunks.
    No side effects beyond reading the source file.
    """

    _MIME_EXTRACTORS: dict[str, str] = {
        SupportedMimeType.PDF.value: "_extract_pdf",
        SupportedMimeType.TEXT.value: "_extract_text",
        SupportedMimeType.MARKDOWN.value: "_extract_text",
    }

    _EXTENSION_TO_MIME: dict[str, str] = {
        ".pdf": SupportedMimeType.PDF.value,
        ".txt": SupportedMimeType.TEXT.value,
        ".md": SupportedMimeType.MARKDOWN.value,
    }

    def __init__(self, chunking_config: ChunkingConfig | None = None):
        self.config = chunking_config or ChunkingConfig()

    @staticmethod
    def compute_document_id(file_path: str) -> str:
        """
        Content-addressed ID via SHA-256 (truncated to 16 hex chars).

        Same file re-ingested produces the same ID, enabling idempotent
        upserts in the vector store without manual deduplication.
        """
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha.update(block)
        return sha.hexdigest()[:16]

    def detect_mime_type(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        if suffix not in self._EXTENSION_TO_MIME:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported extensions: {list(self._EXTENSION_TO_MIME.keys())}"
            )
        return self._EXTENSION_TO_MIME[suffix]

    def create_document(self, file_path: str) -> Document:
        """Build the Document envelope before chunk extraction."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        document_id = self.compute_document_id(file_path)
        mime_type = self.detect_mime_type(file_path)
        title = path.stem.replace("_", " ").replace("-", " ").title()

        return Document(
            document_id=document_id,
            file_path=str(path.resolve()),
            title=title,
            mime_type=mime_type,
        )

    def process(self, file_path: str) -> tuple[Document, list[Chunk]]:
        """
        Entry point: returns a (Document, [Chunk]) pair.

        Sets document.status to PROCESSED or FAILED, so callers have a
        single place to check for errors without catching exceptions.
        """
        document = self.create_document(file_path)

        try:
            raw_text = self._extract_text_by_mime(file_path, document.mime_type)
            chunks = list(self._chunk_text(raw_text, document.document_id))
            document.chunk_count = len(chunks)
            document.status = DocumentStatus.PROCESSED
            document.ingested_at = datetime.now(timezone.utc)
        except Exception as exc:
            document.status = DocumentStatus.FAILED
            document.error_message = str(exc)
            chunks = []

        return document, chunks

    def _extract_text_by_mime(self, file_path: str, mime_type: str) -> str:
        method_name = self._MIME_EXTRACTORS.get(mime_type)
        if not method_name:
            raise ValueError(f"No extractor for MIME type: {mime_type}")
        return getattr(self, method_name)(file_path)

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        """
        Extract text from all pages, joining with double-newline so
        downstream chunking doesn't merge content across page boundaries.
        """
        reader = pypdf.PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    @staticmethod
    def _extract_text(file_path: str) -> str:
        return Path(file_path).read_text(encoding="utf-8")

    def _chunk_text(self, text: str, document_id: str) -> Generator[Chunk, None, None]:
        """
        Fixed-size sliding window chunker.

        The overlap window ensures a sentence bisected by a chunk boundary
        still has sufficient context in both adjacent chunks.

        Extension point: replace this method with a semantic chunker
        (e.g. split on paragraph/heading boundaries) without touching any
        other code.
        """
        cfg = self.config
        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + cfg.chunk_size, text_length)
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= cfg.min_chunk_size:
                yield Chunk(
                    chunk_id=str(uuid.uuid4()),
                    source_document_id=document_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    metadata={
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end,
                    },
                )
                chunk_index += 1

            if end == text_length:
                break
            start += cfg.chunk_size - cfg.chunk_overlap
