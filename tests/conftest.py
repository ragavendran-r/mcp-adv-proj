"""
Shared fixtures for the test suite.

Module-level env var setup runs before any test module is imported,
so rag_server's module-level RAGEngine uses the temp chroma dir.
"""

import hashlib
import os
import tempfile

import pytest

# Must be set before rag_server (or any module that imports RAGEngine at
# module level) is first imported. pytest loads conftest.py first.
_TEST_CHROMA_DIR = tempfile.mkdtemp(prefix="pytest_chroma_")
os.environ["CHROMA_PERSIST_DIR"] = _TEST_CHROMA_DIR


class MockEmbeddingFunction:
    """
    Deterministic mock embedding function.
    Avoids downloading all-MiniLM-L6-v2 during tests.
    Implements the interface required by ChromaDB 1.x:
      - __call__   for ingest-time embedding
      - embed_query for query-time embedding
      - name / is_legacy for collection configuration validation
    """

    def name(self) -> str:
        return "mock_embedding"

    def is_legacy(self) -> bool:
        return False

    def _embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            vec = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(384)]
            result.append(vec)
        return result

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """ChromaDB 1.x calls this for query embeddings."""
        return self._embed(input)


@pytest.fixture
def mock_embed_fn():
    return MockEmbeddingFunction()


@pytest.fixture
def long_text():
    return ("The quick brown fox jumps over the lazy dog. " * 60).strip()


@pytest.fixture
def tmp_text_file(tmp_path, long_text):
    f = tmp_path / "sample.txt"
    f.write_text(long_text * 4)
    return str(f)


@pytest.fixture
def tmp_md_file(tmp_path):
    f = tmp_path / "sample.md"
    f.write_text("# Test Heading\n\n" + ("Paragraph content. " * 100))
    return str(f)
