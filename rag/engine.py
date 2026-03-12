import os
from datetime import datetime, timezone

import chromadb
from google import genai
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from rag.doc_processor import ChunkingConfig, DocumentProcessor
from rag.models import (
    Chunk,
    Document,
    DocumentStatus,
    RAGResponse,
    RetrievedChunk,
)


class RAGEngine:
    """
    Orchestrates the full RAG pipeline: ingest → retrieve → generate.

    Design decisions:
    - ChromaDB PersistentClient: index survives process restarts without
      running a separate server process.
    - Two collections: one for chunks (with embeddings) and one for document
      metadata (without). Keeps query performance clean and allows schema
      evolution of each independently.
    - DefaultEmbeddingFunction: uses all-MiniLM-L6-v2 via sentence-transformers,
      bundled with chromadb. Injectable for testing or model swaps.
    - Cosine similarity: insensitive to document length, important when
      chunking produces variable-length pieces.
    - Async-native Anthropic client: matches the FastMCP event loop.
    """

    CHUNKS_COLLECTION = "rag_chunks"
    DOCS_COLLECTION = "rag_documents"
    DEFAULT_MODEL = "gemini-2.5-flash"
    DEFAULT_N_RESULTS = 5

    def __init__(
        self,
        persist_directory: str = "./.chromadb",
        model: str = DEFAULT_MODEL,
        embedding_function=None,
        chunking_config: ChunkingConfig | None = None,
    ):
        self.model = model
        self._gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        self._chroma = chromadb.PersistentClient(path=persist_directory)
        self._embed_fn = embedding_function or DefaultEmbeddingFunction()

        # Chunk collection stores text + embeddings
        self._chunks_col = self._chroma.get_or_create_collection(
            name=self.CHUNKS_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Document metadata collection — no embeddings needed
        self._docs_col = self._chroma.get_or_create_collection(
            name=self.DOCS_COLLECTION,
        )

        self._processor = DocumentProcessor(chunking_config)

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    async def ingest_document(self, file_path: str) -> Document:
        """
        Full ingestion pipeline for a single file.

        Idempotent: re-ingesting the same file replaces existing chunks via
        upsert, so callers do not need to track prior ingestion state.
        Returns the Document with status PROCESSED or FAILED.
        """
        document, chunks = self._processor.process(file_path)

        if document.status == DocumentStatus.FAILED:
            return document

        self._docs_col.upsert(
            ids=[document.document_id],
            documents=[document.title],
            metadatas=[{
                "document_id": document.document_id,
                "file_path": document.file_path,
                "title": document.title,
                "mime_type": document.mime_type,
                "chunk_count": document.chunk_count,
                "ingested_at": document.ingested_at.isoformat(),
                "status": document.status.value,
            }],
        )

        if chunks:
            self._chunks_col.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )

        return document

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def _retrieve(self, query: str, n_results: int) -> list[RetrievedChunk]:
        """
        Embed the query and retrieve the top-k most similar chunks.

        ChromaDB automatically embeds the query using the same function
        registered on the collection. The _docs_col lookup enriches each
        chunk with its document title for citation rendering.
        """
        if self._chunks_col.count() == 0:
            return []

        results = self._chunks_col.query(
            query_texts=[query],
            n_results=min(n_results, self._chunks_col.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [0, 1]
            similarity = 1.0 - (distance / 2.0)

            chunk = Chunk(
                chunk_id="",
                source_document_id=meta.get("document_id", ""),
                text=text,
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            )
            title = self._get_document_title(meta.get("document_id", ""))
            retrieved.append(RetrievedChunk(chunk=chunk, score=similarity, document_title=title))

        return retrieved

    def _get_document_title(self, document_id: str) -> str:
        try:
            result = self._docs_col.get(ids=[document_id], include=["metadatas"])
            if result["metadatas"]:
                return result["metadatas"][0].get("title", "Unknown Document")
        except Exception:
            pass
        return "Unknown Document"

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a structured context block.

        Explicit [Source N] labels let the generation prompt instruct Claude
        to cite by number, producing verifiable, grounded answers.
        """
        parts = []
        for i, rc in enumerate(chunks, start=1):
            parts.append(
                f"[Source {i}] (from '{rc.document_title}', "
                f"relevance: {rc.score:.2f})\n{rc.chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are a precise research assistant. Answer the question using ONLY the provided sources.
If the sources do not contain sufficient information, say so explicitly — do not hallucinate.
Cite sources using [Source N] notation inline.

<sources>
{context}
</sources>

Question: {query}

Answer:"""

    async def query(self, query: str, n_results: int = DEFAULT_N_RESULTS) -> RAGResponse:
        """
        Full RAG pipeline: retrieve relevant chunks, then generate a cited answer.

        Separating _retrieve() and generation makes each step independently
        testable and allows future enhancements like re-ranking.
        """
        retrieved_chunks = self._retrieve(query, n_results)

        if not retrieved_chunks:
            return RAGResponse(
                answer="No documents have been indexed yet. Use ingest_document first.",
                sources=[],
                query=query,
                model=self.model,
                retrieved_chunk_count=0,
            )

        context = self._build_context_block(retrieved_chunks)
        prompt = self._build_prompt(query, context)

        response = await self._gemini.aio.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        answer = response.text

        return RAGResponse(
            answer=answer,
            sources=retrieved_chunks,
            query=query,
            model=self.model,
            retrieved_chunk_count=len(retrieved_chunks),
        )

    # -------------------------------------------------------------------------
    # Catalogue
    # -------------------------------------------------------------------------

    def list_documents(self) -> list[Document]:
        """
        Return all indexed documents from the metadata collection.

        O(documents), not O(chunks) — does not touch the chunk collection.
        """
        result = self._docs_col.get(include=["metadatas"])
        documents = []
        for meta in result["metadatas"]:
            doc = Document(
                document_id=meta["document_id"],
                file_path=meta["file_path"],
                title=meta["title"],
                mime_type=meta["mime_type"],
                status=DocumentStatus(meta["status"]),
                chunk_count=meta["chunk_count"],
                ingested_at=datetime.fromisoformat(meta["ingested_at"]),
            )
            documents.append(doc)
        return documents

    def delete_document(self, document_id: str) -> bool:
        """
        Remove a document and all its chunks from the index.

        Two-phase delete: chunks first (identified via metadata filter),
        then the document metadata record.
        """
        chunk_results = self._chunks_col.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        if chunk_results["ids"]:
            self._chunks_col.delete(ids=chunk_results["ids"])

        try:
            self._docs_col.delete(ids=[document_id])
            return True
        except Exception:
            return False
