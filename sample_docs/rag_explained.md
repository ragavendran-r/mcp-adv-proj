# Retrieval-Augmented Generation (RAG): Concepts and Implementation

## The Problem RAG Solves

Large language models (LLMs) have two fundamental limitations when used as knowledge stores. First, their knowledge is frozen at training time — they cannot answer questions about events or documents that postdate their training cutoff. Second, they hallucinate: when the model does not know an answer, it sometimes generates a plausible-sounding but factually incorrect response rather than admitting uncertainty.

Retrieval-Augmented Generation addresses both problems by separating *storage* from *reasoning*. The model's parametric knowledge (encoded in its weights) handles reasoning, language understanding, and synthesis. An external vector database handles knowledge storage and retrieval. At query time, relevant passages are fetched from the database and injected into the prompt, giving the model grounded context to reason over.

## The RAG Pipeline

A RAG system consists of two phases: an offline ingestion phase and an online query phase.

**Ingestion phase:**
1. Load documents from the source (files, databases, web pages, APIs).
2. Split each document into chunks — fixed-size passages with overlapping boundaries.
3. Generate an embedding vector for each chunk using an embedding model.
4. Store chunks and their embeddings in a vector database.

**Query phase:**
1. Receive a user question.
2. Embed the question using the same embedding model.
3. Perform a nearest-neighbor search in the vector database to retrieve the top-k most similar chunks.
4. Construct a prompt that includes the retrieved chunks as context.
5. Send the prompt to the LLM and return its response.

## Why RAG Reduces Hallucination

When context is injected into the prompt, a well-instructed model will answer by citing and reasoning over that context rather than drawing on its (potentially incorrect) parametric knowledge. The prompt can explicitly instruct: "Answer using only the provided sources. If the sources do not contain sufficient information, say so." This shifts the model's behavior from generation to comprehension — a much lower-hallucination regime.

The quality of the retrieval step is critical. If irrelevant chunks are retrieved, the model may still produce incorrect answers — or may correctly decline to answer, which is better than hallucinating. If relevant chunks are missed, the model falls back on parametric memory. Good chunking, good embeddings, and an appropriate similarity threshold are what separate a reliable RAG system from an unreliable one.

## Chunking Strategies

Chunking is one of the most consequential implementation decisions in a RAG system.

**Fixed-size chunking** splits text at a fixed character or token count, with an optional overlap between adjacent chunks. The overlap ensures that a sentence bisected by a chunk boundary still appears in its entirety in one of the two chunks, preserving local context. This is the simplest strategy and works well for unstructured prose.

**Semantic chunking** splits at natural boundaries — paragraph breaks, section headings, or sentence endings — rather than at a fixed count. This produces chunks that are semantically coherent but variable in size. Variable size can cause issues if the embedding model has a tight token limit.

**Hierarchical chunking** creates chunks at multiple granularities (e.g., paragraph-level and section-level) and indexes both. At query time, the system retrieves at the finer granularity but uses the coarser chunk as the actual context window, giving the model more surrounding information.

A commonly effective starting point: 1000-character chunks with 200-character overlap.

## Embedding Models

An embedding model converts a text string into a dense vector — a list of floating-point numbers where semantically similar texts produce geometrically close vectors. The choice of embedding model determines the quality of retrieval.

**all-MiniLM-L6-v2** is a lightweight sentence-transformers model (22M parameters) that produces 384-dimensional vectors. It runs on CPU, fits in under 100 MB of RAM, and achieves competitive retrieval benchmarks. It is the default embedding model in ChromaDB and is an excellent choice for local development and small-to-medium corpora.

**text-embedding-3-large** (OpenAI) and **voyage-3** (Voyage AI, Anthropic's recommended partner) are larger, API-based models that produce higher-quality embeddings, particularly on domain-specific or technical text. For production systems where retrieval accuracy is critical, an API-based embedding model is typically worth the cost.

**Critically**: the same embedding model must be used at ingestion time and query time. Embedding a query with a different model than was used to index the documents produces meaningless similarity scores.

## Vector Databases

A vector database stores embedding vectors and supports fast approximate nearest-neighbor (ANN) search. The core operation is: "given a query vector, find the k stored vectors most similar to it."

**ChromaDB** is an embedded, open-source vector database written in Python. It supports persistent storage on disk, CRUD operations on documents, metadata filtering, and pluggable embedding functions. Its `PersistentClient` mode stores data to a local directory, requiring no running server. This makes it ideal for development, prototyping, and single-node deployments.

**FAISS** (Facebook AI Similarity Search) is a high-performance ANN library optimized for in-memory search. It does not include persistence or metadata out of the box, making it lower-level than ChromaDB. FAISS is the right choice when raw search throughput is the primary constraint.

**Pinecone, Weaviate, Qdrant** are managed or self-hosted vector databases designed for production-scale deployments with multi-tenant support, replication, and horizontal scaling.

## Similarity Metrics

The most common similarity metrics for text embeddings are:

**Cosine similarity** measures the angle between two vectors, ignoring their magnitude. It is length-normalized, which means a short chunk and a long chunk that discuss the same topic receive similar scores. This is usually the right choice for text.

**L2 (Euclidean) distance** measures the absolute distance between vectors. It is sensitive to vector magnitude, which can disadvantage short documents even when they are topically relevant. Generally avoided for text embeddings.

ChromaDB uses cosine similarity by default when a collection is created with `"hnsw:space": "cosine"`. ChromaDB returns distances in the range [0, 2]; similarity can be recovered as `1 - (distance / 2)`.

## Re-ranking

A common enhancement is two-stage retrieval: retrieve a larger candidate set (e.g., top-50) using fast ANN search, then re-rank the candidates using a slower but more accurate cross-encoder model before selecting the final top-k for the prompt. Cross-encoders attend jointly to the query and the candidate, producing much more accurate relevance scores than bi-encoders (embedding models), but at higher computational cost. Re-ranking is particularly valuable when the query and the documents are in different styles (e.g., a conversational question vs. technical documentation).
