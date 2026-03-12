# MCP + RAG Demo Project

A hands-on project demonstrating **Retrieval-Augmented Generation (RAG)** and **Model Context Protocol (MCP)** — covering document ingestion, vector search, LLM-generated answers, sampling, logging, progress reporting, and video conversion.

[![CI](https://github.com/ragavendran-r/mcp-adv-proj/actions/workflows/ci.yml/badge.svg)](https://github.com/ragavendran-r/mcp-adv-proj/actions/workflows/ci.yml)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Running the Projects](#running-the-projects)
  - [RAG Demo (interactive CLI)](#rag-demo-interactive-cli)
  - [RAG MCP Server](#rag-mcp-server)
  - [Sampling & Logging Demo](#sampling--logging-demo)
  - [Roots & Video Conversion Demo](#roots--video-conversion-demo)
- [Running Tests](#running-tests)
- [Architecture](#architecture)

---

## Overview

| Module | Description |
|---|---|
| **RAG pipeline** | Ingest documents → chunk → embed → store in ChromaDB → retrieve → generate cited answers via Google Gemini |
| **RAG MCP Server** | Exposes the RAG pipeline as four MCP tools over stdio transport |
| **RAG Demo CLI** | Interactive terminal app for querying the RAG system end-to-end |
| **Sampling demo** | Shows MCP sampling, server-side logging, and progress reporting |
| **Roots demo** | File system access with controlled root paths and MP4 video conversion |

---

## Project Structure

```
.
├── rag/
│   ├── engine.py          # RAGEngine — orchestrates ingest, retrieve, query
│   ├── doc_processor.py   # Document → chunks pipeline (PDF, TXT, Markdown)
│   └── models.py          # Dataclasses: Chunk, Document, RAGResponse, …
├── core/
│   ├── claude.py          # Async Anthropic API wrapper (chat + streaming)
│   ├── tools.py           # MCP tool definitions (file system, video)
│   ├── video_converter.py # FFmpeg-backed MP4 → AVI/MOV/WebM/MKV/GIF
│   └── utils.py           # Shared helpers
├── rag_server.py          # MCP server exposing RAG as tools
├── rag_demo.py            # Interactive CLI demo
├── server.py              # MCP server for sampling/logging/progress demo
├── client.py              # MCP client for sampling demo
├── main.py                # Entry point for roots + video conversion demo
├── sample_docs/           # Sample Markdown files for demo ingestion
├── tests/                 # Full test suite (97% coverage)
└── .github/workflows/
    └── ci.yml             # CI: runs tests, fails if coverage < 80%
```

---

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Google Gemini API key (for RAG features)
- Anthropic API key (for chat/sampling features)
- FFmpeg (for video conversion)

Install FFmpeg on macOS:

```bash
brew install ffmpeg
```

---

## Setup

1. Install uv if not already installed:

```bash
pip install uv
```

2. Install all dependencies:

```bash
uv sync --all-groups
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required for RAG features
GEMINI_API_KEY="your-google-gemini-api-key"

# Optional RAG overrides
GEMINI_MODEL="gemini-2.5-flash"        # default: gemini-2.5-flash
CHROMA_PERSIST_DIR="./.chromadb"       # default: ./.chromadb

# Required for chat / sampling demo
ANTHROPIC_API_KEY="your-anthropic-api-key"
CLAUDE_MODEL="claude-sonnet-4-0"       # default: claude-sonnet-4-0
```

---

## Running the Projects

### RAG Demo (interactive CLI)

An interactive terminal application that auto-ingests the `sample_docs/` directory on first run and lets you query documents conversationally.

```bash
uv run rag_demo.py
```

**Commands inside the demo:**

| Command | Description |
|---|---|
| `<question>` | Query the RAG system |
| `/ingest <path>` | Index a file or all supported files in a directory |
| `/list` | Show all indexed documents |
| `/delete <id>` | Remove a document by its ID |
| `/help` | Show help text |
| `Ctrl-C` | Exit |

**Supported file types:** `.txt`, `.md`, `.pdf`

---

### RAG MCP Server

Exposes the RAG pipeline as four MCP tools over stdio transport. Connect to it from any MCP client (e.g. Claude Desktop).

```bash
uv run rag_server.py
```

**Available MCP tools:**

| Tool | Description |
|---|---|
| `ingest_document` | Index a file into the vector store |
| `query_rag` | Retrieve relevant chunks and generate a cited answer |
| `list_documents` | List all indexed documents |
| `delete_document` | Remove a document and all its chunks |

---

### Sampling & Logging Demo

Demonstrates MCP sampling, server-side logging, and progress reporting.

```bash
uv run client.py
```

---

### Roots & Video Conversion Demo

Demonstrates MCP roots (controlled file system access) and MP4 video conversion via FFmpeg.

```bash
uv run main.py <root1> [root2] ...
```

Examples:

```bash
# Single directory
uv run main.py /path/to/videos

# Multiple directories
uv run main.py ~/videos ~/Documents

# Current directory
uv run main.py .
```

**Available tools:** `list_roots`, `read_dir`, `convert_video`

**Supported video output formats:** `avi`, `mov`, `webm`, `mkv`, `gif`

---

## Running Tests

```bash
# Run all tests with coverage report
uv run pytest tests/ --cov --cov-report=term-missing

# Enforce the 80% coverage gate (same check as CI)
uv run pytest tests/ --cov --cov-fail-under=80
```

Current coverage: **97%** across 142 tests.

The CI pipeline (`.github/workflows/ci.yml`) runs automatically on every push and pull request to `main`. PRs are blocked from merging if coverage drops below 80%.

---

## Architecture

```
Document
   │
   ▼
DocumentProcessor          doc_processor.py
  ├─ detect MIME type
  ├─ extract text (TXT / MD / PDF)
  ├─ content-addressed ID (SHA-256)
  └─ sliding-window chunking (1000 chars, 200 overlap)
   │
   ▼
ChromaDB (PersistentClient) rag/engine.py
  ├─ rag_chunks collection   (embeddings + text)
  └─ rag_documents collection (metadata catalogue)
   │
   ▼  query time
Embedding + cosine similarity retrieval
   │
   ▼
Google Gemini (generate_content)
   │
   ▼
RAGResponse  ─── sources, answer, chunk count, model
```
