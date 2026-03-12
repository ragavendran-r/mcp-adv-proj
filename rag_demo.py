"""
RAG Interactive Demo

Showcases a full RAG pipeline built on MCP + ChromaDB + Claude.
Auto-ingests sample_docs/ on first run if the index is empty.

Usage:
    uv run rag_demo.py

Commands:
    /ingest <path>    Index a file or every supported file in a directory
    /list             Show all indexed documents
    /delete <id>      Remove a document by its ID
    /help             Show this help text
    <question>        Query the RAG system
    Ctrl-C            Exit
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import types
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from pyboxen import boxen

from mcp_client import MCPClient


def _extract_text(result: types.CallToolResult, fallback: str = "{}") -> str:
    """Safely extract text from an MCP tool result's first content item."""
    if not result.content:
        return fallback
    item = result.content[0]
    if isinstance(item, types.TextContent):
        return item.text
    return fallback

load_dotenv()

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _print_banner():
    content = (
        "RAG Demo  |  MCP + ChromaDB + Claude\n"
        "Knowledge base powered by vector search\n"
        "Type /help for commands  •  Ctrl-C to exit"
    )
    print(boxen(content, title="RAG System", style="double", color="cyan", padding=1))


def _render_rag_response(result_json: str):
    data = json.loads(result_json)
    answer = data.get("answer", "")
    sources = data.get("sources", [])
    chunk_count = data.get("retrieved_chunks", 0)

    print(boxen(answer, title="Answer", style="rounded", color="green", padding=1))

    if sources:
        lines = [f"Retrieved {chunk_count} chunk(s) from:\n"]
        for i, s in enumerate(sources, 1):
            excerpt = s["excerpt"]
            if len(excerpt) > 120:
                excerpt = excerpt[:120] + "..."
            lines.append(
                f"  [{i}] {s['document_title']}  "
                f"(relevance: {s['relevance_score']:.1%})\n"
                f"      {excerpt}"
            )
        print(
            boxen(
                "\n".join(lines).strip(),
                title="Sources",
                style="rounded",
                color="yellow",
                padding=0,
            )
        )


def _render_document_list(result_json: str):
    data = json.loads(result_json)
    count = data.get("document_count", 0)
    docs = data.get("documents", [])

    if count == 0:
        print("Index is empty. Use /ingest <path> to add documents.")
        return

    lines = [f"{count} document(s) in index:\n"]
    for doc in docs:
        ingested = doc.get("ingested_at", "unknown")
        lines.append(
            f"  [{doc['document_id']}]  {doc['title']}\n"
            f"    {doc['chunk_count']} chunks  •  {doc['mime_type']}  •  {ingested}"
        )
    print(
        boxen(
            "\n".join(lines).strip(),
            title="Knowledge Base",
            style="rounded",
            color="blue",
            padding=0,
        )
    )


async def _ingest_path(session, path_str: str):
    path = Path(path_str).expanduser().resolve()

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = [f for f in sorted(path.iterdir()) if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    else:
        print(f"Path not found: {path}")
        return

    if not files:
        print(f"No supported files (.pdf, .txt, .md) found in: {path}")
        return

    for file in files:
        print(f"  Ingesting {file.name}...")
        result = await session.call_tool("ingest_document", {"file_path": str(file)})
        data = json.loads(_extract_text(result))
        if data.get("status") == "success":
            print(
                f"  Indexed '{data['title']}' "
                f"— {data['chunk_count']} chunks  [{data['document_id']}]"
            )
        else:
            print(f"  Failed: {data.get('error', 'unknown error')}")


async def run_demo():
    _print_banner()

    session_style = Style.from_dict({"prompt": "#888888"})
    prompt_session = PromptSession(style=session_style)

    async with MCPClient(command="uv", args=["run", "rag_server.py"]) as rag_client:
        session = rag_client.session()

        # Auto-ingest sample docs if the index is empty
        list_result = await session.call_tool("list_documents", {})
        list_data = json.loads(_extract_text(list_result))
        if list_data["document_count"] == 0 and SAMPLE_DOCS_DIR.exists():
            print(f"Auto-ingesting sample documents from {SAMPLE_DOCS_DIR.name}/...\n")
            await _ingest_path(session, str(SAMPLE_DOCS_DIR))
            print()

        print("Ready. Ask a question or type /help\n")

        while True:
            try:
                user_input = (await prompt_session.prompt_async("rag> ")).strip()
                if not user_input:
                    continue

                if user_input in ("/help", "/?"):
                    print(
                        "  /ingest <path>    Index a file or directory\n"
                        "  /list             List indexed documents\n"
                        "  /delete <id>      Delete a document by ID\n"
                        "  /help             Show this help\n"
                        "  <question>        Query the RAG knowledge base"
                    )

                elif user_input.startswith("/ingest "):
                    path_arg = user_input[len("/ingest "):].strip()
                    await _ingest_path(session, path_arg)

                elif user_input == "/list":
                    result = await session.call_tool("list_documents", {})
                    _render_document_list(_extract_text(result))

                elif user_input.startswith("/delete "):
                    doc_id = user_input[len("/delete "):].strip()
                    result = await session.call_tool("delete_document", {"document_id": doc_id})
                    data = json.loads(_extract_text(result))
                    print(f"Status: {data['status']}  (id: {data['document_id']})")

                else:
                    print("Searching knowledge base...\n")
                    result = await session.call_tool(
                        "query_rag",
                        {"question": user_input, "n_results": 5},
                    )
                    if result.isError:
                        print(f"Error from RAG server: {_extract_text(result, 'Unknown error')}")
                    else:
                        _render_rag_response(_extract_text(result))

                print()

            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
            except Exception as exc:
                print(f"Error: {exc}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(run_demo())
