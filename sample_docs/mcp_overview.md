# Model Context Protocol (MCP): Architecture and Design

## What is MCP?

The Model Context Protocol (MCP) is an open standard that defines how AI models communicate with external tools, data sources, and services. Rather than hard-coding integrations, MCP creates a universal interface layer: a server exposes capabilities, and any compliant client (such as Claude) can discover and invoke them at runtime.

MCP was designed to solve a fundamental scaling problem in AI tooling. Without a shared protocol, every model provider and every tool vendor must maintain custom integrations — an O(n×m) problem. MCP reduces this to O(n+m): each tool builds one MCP server, each client implements one MCP client.

## Core Primitives

MCP defines three primitives that servers can expose to clients:

**Tools** are callable functions. A tool has a name, a JSON Schema describing its input parameters, and a description the model uses to decide when to invoke it. Tools are the most common primitive; virtually every MCP server exposes at least one. Examples: `search_documents`, `run_query`, `convert_file`.

**Resources** are readable data sources, identified by URI. A resource might represent a file, a database row, a live API feed, or any other addressable datum. Resources support MIME types, letting the client understand whether it is receiving plain text, JSON, binary data, etc.

**Prompts** are pre-defined prompt templates with optional dynamic arguments. They allow servers to share curated, reusable prompts — for example, a code review prompt parameterized by language and severity threshold.

## Transport Mechanisms

MCP is transport-agnostic at the protocol level, but two transports are in common use:

**stdio (standard I/O)** — the client spawns the server as a subprocess and communicates via stdin/stdout pipes. This is the simplest transport: no network configuration, no port management, no authentication overhead. It is ideal for local development, CLI tools, and single-user deployments. The tradeoff is that stdio is inherently point-to-point: one client process maps to one server process.

**SSE (Server-Sent Events)** — the server runs as a persistent HTTP service. Clients connect over HTTP, receive events via the SSE stream, and send requests via POST. SSE transport supports multiple concurrent clients sharing a single server instance, making it appropriate for multi-user or cloud deployments. It also allows the server to live on a different host from the client.

## Session Lifecycle

A session begins when the client sends an `initialize` request, including its protocol version and capability declarations. The server responds with its own capabilities — which primitives it exposes and what protocol features it supports. This handshake lets old clients work with new servers (and vice versa) by negotiating a compatible feature set.

After initialization, the client can call `tools/list`, `resources/list`, or `prompts/list` to discover available capabilities. Tool invocations are sent as `tools/call` requests and return structured results. Long-running tools can emit progress notifications mid-execution, allowing the client to render progress indicators.

The session ends when either party closes the transport. For stdio, this typically happens when the client process exits or explicitly terminates the subprocess.

## Roots: Scoped File Access

MCP includes a "roots" mechanism that lets the client declare which file system paths the server is allowed to access. When a client opens a session, it can provide a list of Root objects — each containing a `file://` URI. The server can request this list at any time via `roots/list`. Servers are expected to respect these boundaries and refuse operations that fall outside the declared roots.

This design keeps authorization in the client's hands. The server does not need to know the user's identity or maintain its own ACL; it simply checks the roots list provided by the trusted client.

## Sampling: Model-in-the-Loop

MCP supports a "sampling" capability where the server can ask the client to perform an LLM inference call on its behalf. The server sends a `sampling/createMessage` request with a prompt; the client invokes its model and returns the result. This allows servers to incorporate AI reasoning without embedding a model themselves — the client's model is the model.

This is architecturally powerful: a single Claude client can orchestrate multiple MCP servers, each of which may in turn use Claude (via sampling) for subtasks, creating a tree of AI-assisted computation.

## FastMCP: Python Server Framework

FastMCP is a Python library that dramatically reduces the boilerplate of writing MCP servers. A minimal FastMCP server looks like this:

```python
from mcp.server.fastmcp import FastMCP, Context
mcp = FastMCP(name="My Server")

@mcp.tool()
async def my_tool(input: str, *, ctx: Context) -> str:
    await ctx.info("Processing...")
    return f"Result: {input}"

mcp.run(transport="stdio")
```

The `@mcp.tool()` decorator handles JSON Schema generation from type hints, input validation via Pydantic, and response serialization. The `Context` object provides access to logging, progress reporting, and the session's root list.
