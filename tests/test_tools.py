"""Tests for core/tools.py — ToolManager."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import TextContent

from core.tools import ToolManager


def _make_tool(name: str):
    t = MagicMock()
    t.name = name
    t.description = f"Does {name}"
    t.inputSchema = {"type": "object", "properties": {}}
    return t


def _make_client(*tool_names: str) -> MagicMock:
    client = MagicMock()
    tools = [_make_tool(n) for n in tool_names]
    client.list_tools = AsyncMock(return_value=tools)
    return client


def _make_text_tool_result(text: str, is_error: bool = False):
    result = MagicMock()
    result.isError = is_error
    content = MagicMock(spec=TextContent)
    content.text = text
    result.content = [content]
    return result


class TestGetAllTools:
    async def test_aggregates_from_multiple_clients(self):
        c1 = _make_client("search", "summarize")
        c2 = _make_client("convert")
        tools = await ToolManager.get_all_tools({"c1": c1, "c2": c2})
        names = [t["name"] for t in tools]
        assert "search" in names
        assert "summarize" in names
        assert "convert" in names

    async def test_empty_clients(self):
        tools = await ToolManager.get_all_tools({})
        assert tools == []

    async def test_tool_schema_structure(self):
        client = _make_client("my_tool")
        tools = await ToolManager.get_all_tools({"c": client})
        assert tools[0]["name"] == "my_tool"
        assert "description" in tools[0]
        assert "input_schema" in tools[0]


class TestFindClientWithTool:
    async def test_finds_correct_client(self):
        c1 = _make_client("tool_a")
        c2 = _make_client("tool_b")
        result = await ToolManager._find_client_with_tool([c1, c2], "tool_b")
        assert result is c2

    async def test_returns_none_if_not_found(self):
        c1 = _make_client("tool_a")
        result = await ToolManager._find_client_with_tool([c1], "nonexistent")
        assert result is None

    async def test_returns_first_match(self):
        c1 = _make_client("shared")
        c2 = _make_client("shared")
        result = await ToolManager._find_client_with_tool([c1, c2], "shared")
        assert result is c1


class TestBuildToolResultPart:
    def test_success_result(self):
        part = ToolManager._build_tool_result_part("id123", "output text", "success")
        assert part["tool_use_id"] == "id123"
        assert part["type"] == "tool_result"
        assert part["content"] == "output text"
        assert part["is_error"] is False

    def test_error_result(self):
        part = ToolManager._build_tool_result_part("id456", "error message", "error")
        assert part["is_error"] is True

    def test_content_is_preserved(self):
        part = ToolManager._build_tool_result_part("x", '{"key": "value"}', "success")
        assert part["content"] == '{"key": "value"}'


class TestExecuteToolRequests:
    def _make_message_with_tool_use(self, tool_name: str, tool_id: str, tool_input: dict):
        msg = MagicMock()
        block = MagicMock()
        block.type = "tool_use"
        block.id = tool_id
        block.name = tool_name
        block.input = tool_input
        msg.content = [block]
        return msg

    async def test_successful_tool_execution(self):
        client = _make_client("my_tool")
        client.call_tool = AsyncMock(return_value=_make_text_tool_result('{"result": "done"}'))
        msg = self._make_message_with_tool_use("my_tool", "call_1", {"param": "value"})

        parts = await ToolManager.execute_tool_requests({"c": client}, msg)

        assert len(parts) == 1
        assert parts[0]["tool_use_id"] == "call_1"
        assert parts[0]["is_error"] is False

    async def test_tool_not_found_returns_error(self):
        client = _make_client("other_tool")
        msg = self._make_message_with_tool_use("missing_tool", "call_2", {})

        parts = await ToolManager.execute_tool_requests({"c": client}, msg)

        assert len(parts) == 1
        assert parts[0]["is_error"] is True
        assert "Could not find that tool" in parts[0]["content"]

    async def test_tool_execution_exception_returns_error(self):
        client = _make_client("bad_tool")
        client.call_tool = AsyncMock(side_effect=RuntimeError("something broke"))
        msg = self._make_message_with_tool_use("bad_tool", "call_3", {})

        parts = await ToolManager.execute_tool_requests({"c": client}, msg)

        assert len(parts) == 1
        content = json.loads(parts[0]["content"])
        assert "error" in content

    async def test_multiple_tool_calls(self):
        client = _make_client("tool_a", "tool_b")
        client.call_tool = AsyncMock(return_value=_make_text_tool_result("ok"))

        msg = MagicMock()
        b1 = MagicMock()
        b1.type = "tool_use"
        b1.id = "id1"
        b1.name = "tool_a"
        b1.input = {}
        b2 = MagicMock()
        b2.type = "tool_use"
        b2.id = "id2"
        b2.name = "tool_b"
        b2.input = {}
        msg.content = [b1, b2]

        parts = await ToolManager.execute_tool_requests({"c": client}, msg)
        assert len(parts) == 2

    async def test_no_tool_use_blocks_returns_empty(self):
        client = _make_client("tool")
        msg = MagicMock()
        msg.content = []  # no tool_use blocks
        parts = await ToolManager.execute_tool_requests({"c": client}, msg)
        assert parts == []
