"""Tests for core/claude.py — Claude API wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.claude import Claude


def _make_message(text: str = "Hello!", stop_reason: str = "end_turn"):
    msg = MagicMock()
    msg.stop_reason = stop_reason
    block = MagicMock()
    block.type = "text"
    block.text = text
    msg.content = [block]
    return msg


@pytest.fixture
def claude():
    with patch("core.claude.AsyncAnthropic"):
        return Claude(model="claude-test")


class TestMessageHelpers:
    def test_add_user_message_string(self, claude):
        messages = []
        claude.add_user_message(messages, "Hello")
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_add_assistant_message_string(self, claude):
        messages = []
        claude.add_assistant_message(messages, "Hi there")
        assert messages == [{"role": "assistant", "content": "Hi there"}]

    def test_add_user_message_non_message_uses_value_directly(self, claude):
        """Non-Message values are stored as content directly."""
        raw_list = ["block1", "block2"]
        messages = []
        claude.add_user_message(messages, raw_list)
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] is raw_list

    def test_add_assistant_message_non_message_uses_value_directly(self, claude):
        raw_list = ["block1", "block2"]
        messages = []
        claude.add_assistant_message(messages, raw_list)
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] is raw_list

    def test_text_from_message_single_text_block(self, claude):
        msg = _make_message("The answer.")
        assert claude.text_from_message(msg) == "The answer."

    def test_text_from_message_multiple_text_blocks(self, claude):
        msg = MagicMock()
        b1 = MagicMock()
        b1.type = "text"
        b1.text = "Hello "
        b2 = MagicMock()
        b2.type = "text"
        b2.text = "world"
        b3 = MagicMock()
        b3.type = "tool_use"  # should be skipped
        msg.content = [b1, b2, b3]
        assert claude.text_from_message(msg) == "Hello world"

    def test_text_from_message_no_text_blocks(self, claude):
        msg = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        msg.content = [tool_block]
        assert claude.text_from_message(msg) == ""


class TestChat:
    async def test_chat_basic(self, claude):
        expected = _make_message("Response text")
        claude.client.messages.create = AsyncMock(return_value=expected)

        result = await claude.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result is expected
        claude.client.messages.create.assert_awaited_once()

    async def test_chat_passes_model(self, claude):
        claude.client.messages.create = AsyncMock(return_value=_make_message())
        await claude.chat(messages=[])
        call_kwargs = claude.client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-test"

    async def test_chat_with_system(self, claude):
        claude.client.messages.create = AsyncMock(return_value=_make_message())
        await claude.chat(messages=[], system="You are helpful.")
        call_kwargs = claude.client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."

    async def test_chat_without_system_omits_key(self, claude):
        claude.client.messages.create = AsyncMock(return_value=_make_message())
        await claude.chat(messages=[])
        call_kwargs = claude.client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs

    async def test_chat_with_tools(self, claude):
        claude.client.messages.create = AsyncMock(return_value=_make_message())
        tools = [{"name": "search", "description": "...", "input_schema": {}}]
        await claude.chat(messages=[], tools=tools)
        call_kwargs = claude.client.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == tools

    async def test_chat_without_tools_omits_key(self, claude):
        claude.client.messages.create = AsyncMock(return_value=_make_message())
        await claude.chat(messages=[])
        call_kwargs = claude.client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs

    async def test_chat_thinking_enabled(self, claude):
        claude.client.messages.create = AsyncMock(return_value=_make_message())
        await claude.chat(messages=[], thinking=True, thinking_budget=2048)
        call_kwargs = claude.client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs["thinking"]["budget_tokens"] == 2048


async def _async_iter(items):
    """Async generator helper for mocking stream iteration."""
    for item in items:
        yield item


def _make_mock_stream(events=(), final_message=None):
    """Build a mock async context manager that yields events and returns a final message."""
    if final_message is None:
        final_message = _make_message()

    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    mock_stream.__aiter__ = lambda _: _async_iter(events)
    mock_stream.get_final_message = AsyncMock(return_value=final_message)
    return mock_stream


class TestChatStream:
    async def test_chat_stream_returns_final_message(self, claude):
        expected = _make_message("streamed response")
        claude.client.messages.stream = MagicMock(
            return_value=_make_mock_stream(final_message=expected)
        )
        result = await claude.chat_stream(messages=[{"role": "user", "content": "Hi"}])
        assert result is expected

    async def test_chat_stream_with_on_event(self, claude):
        expected = _make_message("event response")
        events_received = []

        async def on_event(ev):
            events_received.append(ev)

        fake_event = MagicMock()
        claude.client.messages.stream = MagicMock(
            return_value=_make_mock_stream(events=[fake_event], final_message=expected)
        )

        result = await claude.chat_stream(messages=[], on_event=on_event)
        assert result is expected
        assert fake_event in events_received

    async def test_chat_stream_passes_model(self, claude):
        claude.client.messages.stream = MagicMock(return_value=_make_mock_stream())
        await claude.chat_stream(messages=[])
        call_kwargs = claude.client.messages.stream.call_args.kwargs
        assert call_kwargs["model"] == "claude-test"
