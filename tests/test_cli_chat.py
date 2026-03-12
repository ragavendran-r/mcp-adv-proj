"""Tests for core/cli_chat.py — prompt message converters."""

from unittest.mock import MagicMock

from core.cli_chat import (
    convert_prompt_message_to_message_param,
    convert_prompt_messages_to_message_params,
)


def _text_content(text: str) -> MagicMock:
    c = MagicMock()
    c.type = "text"
    c.text = text
    return c


def _make_prompt_message(role: str, content) -> MagicMock:
    msg = MagicMock()
    msg.role = role
    msg.content = content
    return msg


class TestConvertPromptMessageToMessageParam:
    def test_user_role(self):
        msg = _make_prompt_message("user", _text_content("Hello"))
        result = convert_prompt_message_to_message_param(msg)
        assert result["role"] == "user"

    def test_assistant_role(self):
        msg = _make_prompt_message("assistant", _text_content("Hi there"))
        result = convert_prompt_message_to_message_param(msg)
        assert result["role"] == "assistant"

    def test_text_content_object(self):
        msg = _make_prompt_message("user", _text_content("The question"))
        result = convert_prompt_message_to_message_param(msg)
        assert result["content"] == "The question"

    def test_text_content_dict(self):
        msg = _make_prompt_message("user", {"type": "text", "text": "From dict"})
        result = convert_prompt_message_to_message_param(msg)
        assert result["content"] == "From dict"

    def test_list_of_text_content_objects(self):
        blocks = [_text_content("Part 1"), _text_content("Part 2")]
        msg = _make_prompt_message("user", blocks)
        result = convert_prompt_message_to_message_param(msg)
        assert isinstance(result["content"], list)
        assert result["content"][0]["text"] == "Part 1"
        assert result["content"][1]["text"] == "Part 2"

    def test_list_of_text_content_dicts(self):
        blocks = [
            {"type": "text", "text": "block one"},
            {"type": "text", "text": "block two"},
        ]
        msg = _make_prompt_message("assistant", blocks)
        result = convert_prompt_message_to_message_param(msg)
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2

    def test_list_with_non_text_blocks_filters_them(self):
        text_block = _text_content("keep me")
        image_block = MagicMock()
        image_block.type = "image"
        msg = _make_prompt_message("user", [text_block, image_block])
        result = convert_prompt_message_to_message_param(msg)
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1

    def test_unknown_content_returns_empty_string(self):
        msg = _make_prompt_message("user", 12345)
        result = convert_prompt_message_to_message_param(msg)
        assert result["content"] == ""

    def test_non_user_role_maps_to_assistant(self):
        msg = _make_prompt_message("system", _text_content("system prompt"))
        result = convert_prompt_message_to_message_param(msg)
        assert result["role"] == "assistant"


class TestConvertPromptMessagesToMessageParams:
    def test_converts_list(self):
        messages = [
            _make_prompt_message("user", _text_content("Q1")),
            _make_prompt_message("assistant", _text_content("A1")),
            _make_prompt_message("user", _text_content("Q2")),
        ]
        results = convert_prompt_messages_to_message_params(messages)
        assert len(results) == 3
        assert results[0]["role"] == "user"
        assert results[1]["role"] == "assistant"

    def test_empty_list(self):
        assert convert_prompt_messages_to_message_params([]) == []
