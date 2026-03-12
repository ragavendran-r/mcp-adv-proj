"""Tests for core/utils.py — file URL helpers."""

from pathlib import Path

from core.utils import file_url_to_path


class TestFileUrlToPath:
    def test_simple_unix_path(self):
        result = file_url_to_path("file:///Users/alice/docs/file.txt")
        assert result == Path("/Users/alice/docs/file.txt")

    def test_path_with_spaces(self):
        result = file_url_to_path("file:///Users/alice/my%20docs/file.txt")
        assert result == Path("/Users/alice/my docs/file.txt")

    def test_path_with_nested_dirs(self):
        result = file_url_to_path("file:///home/user/a/b/c/doc.md")
        assert result == Path("/home/user/a/b/c/doc.md")

    def test_returns_path_object(self):
        result = file_url_to_path("file:///tmp/test.txt")
        assert isinstance(result, Path)

    def test_windows_style_path(self):
        """Paths like /C:/Users/... should strip the leading slash."""
        result = file_url_to_path("file:///C:/Users/alice/file.txt")
        # The function strips the leading slash for Windows-style paths
        assert str(result) == "C:/Users/alice/file.txt"

    def test_url_encoded_special_chars(self):
        result = file_url_to_path("file:///tmp/hello%20world.txt")
        assert result == Path("/tmp/hello world.txt")

    def test_accepts_string_or_url_object(self):
        """Should work with anything that can be str()-ed."""
        from pydantic import AnyUrl
        url = AnyUrl("file:///tmp/file.txt")
        result = file_url_to_path(url)
        assert result == Path("/tmp/file.txt")
