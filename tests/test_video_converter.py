"""Tests for core/video_converter.py — VideoConverter."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.video_converter import VideoConverter


class TestValidateInput:
    def test_valid_mp4(self, tmp_path):
        f = tmp_path / "video.mp4"
        f.touch()
        result = VideoConverter.validate_input(str(f))
        assert result == f

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Input file not found"):
            VideoConverter.validate_input(str(tmp_path / "missing.mp4"))

    def test_non_mp4_raises(self, tmp_path):
        f = tmp_path / "video.avi"
        f.touch()
        with pytest.raises(ValueError, match="Input file must be an MP4"):
            VideoConverter.validate_input(str(f))

    def test_uppercase_mp4_extension_accepted(self, tmp_path):
        """validate_input lowercases the path, so .MP4 is treated as .mp4."""
        f = tmp_path / "VIDEO.MP4"
        f.touch()
        result = VideoConverter.validate_input(str(f))
        assert result == f


class TestGenerateOutputPath:
    def test_replaces_extension(self):
        result = VideoConverter.generate_output_path("/videos/clip.mp4", "mov")
        assert result == "/videos/clip.mov"

    def test_lowercase_format(self):
        result = VideoConverter.generate_output_path("/tmp/v.mp4", "AVI")
        assert result.endswith(".avi")

    def test_gif_output(self):
        result = VideoConverter.generate_output_path("/tmp/screen.mp4", "gif")
        assert result == "/tmp/screen.gif"


class TestBuildFfmpegCommand:
    def test_gif_command_structure(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.touch()
        cmd = VideoConverter.build_ffmpeg_command(str(f), "/out/v.gif", "gif")
        assert "ffmpeg" in cmd
        assert "-vf" in cmd
        assert "fps=15" in cmd[cmd.index("-vf") + 1]
        assert "/out/v.gif" in cmd

    def test_standard_format_command(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.touch()
        cmd = VideoConverter.build_ffmpeg_command(str(f), "/out/v.mov", "mov")
        assert "ffmpeg" in cmd
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "/out/v.mov" in cmd

    def test_webm_format(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.touch()
        cmd = VideoConverter.build_ffmpeg_command(str(f), "/out/v.webm", "webm")
        assert "/out/v.webm" == cmd[-1]

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.touch()
        with pytest.raises(ValueError, match="Unsupported output format"):
            VideoConverter.build_ffmpeg_command(str(f), "/out/v.xyz", "xyz")

    def test_y_flag_present(self, tmp_path):
        """The -y flag should always be present to overwrite output without prompting."""
        f = tmp_path / "v.mp4"
        f.touch()
        cmd = VideoConverter.build_ffmpeg_command(str(f), "/out/v.avi", "avi")
        assert "-y" in cmd

    def test_all_supported_formats(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.touch()
        for fmt in VideoConverter.SUPPORTED_FORMATS:
            cmd = VideoConverter.build_ffmpeg_command(str(f), f"/out/v.{fmt}", fmt)
            assert len(cmd) > 0


class TestConvertAsync:
    async def test_successful_conversion(self, tmp_path):
        f = tmp_path / "video.mp4"
        f.touch()

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
            result = await VideoConverter.convert(str(f), "avi")

        assert "Successfully converted" in result

    async def test_ffmpeg_failure_raises(self, tmp_path):
        f = tmp_path / "video.mp4"
        f.touch()

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"error output"))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
            with pytest.raises(RuntimeError, match="FFmpeg conversion failed"):
                await VideoConverter.convert(str(f), "avi")

    async def test_ffmpeg_not_found_raises(self, tmp_path):
        f = tmp_path / "video.mp4"
        f.touch()

        with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError)):
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                await VideoConverter.convert(str(f), "avi")

    async def test_convert_validates_input_first(self, tmp_path):
        """convert() should raise before spawning ffmpeg if file is missing."""
        with pytest.raises(ValueError, match="Input file not found"):
            await VideoConverter.convert(str(tmp_path / "ghost.mp4"), "avi")
