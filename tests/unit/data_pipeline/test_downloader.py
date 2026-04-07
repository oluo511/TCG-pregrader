# Feature: training-data-pipeline
# Tests: 3.2 — ImageDownloader unit tests (Requirements 4.1–4.4)
#
# Magic byte reference:
#   JPEG: FF D8 FF ...
#   PNG:  89 50 4E 47 ...
#   GIF:  47 49 46 38 ... (invalid — must raise InvalidImageError)

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from data_pipeline.config import PipelineSettings
from data_pipeline.downloader import ImageDownloader
from data_pipeline.exceptions import DownloadError, InvalidImageError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(tmp_path: Path) -> PipelineSettings:
    return PipelineSettings(
        psa_api_token=SecretStr("test-token"),
        output_dir=tmp_path / "images",
    )


def _make_response(status_code: int, content: bytes = b"") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100
PNG_BYTES  = b"\x89\x50\x4e\x47" + b"\x00" * 100
GIF_BYTES  = b"\x47\x49\x46\x38" + b"\x00" * 100


# ---------------------------------------------------------------------------
# 3.2a — Skip download and log INFO when file already exists on disk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_skips_download_when_file_exists(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Idempotent restarts: if {cert_number}.jpg already exists, the downloader
    must return the existing path without making any HTTP request (Req 4.1).

    structlog writes to stdout by default in test environments, so we use
    capsys instead of caplog to capture the log output.
    """
    settings = _make_settings(tmp_path)
    downloader = ImageDownloader(settings)

    output_dir = tmp_path / "images"
    output_dir.mkdir(parents=True)
    existing = output_dir / "12345678.jpg"
    existing.write_bytes(JPEG_BYTES)

    with patch.object(downloader._client, "get", new_callable=AsyncMock) as mock_get:
        result = await downloader.download("http://example.com/img.jpg", "12345678", output_dir)

    assert result == existing
    mock_get.assert_not_called()
    captured = capsys.readouterr()
    assert "image_already_exists" in captured.out


# ---------------------------------------------------------------------------
# 3.2b — InvalidImageError raised for random bytes with no valid magic prefix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_magic_bytes_raises_error(tmp_path: Path) -> None:
    """
    Random bytes that don't match JPEG or PNG magic must raise InvalidImageError
    before any file is written to disk (Req 4.3).
    """
    settings = _make_settings(tmp_path)
    downloader = ImageDownloader(settings)
    output_dir = tmp_path / "images"

    random_bytes = b"\x00\x01\x02\x03" + b"\xAB" * 100
    resp = _make_response(200, content=random_bytes)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with patch.object(downloader._client, "get", new_callable=AsyncMock, return_value=resp):
            with pytest.raises(InvalidImageError):
                await downloader.download("http://example.com/img.bin", "99999999", output_dir)

    # Nothing should have been written
    assert not any(output_dir.glob("*")) if output_dir.exists() else True


# ---------------------------------------------------------------------------
# 3.2c — InvalidImageError raised for GIF magic bytes (47 49 46)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gif_magic_bytes_raises_invalid_image_error(tmp_path: Path) -> None:
    """
    GIF is a recognised image format but not accepted by the pipeline — only
    JPEG and PNG are valid slab photo formats (Req 4.3).
    """
    settings = _make_settings(tmp_path)
    downloader = ImageDownloader(settings)
    output_dir = tmp_path / "images"

    resp = _make_response(200, content=GIF_BYTES)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with patch.object(downloader._client, "get", new_callable=AsyncMock, return_value=resp):
            with pytest.raises(InvalidImageError, match="Unrecognised"):
                await downloader.download("http://example.com/img.gif", "11111111", output_dir)


# ---------------------------------------------------------------------------
# 3.2d — Successful JPEG download writes file with cert number as basename
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_jpeg_download_writes_correct_filename(tmp_path: Path) -> None:
    """
    A valid JPEG response must be written to {output_dir}/{cert_number}.jpg.
    The cert number is the canonical key — not the URL filename (Req 4.2).
    """
    settings = _make_settings(tmp_path)
    downloader = ImageDownloader(settings)
    output_dir = tmp_path / "images"

    resp = _make_response(200, content=JPEG_BYTES)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with patch.object(downloader._client, "get", new_callable=AsyncMock, return_value=resp):
            result = await downloader.download("http://example.com/some-random-name.jpg", "87654321", output_dir)

    assert result == output_dir / "87654321.jpg"
    assert result.exists()
    assert result.read_bytes() == JPEG_BYTES


# ---------------------------------------------------------------------------
# 3.2e — Successful PNG download writes file with .png extension
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_png_download_writes_correct_filename(tmp_path: Path) -> None:
    """PNG magic bytes must produce a .png file, not .jpg (Req 4.2)."""
    settings = _make_settings(tmp_path)
    downloader = ImageDownloader(settings)
    output_dir = tmp_path / "images"

    resp = _make_response(200, content=PNG_BYTES)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with patch.object(downloader._client, "get", new_callable=AsyncMock, return_value=resp):
            result = await downloader.download("http://example.com/card.png", "55555555", output_dir)

    assert result == output_dir / "55555555.png"
    assert result.exists()


# ---------------------------------------------------------------------------
# 3.2f — DownloadError raised after exhausting retries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_error_after_exhausting_retries(tmp_path: Path) -> None:
    """
    After max_retries (3) are exhausted on persistent 5xx responses, the
    downloader must raise DownloadError — not silently return None (Req 4.4).
    """
    settings = _make_settings(tmp_path)
    downloader = ImageDownloader(settings)
    output_dir = tmp_path / "images"

    server_error = _make_response(503, content=b"Service Unavailable")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with patch.object(downloader._client, "get", new_callable=AsyncMock, return_value=server_error):
            with pytest.raises(DownloadError):
                await downloader.download("http://example.com/img.jpg", "77777777", output_dir)
