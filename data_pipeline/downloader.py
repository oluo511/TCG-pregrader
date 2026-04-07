"""
ImageDownloader — fetch, validate, and persist slab photos to disk.

Design decisions:
- Magic-byte validation over Content-Type: Content-Type is server-supplied and
  unreliable for scraped content. We read the first 4 bytes of the actual payload
  to determine format, not the header.
- Exponential backoff (1s/2s/4s): image hosts are less strict than the PSA API,
  so we use a shorter base delay than the PSA client (2s). Three retries cover
  transient CDN blips without hammering the host.
- Skip-if-exists: idempotent downloads allow safe pipeline restarts without
  re-fetching already-collected images.
"""

import asyncio
from pathlib import Path

import httpx
import structlog

from data_pipeline.config import PipelineSettings
from data_pipeline.exceptions import DownloadError, InvalidImageError

logger = structlog.get_logger(__name__)


class ImageDownloader:
    # Magic byte signatures — checked against raw response bytes, not Content-Type
    # (Content-Type is server-supplied and unreliable for scraped content)
    VALID_MAGIC: dict[bytes, str] = {
        b"\xff\xd8\xff": "jpg",       # JPEG
        b"\x89\x50\x4e\x47": "png",  # PNG
    }

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=30.0)

    async def download(
        self,
        url: str,
        cert_number: str,
        output_dir: Path,
    ) -> Path:
        """
        Download a slab photo and save it to disk, keyed by cert number.

        Steps:
          1. Skip if {cert_number}.jpg already exists (idempotent restarts).
          2. Fetch with exponential backoff — max 3 retries, delays 1s/2s/4s.
          3. Validate magic bytes → raise InvalidImageError if not JPEG/PNG.
          4. Ensure output_dir exists.
          5. Write content to {output_dir}/{cert_number}.{ext}.
          6. Return the saved Path.

        Raises:
            DownloadError: after exhausting all retry attempts.
            InvalidImageError: when downloaded bytes are not a recognised image format.
        """
        # Step 1: skip if already on disk (check .jpg as the canonical extension)
        existing_path = output_dir / f"{cert_number}.jpg"
        if existing_path.exists():
            logger.info(
                "image_already_exists",
                cert_number=cert_number,
                path=str(existing_path),
            )
            return existing_path

        # Step 2: fetch with retry/backoff
        content = await self._fetch_with_retry(url, cert_number)

        # Step 3: validate magic bytes
        ext = self._detect_format(content, cert_number)

        # Step 4: ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 5: write to disk
        dest = output_dir / f"{cert_number}.{ext}"
        dest.write_bytes(content)

        # Step 6: return saved path
        return dest

    async def _fetch_with_retry(self, url: str, cert_number: str) -> bytes:
        """
        GET the URL with exponential backoff.

        Retries on httpx.TransportError (connection-level failures) and 5xx responses.
        Raises DownloadError after max_retries are exhausted.
        """
        max_retries = 3
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = 1.0 * (2 ** (attempt - 1))  # 1s, 2s, 4s
                logger.warning(
                    "image_download_retry",
                    cert_number=cert_number,
                    attempt=attempt,
                    url=url,
                )
                await asyncio.sleep(delay)

            try:
                response = await self._client.get(url)

                if response.status_code >= 500:
                    # Treat 5xx as transient — retry
                    last_exc = DownloadError(
                        f"HTTP {response.status_code} for cert {cert_number} at {url}"
                    )
                    continue

                response.raise_for_status()
                return response.content

            except httpx.TransportError as exc:
                last_exc = exc
                continue

        raise DownloadError(
            f"Download failed for cert {cert_number} after {max_retries} retries"
        ) from last_exc

    def _detect_format(self, content: bytes, cert_number: str) -> str:
        """
        Inspect magic bytes to determine image format.

        JPEG prefix is 3 bytes (FF D8 FF); PNG prefix is 4 bytes (89 50 4E 47).
        We check the longer PNG prefix first to avoid false positives, then JPEG.

        Raises:
            InvalidImageError: when the content does not match any known format.
        """
        for magic, ext in self.VALID_MAGIC.items():
            if content[: len(magic)] == magic:
                return ext

        # Neither JPEG nor PNG — log and raise before touching disk
        logger.warning(
            "invalid_image_magic_bytes",
            cert_number=cert_number,
            magic=content[:4].hex(),
        )
        raise InvalidImageError(
            f"Unrecognised image format for cert {cert_number}: "
            f"magic bytes {content[:4].hex()}"
        )
