"""
Deduplicator — cross-run cert number deduplication with atomic persistence.

Why atomic writes?
  A plain file.write() is not crash-safe: if the process is killed mid-write,
  the state file is left partially written and the next run reads corrupt JSON.
  Writing to a temp file then calling os.replace() is atomic on POSIX — the
  rename is a single syscall, so the state file is either the old version or
  the new version, never a partial mix (Req 5.4).

Why an in-memory set + explicit persist()?
  Calling persist() on every mark_seen() would hammer disk I/O during a bulk
  scrape run. The orchestrator calls persist() once at the end of each run,
  giving us O(1) membership checks during the run and a single write on exit.
"""

import json
import os
import tempfile
from pathlib import Path

import structlog

from data_pipeline.config import PipelineSettings

logger = structlog.get_logger(__name__)


class Deduplicator:
    """
    Tracks which PSA cert numbers have already been processed.

    Lifecycle:
        dedup = Deduplicator(settings)
        dedup.load()                        # hydrate from disk
        if not dedup.is_seen(cert):
            dedup.mark_seen(cert, source)   # add to in-memory set
        dedup.persist()                     # flush to disk atomically
    """

    def __init__(self, settings: PipelineSettings) -> None:
        self._path: Path = settings.seen_certs_path
        self._seen: set[str] = set()

    def load(self) -> None:
        """
        Hydrate the in-memory set from the state file on disk.

        Silently starts with an empty set if the file does not exist yet —
        this is the expected state on the very first pipeline run (Req 5.1).
        """
        if not self._path.exists():
            logger.debug("deduplicator_no_state_file", path=str(self._path))
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._seen = set(data.get("seen", []))
            logger.info(
                "deduplicator_loaded",
                path=str(self._path),
                count=len(self._seen),
            )
        except Exception as exc:
            # Fail-open: a corrupt state file should not block the pipeline.
            # Log the error and start fresh — worst case we re-download some certs.
            logger.warning(
                "deduplicator_load_error",
                path=str(self._path),
                error=str(exc),
            )
            self._seen = set()

    def is_seen(self, cert_number: str) -> bool:
        """Return True if cert_number has already been processed (Req 5.2)."""
        return cert_number in self._seen

    def mark_seen(self, cert_number: str, source: str) -> None:
        """
        Add cert_number to the in-memory seen set.

        Logs DEBUG when a cert is marked a second time — this is not an error
        (scrapers can surface the same cert from multiple sources), but it is
        useful signal for diagnosing unexpected duplication (Req 5.3).
        """
        if cert_number in self._seen:
            logger.debug(
                "deduplicator_duplicate_cert",
                cert_number=cert_number,
                source=source,
            )
        self._seen.add(cert_number)

    def persist(self) -> None:
        """
        Flush the in-memory set to disk using an atomic write (Req 5.4).

        Write to a sibling temp file first, then os.replace() to swap it in.
        os.replace() is atomic on POSIX — the state file is never partially
        written from the reader's perspective.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps({"seen": sorted(self._seen)}, indent=2)

        # Write to a temp file in the same directory so os.replace() is a
        # same-filesystem rename (cross-device rename would not be atomic).
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=".seen_certs_tmp_",
            suffix=".json",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path_str, self._path)
            logger.debug(
                "deduplicator_persisted",
                path=str(self._path),
                count=len(self._seen),
            )
        except Exception:
            # Clean up the temp file if the replace failed.
            try:
                os.unlink(tmp_path_str)
            except OSError:
                pass
            raise
