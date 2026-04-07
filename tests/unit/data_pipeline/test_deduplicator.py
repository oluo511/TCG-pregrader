# Feature: training-data-pipeline
# Tests: 6.2 — Deduplicator unit tests (Requirements 5.1–5.4)
#
# Why capsys instead of caplog for structlog assertions?
#   structlog writes to stdout by default in test environments (no logging
#   integration configured). caplog only captures Python's stdlib logging
#   module — it will never see structlog output. capsys captures stdout/stderr
#   directly, which is where structlog events land.

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from data_pipeline.config import PipelineSettings
from data_pipeline.deduplicator import Deduplicator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(tmp_path: Path) -> PipelineSettings:
    """Minimal settings pointing the state file at a temp directory."""
    return PipelineSettings(
        psa_api_token=SecretStr("test-token"),
        seen_certs_path=tmp_path / ".seen_certs.json",
    )


def _write_state(path: Path, certs: list[str]) -> None:
    """Write a valid state file with the given cert numbers."""
    path.write_text(json.dumps({"seen": certs}), encoding="utf-8")


# ---------------------------------------------------------------------------
# 6.2a — load() populates set from existing state file (Req 5.1)
# ---------------------------------------------------------------------------

def test_load_populates_set_from_state_file(tmp_path: Path) -> None:
    """
    On startup the deduplicator must hydrate its in-memory set from the JSON
    state file so certs processed in previous runs are not re-downloaded.
    """
    settings = _make_settings(tmp_path)
    _write_state(settings.seen_certs_path, ["11111111", "22222222", "33333333"])

    dedup = Deduplicator(settings)
    dedup.load()

    assert dedup.is_seen("11111111")
    assert dedup.is_seen("22222222")
    assert dedup.is_seen("33333333")
    assert not dedup.is_seen("99999999")


def test_load_starts_empty_when_no_state_file(tmp_path: Path) -> None:
    """
    First-run behaviour: no state file → empty set, no error raised (Req 5.1).
    """
    settings = _make_settings(tmp_path)
    # Deliberately do NOT create the state file.

    dedup = Deduplicator(settings)
    dedup.load()  # must not raise

    assert not dedup.is_seen("12345678")


# ---------------------------------------------------------------------------
# 6.2b — persist() writes atomically via temp file + os.replace (Req 5.4)
# ---------------------------------------------------------------------------

def test_persist_uses_atomic_replace(tmp_path: Path) -> None:
    """
    persist() must call os.replace() to swap in the new state file atomically.
    We patch os.replace to intercept the call and verify:
      1. It was called exactly once.
      2. The source (tmp) and destination (final path) are correct.
      3. The temp file is a sibling of the final path (same directory → same
         filesystem → rename is atomic on POSIX).
    """
    settings = _make_settings(tmp_path)
    dedup = Deduplicator(settings)
    dedup.mark_seen("12345678", source="ebay")

    replace_calls: list[tuple[str, str]] = []
    original_replace = os.replace

    def capturing_replace(src: str, dst: str) -> None:
        replace_calls.append((src, dst))
        original_replace(src, dst)  # still perform the actual rename

    with patch("data_pipeline.deduplicator.os.replace", side_effect=capturing_replace):
        dedup.persist()

    assert len(replace_calls) == 1
    tmp_src, final_dst = replace_calls[0]

    # Temp file must be in the same directory as the final state file.
    assert Path(tmp_src).parent == settings.seen_certs_path.parent
    assert Path(final_dst) == settings.seen_certs_path

    # The final file must exist and contain the correct data.
    data = json.loads(settings.seen_certs_path.read_text())
    assert "12345678" in data["seen"]


def test_persist_no_partial_file_on_disk(tmp_path: Path) -> None:
    """
    After persist() completes, only the final state file should exist —
    no leftover temp files in the directory (Req 5.4).
    """
    settings = _make_settings(tmp_path)
    dedup = Deduplicator(settings)
    dedup.mark_seen("55555555", source="cardladder")
    dedup.persist()

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0] == settings.seen_certs_path


# ---------------------------------------------------------------------------
# 6.2c — mark_seen logs DEBUG on duplicate cert number (Req 5.3)
# ---------------------------------------------------------------------------

def test_mark_seen_logs_debug_on_duplicate(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    When the same cert number is marked twice, a DEBUG log event must be emitted
    so operators can diagnose unexpected duplication across scrapers.

    structlog writes to stdout — we use capsys, not caplog (see module docstring).
    """
    settings = _make_settings(tmp_path)
    dedup = Deduplicator(settings)

    dedup.mark_seen("12345678", source="ebay")
    dedup.mark_seen("12345678", source="cardladder")  # duplicate → DEBUG log

    captured = capsys.readouterr()
    assert "deduplicator_duplicate_cert" in captured.out
    assert "12345678" in captured.out


def test_mark_seen_no_debug_log_on_first_occurrence(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    The first time a cert is marked, no duplicate-warning log should appear.
    Only the second (and subsequent) calls should trigger the DEBUG event.
    """
    settings = _make_settings(tmp_path)
    dedup = Deduplicator(settings)

    dedup.mark_seen("99999999", source="ebay")

    captured = capsys.readouterr()
    assert "deduplicator_duplicate_cert" not in captured.out


# ---------------------------------------------------------------------------
# 6.2d — Incremental run: old certs preserved after load → mark → persist (Req 5.1, 5.3, 5.4)
# ---------------------------------------------------------------------------

def test_incremental_run_preserves_old_certs(tmp_path: Path) -> None:
    """
    Simulates a second pipeline run:
      Run 1 leaves ["11111111", "22222222"] in the state file.
      Run 2 loads that state, marks a new cert "33333333", and persists.
      The final state file must contain all three certs — old ones must not
      be lost (Req 5.1, 5.4).
    """
    settings = _make_settings(tmp_path)

    # --- Simulate Run 1 output ---
    _write_state(settings.seen_certs_path, ["11111111", "22222222"])

    # --- Run 2 ---
    dedup = Deduplicator(settings)
    dedup.load()

    # Existing certs must be visible immediately after load.
    assert dedup.is_seen("11111111")
    assert dedup.is_seen("22222222")

    dedup.mark_seen("33333333", source="ebay")
    dedup.persist()

    # Re-read from disk to confirm durability.
    data = json.loads(settings.seen_certs_path.read_text())
    persisted = set(data["seen"])

    assert "11111111" in persisted
    assert "22222222" in persisted
    assert "33333333" in persisted
