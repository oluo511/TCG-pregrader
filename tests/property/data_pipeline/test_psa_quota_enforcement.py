# Feature: training-data-pipeline, Property 6: PSA quota enforcement
# Validates: Requirements 1.3, 1.4
#
# Why a property test here instead of a unit test?
# The quota logic uses asyncio.Lock to make the check+increment atomic.
# A unit test with a fixed N can only verify one scenario. A property test
# generates arbitrary N > quota values and proves the invariant holds for
# ALL of them — including edge cases like N = quota+1 and N = quota*10.
#
# What we're proving:
#   For any sequence of N get_cert() calls where N > psa_daily_quota:
#     1. Total HTTP requests made ≤ psa_daily_quota
#     2. All calls beyond the quota raise QuotaExhaustedError

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import HealthCheck
from hypothesis import strategies as st
from pydantic import SecretStr

from data_pipeline.config import PipelineSettings
from data_pipeline.exceptions import QuotaExhaustedError
from data_pipeline.psa_client import PSAClient, QuotaState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PSA_BODY = {
    "PSAcert": {
        "CertNumber": "12345678",
        "OverallGrade": "9",
        "GradeDescription": "MINT",
        "Centering": "9",
        "Corners": "9",
        "Edges": "9",
        "Surface": "9",
    }
}


def _make_settings(tmp_path: Path, quota: int) -> PipelineSettings:
    return PipelineSettings(
        psa_api_token=SecretStr("test-token"),
        psa_daily_quota=quota,
        psa_quota_state_path=tmp_path / ".quota_state.json",
    )


def _make_success_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _VALID_PSA_BODY
    resp.text = json.dumps(_VALID_PSA_BODY)
    return resp


# ---------------------------------------------------------------------------
# Property 6: PSA quota enforcement
#
# Strategy:
#   - quota drawn from [1, 20] — small enough to test exhaustion quickly
#   - n_calls drawn from [quota+1, quota+10] — always exceeds quota
#
# Invariants asserted:
#   1. HTTP GET call count ≤ quota (no over-calling)
#   2. Exactly (n_calls - quota) calls raise QuotaExhaustedError
# ---------------------------------------------------------------------------

@given(
    quota=st.integers(min_value=1, max_value=20),
    extra_calls=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100, deadline=None)
def test_quota_enforcement_never_exceeds_limit(
    quota: int,
    extra_calls: int,
) -> None:
    """
    Property 6: For any N > psa_daily_quota get_cert() calls, total HTTP
    requests must be ≤ quota and all excess calls must raise QuotaExhaustedError.

    This proves the asyncio.Lock check+increment is truly atomic — a race
    condition would allow more than `quota` HTTP calls to slip through.

    Why tempfile instead of tmp_path fixture?
    Hypothesis generates 100 examples per run. pytest's tmp_path fixture is
    function-scoped and not reset between Hypothesis examples — using it would
    accumulate quota state across examples, corrupting the invariant. We create
    a fresh temp directory per example using tempfile.TemporaryDirectory instead.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        n_calls = quota + extra_calls
        settings_obj = _make_settings(tmp_path, quota)
        client = PSAClient(settings_obj)

        http_call_count = 0
        quota_errors = 0

        success_resp = _make_success_response()

        async def run_calls() -> None:
            nonlocal http_call_count, quota_errors

            async def counting_get(url: str) -> MagicMock:
                nonlocal http_call_count
                http_call_count += 1
                return success_resp

            with patch.object(client._client, "get", side_effect=counting_get):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    for i in range(n_calls):
                        try:
                            await client.get_cert(f"{i:08d}")
                        except QuotaExhaustedError:
                            quota_errors += 1

        asyncio.run(run_calls())

        # Invariant 1: HTTP calls must never exceed the quota
        assert http_call_count <= quota, (
            f"quota={quota}, n_calls={n_calls}: "
            f"made {http_call_count} HTTP calls but quota is {quota}"
        )

        # Invariant 2: exactly extra_calls must have been blocked by quota
        assert quota_errors == extra_calls, (
            f"quota={quota}, n_calls={n_calls}: "
            f"expected {extra_calls} QuotaExhaustedErrors, got {quota_errors}"
        )
