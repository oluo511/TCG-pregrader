"""
Unit test configuration.

Why patch load_settings here instead of in each test?
The FastAPI lifespan calls load_settings() at TestClient startup. Without a
real .env file in the test environment, PregraderSettings raises a
ValidationError on the required pokemon_model_artifact_path field — before
any dependency_overrides are applied.

Patching at the conftest level with autouse=True means every test in this
directory gets a pre-configured settings mock without any per-test boilerplate.
The patch targets pregrader.api.app.load_settings (the name as imported in
app.py), not the source module, which is the correct mock target.

We also patch ModelRegistry in the lifespan so it never attempts to call
tf.saved_model.load() during tests — that would require real artifact files
and a GPU/CPU TF environment.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pregrader.enums import CardType


@pytest.fixture(autouse=True)
def mock_load_settings():
    """Provide a valid PregraderSettings mock and a no-op ModelRegistry for
    all unit tests.

    Prevents the FastAPI lifespan from:
      1. Calling the real load_settings() (requires POKEMON_MODEL_ARTIFACT_PATH)
      2. Calling registry.load() → tf.saved_model.load() (requires real artifacts)
    """
    settings = MagicMock()
    settings.pokemon_model_artifact_path = Path("/fake/pokemon_model")
    settings.one_piece_model_artifact_path = None
    settings.sports_model_artifact_path = None
    settings.enabled_card_types = [CardType.pokemon]
    settings.max_batch_size = 50
    settings.input_width = 224
    settings.input_height = 312
    settings.log_level = "INFO"

    mock_registry = MagicMock()
    mock_registry.is_ready = True

    # Patch in both the API app module and the CLI module so all unit tests
    # that construct a TestClient or invoke the CLI get the same mock.
    with (
        patch("pregrader.api.app.load_settings", return_value=settings),
        patch("pregrader.api.app.ModelRegistry", return_value=mock_registry),
        patch("pregrader.cli.load_settings", return_value=settings),
    ):
        yield settings
