"""
ModelRegistry: maps CardType → loaded TF SavedModel artifact.

Why a dedicated registry instead of a plain dict in app state?
- Centralizes load/get logic so the FastAPI lifespan, CLI, and tests all
  share the same error contract (ModelNotFoundError with a consistent message).
- The `is_ready` flag gives the /ready health probe a single authoritative
  source of truth — no scattered boolean flags across app state.
- Isolating TF I/O here makes it trivial to mock in unit/property tests
  without patching deep into service code.

Technical Debt: Eager loading of all enabled card types at startup means
memory scales linearly with the number of types. At scale, replace with
lazy loading (load on first request, LRU eviction) or delegate to a
dedicated model server (TF Serving / Triton) over gRPC.
"""

from pathlib import Path
from typing import Any

from pregrader.enums import CardType
from pregrader.exceptions import ModelNotFoundError
from pregrader.logging_config import get_logger

# Import TF lazily inside methods so the registry module can be imported
# in test environments that mock tf.saved_model.load without triggering
# the full TF initialization at import time.
import tensorflow as tf

logger = get_logger(service="registry")


class ModelRegistry:
    """Keyed store of loaded TF SavedModel artifacts.

    Lifecycle:
      1. Instantiate once at application startup.
      2. Call `load()` for each enabled card type (FastAPI lifespan / CLI init).
      3. Pass the registry instance to GraderService via dependency injection.
      4. GraderService calls `get()` per request — raises ModelNotFoundError
         (→ HTTP 404) if the requested type was never loaded.
    """

    def __init__(self) -> None:
        # dict[CardType, Any] — value is Any because TF SavedModel objects
        # don't expose a clean public type stub in the tensorflow package.
        self._models: dict[CardType, Any] = {}

        # Track whether any load() call encountered an error. is_ready is
        # False if even one load failed, so the /ready probe returns 503
        # rather than silently serving requests with a partial registry.
        self._load_error: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, card_type: CardType, artifact_path: Path) -> None:
        """Load a SavedModel artifact and register it under card_type.

        Why check Path.exists() before calling tf.saved_model.load?
        - TF's error message for a missing path is cryptic and doesn't
          include the card_type context. Failing fast here gives operators
          a clear, actionable error at startup rather than a TF stack trace.

        Args:
            card_type: The CardType key to register the model under.
            artifact_path: Filesystem path to the TF SavedModel directory.

        Raises:
            ModelNotFoundError: If artifact_path does not exist on disk.
        """
        # Guard: artifact must exist before we attempt to load it.
        if not artifact_path.exists():
            self._load_error = True
            msg = (
                f"No model loaded for card_type='{card_type.value}'. "
                f"Loaded types: {[t.value for t in self._models]}"
            )
            logger.critical(
                "model_artifact_not_found",
                card_type=card_type.value,
                artifact_path=str(artifact_path),
            )
            raise ModelNotFoundError(msg)

        # Load the SavedModel — this is the expensive I/O call that should
        # happen exactly once per card type at startup, not per request.
        model = tf.saved_model.load(str(artifact_path))
        self._models[card_type] = model

        logger.info(
            "model_loaded",
            card_type=card_type.value,
            artifact_path=str(artifact_path),
        )

    def get(self, card_type: CardType) -> Any:
        """Return the loaded model for card_type.

        Args:
            card_type: The CardType to look up.

        Returns:
            The loaded TF SavedModel object.

        Raises:
            ModelNotFoundError: If no model has been loaded for card_type.
                Message always includes the requested card_type value and
                the list of currently loaded types so the caller (and the
                HTTP 404 response body) has full diagnostic context.
        """
        if card_type not in self._models:
            msg = (
                f"No model loaded for card_type='{card_type.value}'. "
                f"Loaded types: {[t.value for t in self._models]}"
            )
            logger.critical(
                "model_not_found",
                card_type=card_type.value,
                loaded_types=[t.value for t in self._models],
            )
            raise ModelNotFoundError(msg)

        return self._models[card_type]

    @property
    def is_ready(self) -> bool:
        """True when at least one model is loaded and no load errors occurred.

        Why both conditions?
        - An empty registry (no loads attempted yet) must return False so
          the /ready probe returns 503 during the startup window.
        - A partial load failure (one type failed, another succeeded) must
          also return False — a degraded registry is not considered ready,
          because the failed type would return 404 on any request for it.
        """
        return len(self._models) > 0 and not self._load_error
