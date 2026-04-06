"""
Enumerated types for the TCG Pre-Grader.

Why str + Enum?
- FastAPI/Pydantic serialize str-enums as their string value (e.g., "pokemon")
  rather than the Python repr, which keeps the JSON API clean.
- Query params and form fields are validated against the enum values at the
  FastAPI boundary — unknown card types are rejected before reaching services.
"""

from enum import Enum


class CardType(str, Enum):
    """Supported card game / sport categories.

    Only `pokemon` has a trained model artifact in the MVP. `one_piece` and
    `sports` are registered here so the ModelRegistry, CLI, and API can
    validate and route them without code changes when artifacts are ready.
    """

    pokemon = "pokemon"
    one_piece = "one_piece"
    sports = "sports"
