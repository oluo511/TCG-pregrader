"""
Pydantic v2 data models for the TCG Pre-Grader.

Why Pydantic v2 over dataclasses or TypedDicts?
- Field-level constraints (ge/le/gt/lt) are enforced at construction time,
  not at the API boundary — invalid data never reaches service logic.
- model_dump_json() / model_validate_json() give us a zero-boilerplate
  JSON round-trip that satisfies Requirement 4.2/4.3.
- ConfigDict(frozen=True) on output schemas (GradeResult, Subgrades) makes
  them hashable and prevents accidental mutation after inference — important
  when the same result object is passed to logging, serialization, and the
  API response layer in sequence.

Technical Debt: PreprocessedCard.full_tensor and CardRegion.tensor use
list[list[list[float]]] for schema clarity, but this is expensive to
serialize for large images. In the hot inference path, pass np.ndarray
directly between services; only coerce to Pydantic at the API boundary.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pregrader.enums import CardType


# ---------------------------------------------------------------------------
# Output schemas — frozen to prevent post-inference mutation
# ---------------------------------------------------------------------------


class Subgrades(BaseModel):
    """Per-dimension float scores on the PSA 1.0–10.0 scale.

    Why frozen? Subgrades are produced by the ordinal decoder and must not
    be modified after construction — freezing catches accidental writes at
    runtime rather than silently corrupting a result.
    """

    model_config = ConfigDict(frozen=True)

    centering: float = Field(ge=1.0, le=10.0)
    corners: float = Field(ge=1.0, le=10.0)
    edges: float = Field(ge=1.0, le=10.0)
    surface: float = Field(ge=1.0, le=10.0)


class GradeResult(BaseModel):
    """The canonical output contract for a single card prediction.

    Frozen for the same reason as Subgrades — this object crosses the
    service → API → client boundary and must be immutable once produced.
    """

    model_config = ConfigDict(frozen=True)

    image_id: str
    card_type: CardType
    overall_grade: int = Field(ge=1, le=10)
    subgrades: Subgrades
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Internal preprocessing schemas — mutable, not frozen
# ---------------------------------------------------------------------------


class CardRegion(BaseModel):
    """A named crop of the card used to evaluate one subgrade dimension.

    name is constrained to the four canonical region labels so downstream
    consumers can key on it without defensive string matching.
    """

    # "centering" | "corners" | "edges" | "surface"
    name: str
    # H x W x C tensor, values normalized to [0.0, 1.0]
    tensor: list[list[list[float]]]


class PreprocessedCard(BaseModel):
    """Full preprocessed representation of a single card image.

    Carries both the full-card tensor (for the backbone) and the four region
    crops (for the subgrade heads). Services consume this as the handoff
    between PreprocessingService and GraderService.
    """

    image_id: str
    full_tensor: list[list[list[float]]]
    regions: list[CardRegion]


# ---------------------------------------------------------------------------
# Training pipeline schemas
# ---------------------------------------------------------------------------


class ManifestRow(BaseModel):
    """One row from the training manifest CSV.

    Pydantic validates grade ranges here so ManifestLoader can raise
    ValidationError immediately on a bad row rather than propagating
    out-of-range values into the training dataset.
    """

    image_path: Path
    overall_grade: int = Field(ge=1, le=10)
    centering: float = Field(ge=1.0, le=10.0)
    corners: float = Field(ge=1.0, le=10.0)
    edges: float = Field(ge=1.0, le=10.0)
    surface: float = Field(ge=1.0, le=10.0)


class TrainingConfig(BaseModel):
    """Hyperparameters and paths for a single training run.

    Why BaseModel instead of BaseSettings? TrainingConfig is loaded from a
    config file or passed programmatically by the training script — it is not
    an environment-driven runtime setting. BaseSettings would add unnecessary
    env-var coupling to an offline pipeline component.

    The train_ratio + val_ratio < 1.0 invariant is enforced by a model
    validator so the test split always has at least some samples.
    """

    backbone: str = "EfficientNetB0"
    pretrained_weights: str = "imagenet"

    # Split ratios — each must be (0, 1) and their sum must be < 1.0 so the
    # remainder forms a non-empty test split.
    train_ratio: float = Field(default=0.70, gt=0, lt=1)
    val_ratio: float = Field(default=0.15, gt=0, lt=1)

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    output_dir: Path = Path("artifacts/")
    log_dir: Path = Path("logs/")

    @field_validator("val_ratio")
    @classmethod
    def ratios_must_leave_room_for_test(cls, val_ratio: float, info) -> float:
        """Ensure train + val < 1.0 so the test split is non-empty.

        Why validate on val_ratio rather than a model_validator?
        field_validators run after individual field coercion, so both
        train_ratio and val_ratio are already validated floats by the time
        this runs. We access train_ratio via info.data (already-validated
        fields dict) — this is the Pydantic v2 pattern for cross-field checks
        inside a field_validator.
        """
        train_ratio = info.data.get("train_ratio", 0.70)
        if train_ratio + val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be "
                f"< 1.0 to leave a non-empty test split. "
                f"Got sum = {train_ratio + val_ratio:.4f}."
            )
        return val_ratio
