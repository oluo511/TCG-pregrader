"""
# Feature: pokemon-card-pregrader, Property 7: GradeResult serialization round-trip
# Feature: pokemon-card-pregrader, Property 6: GradeResult output validity

Property 7: For any valid GradeResult object, serializing it to JSON and then
deserializing it must produce an object equal to the original.

Property 6: For any valid GradeResult constructed from in-range field values,
all constraints must hold; constructing from out-of-range values must raise
ValidationError.

Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from pregrader.enums import CardType
from pregrader.schemas import GradeResult, Subgrades

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid float subgrade in [1.0, 10.0]
_subgrade_float = st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Valid Subgrades instance
_valid_subgrades = st.builds(
    Subgrades,
    centering=_subgrade_float,
    corners=_subgrade_float,
    edges=_subgrade_float,
    surface=_subgrade_float,
)

# Valid GradeResult instance
_valid_grade_result = st.builds(
    GradeResult,
    image_id=st.text(min_size=1, max_size=64, alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00")),
    card_type=st.sampled_from(list(CardType)),
    overall_grade=st.integers(min_value=1, max_value=10),
    subgrades=_valid_subgrades,
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)

# Out-of-range float for subgrades — below 1.0 or above 10.0
_invalid_subgrade_float = st.one_of(
    st.floats(max_value=0.9999, allow_nan=False, allow_infinity=False),
    st.floats(min_value=10.0001, allow_nan=False, allow_infinity=False),
).filter(lambda x: not (1.0 <= x <= 10.0))

# Out-of-range overall_grade — below 1 or above 10
_invalid_overall_grade = st.one_of(
    st.integers(max_value=0),
    st.integers(min_value=11),
)

# Out-of-range confidence — below 0.0 or above 1.0
_invalid_confidence = st.one_of(
    st.floats(max_value=-0.0001, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.0001, allow_nan=False, allow_infinity=False),
).filter(lambda x: not (0.0 <= x <= 1.0))


# ---------------------------------------------------------------------------
# Property 7: GradeResult serialization round-trip
# ---------------------------------------------------------------------------


@given(_valid_grade_result)
def test_grade_result_json_round_trip(result: GradeResult) -> None:
    """
    **Validates: Requirements 4.2, 4.3**

    Serializing a GradeResult to JSON and deserializing it must produce an
    object equal to the original. This guards against lossy serialization
    (e.g., float precision truncation, enum value mangling) that would break
    downstream consumers parsing the API response.
    """
    # model_dump_json() uses Pydantic's fast JSON encoder; model_validate_json()
    # runs full validation on the deserialized dict — so this also confirms
    # the round-tripped object satisfies all field constraints.
    json_str = result.model_dump_json()
    reconstructed = GradeResult.model_validate_json(json_str)
    assert result == reconstructed


# ---------------------------------------------------------------------------
# Property 6: GradeResult output validity — in-range construction succeeds
# ---------------------------------------------------------------------------


@given(_valid_grade_result)
def test_valid_grade_result_satisfies_all_constraints(result: GradeResult) -> None:
    """
    **Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.4**

    Any GradeResult built from in-range values must satisfy every field
    constraint. This is the positive case: valid inputs → valid object.
    """
    assert 1 <= result.overall_grade <= 10
    assert 0.0 <= result.confidence <= 1.0
    assert 1.0 <= result.subgrades.centering <= 10.0
    assert 1.0 <= result.subgrades.corners <= 10.0
    assert 1.0 <= result.subgrades.edges <= 10.0
    assert 1.0 <= result.subgrades.surface <= 10.0
    assert result.card_type in CardType


# ---------------------------------------------------------------------------
# Property 6: GradeResult output validity — out-of-range construction raises
# ---------------------------------------------------------------------------


@given(_invalid_overall_grade)
def test_out_of_range_overall_grade_raises_validation_error(grade: int) -> None:
    """
    **Validates: Requirements 4.4**

    overall_grade outside [1, 10] must raise ValidationError at construction
    time — never silently clamp or accept an invalid grade.
    """
    valid_subgrades = Subgrades(centering=5.0, corners=5.0, edges=5.0, surface=5.0)
    with pytest.raises(ValidationError):
        GradeResult(
            image_id="test",
            card_type=CardType.pokemon,
            overall_grade=grade,
            subgrades=valid_subgrades,
            confidence=0.9,
        )


@given(_invalid_confidence)
def test_out_of_range_confidence_raises_validation_error(confidence: float) -> None:
    """
    **Validates: Requirements 3.3, 4.4**

    confidence outside [0.0, 1.0] must raise ValidationError. A confidence
    > 1.0 or < 0.0 is not a valid probability and must be caught before the
    result reaches the API response serializer.
    """
    valid_subgrades = Subgrades(centering=5.0, corners=5.0, edges=5.0, surface=5.0)
    with pytest.raises(ValidationError):
        GradeResult(
            image_id="test",
            card_type=CardType.pokemon,
            overall_grade=5,
            subgrades=valid_subgrades,
            confidence=confidence,
        )


@given(_invalid_subgrade_float)
def test_out_of_range_subgrade_raises_validation_error(bad_value: float) -> None:
    """
    **Validates: Requirements 3.2, 4.4**

    Any subgrade field outside [1.0, 10.0] must raise ValidationError.
    We test centering as the representative field — all four share the same
    constraint definition.
    """
    with pytest.raises(ValidationError):
        Subgrades(
            centering=bad_value,
            corners=5.0,
            edges=5.0,
            surface=5.0,
        )
