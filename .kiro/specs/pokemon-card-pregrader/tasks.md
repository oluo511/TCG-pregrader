# Implementation Plan: TCG Pre-Grader (Pokemon MVP)

## Overview

Incremental build-out from project skeleton → data models → services → API/CLI → training pipeline → tests. Each phase wires into the previous so there is no orphaned code. All implementation is Python with Pydantic v2, FastAPI, Typer, TensorFlow, and Hypothesis.

## Tasks

- [x] 1. Project scaffolding and configuration
  - Create package structure: `src/pregrader/`, `tests/unit/`, `tests/property/`
  - Create `pyproject.toml` with dependencies: fastapi, uvicorn, typer, pydantic, pydantic-settings, tensorflow, opencv-python, pillow, structlog, hypothesis, pytest, pytest-asyncio
  - Create `src/pregrader/exceptions.py` — full `PregraderError` hierarchy: `ImageIngestionError` (→ `InvalidImageFormatError`, `ImageResolutionError`, `BatchSizeError`), `PreprocessingError`, `InferenceError` (→ `ModelNotFoundError`), `ConfigurationError`
  - Create `src/pregrader/enums.py` — `CardType(str, Enum)` with `pokemon`, `one_piece`, `sports`
  - Create `src/pregrader/config.py` — `PregraderSettings(BaseSettings)` with all fields from design: model artifact paths, `enabled_card_types`, `input_width`, `input_height`, `max_batch_size`, `api_host`, `api_port`, `log_level`; load from `.env`
  - Create `.env.example` with all required keys
  - Configure `structlog` JSON renderer in `src/pregrader/logging_config.py`
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ]* 1.1 Write property test for configuration validation (Property 16)
    - **Property 16: Configuration validation completeness**
    - **Validates: Requirements 10.2, 10.3**
    - Use `hypothesis` `st.fixed_dictionaries` to generate valid and invalid config dicts; assert `ValidationError` on missing/out-of-range fields, success on valid configs

- [x] 2. Data models (Pydantic schemas)
  - Create `src/pregrader/schemas.py` — all Pydantic v2 models: `Subgrades`, `GradeResult`, `CardRegion`, `PreprocessedCard`, `ManifestRow`, `TrainingConfig`
  - Enforce all field constraints with `Field(ge=..., le=...)` exactly as specified in design
  - Add `model_config = ConfigDict(frozen=True)` to `GradeResult` and `Subgrades` for immutability
  - _Requirements: 4.1, 4.2, 4.4, 7.1, 8.1_

  - [ ]* 2.1 Write property test for GradeResult round-trip serialization (Property 7)
    - **Property 7: GradeResult serialization round-trip**
    - **Validates: Requirements 4.2, 4.3**
    - Use `hypothesis` `st.builds(GradeResult, ...)` with valid field ranges; assert `model == GradeResult.model_validate_json(model.model_dump_json())`

  - [ ]* 2.2 Write property test for GradeResult output validity (Property 6)
    - **Property 6: GradeResult output validity**
    - **Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.4**
    - Generate random in-range field values; assert construction succeeds and all constraints hold; generate out-of-range values and assert `ValidationError`

- [x] 3. Checkpoint — schemas and config
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. ImageIngestionService
  - Create `src/pregrader/services/ingestion.py` — `ImageIngestionService` class
  - Implement `async validate_and_load(files: list[UploadFile]) -> list[tuple[str, bytes]]`
  - Validation order: (1) batch size ≤ `max_batch_size` → `BatchSizeError`; (2) magic bytes check for JPEG (`FF D8 FF`) / PNG (`89 50 4E 47`) → `InvalidImageFormatError`; (3) decode and check resolution ≥ 300×420 → `ImageResolutionError`
  - Log each validation failure with `structlog` including `image_id`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x]* 4.1 Write property test for image format acceptance (Property 1)
    - **Property 1: Image format acceptance**
    - **Validates: Requirements 1.1, 1.4**
    - Use `hypothesis` `st.binary()` with drawn magic byte prefixes; assert accept iff magic bytes match JPEG or PNG signature

  - [x]* 4.2 Write property test for resolution threshold enforcement (Property 2)
    - **Property 2: Resolution threshold enforcement**
    - **Validates: Requirements 1.2, 1.3**
    - Generate synthetic images of random dimensions; assert `ImageResolutionError` when width < 300 or height < 420, acceptance otherwise

  - [x]* 4.3 Write property test for batch size boundary (Property 3)
    - **Property 3: Batch size boundary**
    - **Validates: Requirements 1.5, 1.6**
    - Generate batches of random size 1–100; assert acceptance for ≤ 50, `BatchSizeError` for > 50 before any image is processed

  - [x]* 4.4 Write unit tests for ImageIngestionService
    - Test `BatchSizeError` raised before any file is opened when batch > 50
    - Test `InvalidImageFormatError` with a GIF magic byte payload
    - Test `ImageResolutionError` with a 100×100 JPEG
    - Test successful load returns `(image_id, bytes)` tuples
    - _Requirements: 1.1–1.6_

- [x] 5. PreprocessingService
  - Create `src/pregrader/services/preprocessing.py` — `PreprocessingService` class
  - Implement `preprocess(raw_bytes: bytes) -> PreprocessedCard` with steps: decode → perspective correction (OpenCV contour/homography) → resize to 224×312 → normalize to [0.0, 1.0] → extract 4 `CardRegion` crops
  - On perspective correction failure: log `WARNING` with `image_id` and continue with uncorrected image (do not raise)
  - Region crop definitions: centering (center 80%), corners (4 corner patches), edges (border strips), surface (inner 60%)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 5.1 Write property test for preprocessor output shape invariant (Property 4)
    - **Property 4: Preprocessor output shape invariant**
    - **Validates: Requirements 2.1, 2.2**
    - Generate random valid images of varying sizes; assert `full_tensor` shape is (312, 224, 3) and all values ∈ [0.0, 1.0]

  - [ ]* 5.2 Write property test for card region extraction completeness (Property 5)
    - **Property 5: Card region extraction completeness**
    - **Validates: Requirements 2.5**
    - Generate random valid images; assert `len(regions) == 4` and `{r.name for r in regions} == {"centering", "corners", "edges", "surface"}`

  - [ ]* 5.3 Write unit tests for PreprocessingService
    - Test output tensor shape and pixel range on a known synthetic image
    - Test perspective correction failure path logs WARNING and returns a result (not raises)
    - Test all four region names are present in output
    - _Requirements: 2.1–2.5_

- [x] 6. ModelRegistry
  - Create `src/pregrader/registry.py` — `ModelRegistry` class
  - Implement `load(card_type: CardType, artifact_path: Path) -> None` using `tf.saved_model.load`
  - Implement `get(card_type: CardType) -> tf.saved_model` — raises `ModelNotFoundError` with message `"No model loaded for card_type='{card_type.value}'. Loaded types: {[...]}"` if not present
  - Implement `is_ready` property — `True` when `len(self._models) > 0` and no load errors
  - Log `INFO` on successful load, `CRITICAL` on `ModelNotFoundError`
  - _Requirements: 3.5, 5.5, 5.6, 5.7_

  - [ ]* 6.1 Write property test for registry load-once invariant (Property 9)
    - **Property 9: Registry load-once invariant per card type**
    - **Validates: Requirements 5.6**
    - Mock `tf.saved_model.load`; generate N random requests per card type; assert `load()` called exactly once per type regardless of N

  - [ ]* 6.2 Write property test for unknown card type → ModelNotFoundError (Property 17)
    - **Property 17: Unknown card type returns ModelNotFoundError / HTTP 404**
    - **Validates: Requirements 3.5, 5.7**
    - Generate `CardType` values not present in registry; assert `ModelNotFoundError` raised and message contains the requested `card_type` value

  - [ ]* 6.3 Write unit tests for ModelRegistry
    - Test `ModelNotFoundError` raised with `card_type` in message for unloaded type
    - Test `is_ready` is `False` before any load, `True` after successful load
    - Test `get()` returns the correct model after `load()`
    - _Requirements: 3.5, 5.5, 5.6_

- [x] 7. GraderService
  - Create `src/pregrader/services/grader.py` — `GraderService` class
  - Implement `__init__(self, registry: ModelRegistry, settings: PregraderSettings)`
  - Implement `async predict(cards: list[PreprocessedCard], card_type: CardType) -> list[GradeResult]`
  - Ordinal decoding: model outputs cumulative probs `P(Y ≤ k)` for k=1..9; compute `P(Y=k) = P(Y≤k) - P(Y≤k-1)`; `overall_grade = argmax + 1`; `confidence = max(P(Y=k))`
  - Subgrades: run same ordinal decoding on each of the 4 region crop heads
  - Per-image `InferenceError`: catch, log `ERROR` with `image_id`, skip that card (continue batch)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.5_

  - [ ]* 7.1 Write property test for GradeResult output validity from grader (Property 6 — grader path)
    - **Property 6: GradeResult output validity**
    - **Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.4**
    - Mock model to return random valid cumulative prob tensors; assert all returned `GradeResult` fields satisfy range constraints

  - [ ]* 7.2 Write property test for response cardinality (Property 8)
    - **Property 8: Response cardinality**
    - **Validates: Requirements 5.2, 6.2**
    - Generate batches of N valid `PreprocessedCard` objects (mocked model); assert `len(results) == N`

  - [ ]* 7.3 Write property test for batch partial-failure resilience (Property 10)
    - **Property 10: Batch partial-failure resilience**
    - **Validates: Requirements 6.5**
    - Generate batches of N cards where exactly one raises `InferenceError`; assert N-1 valid results returned and failure is logged

  - [ ]* 7.4 Write unit tests for GraderService
    - Test ordinal decoding produces `overall_grade ∈ {1..10}` and `confidence ∈ [0.0, 1.0]`
    - Test per-image `InferenceError` is caught and remaining cards are processed
    - Test `ModelNotFoundError` propagates when registry has no model for requested type
    - _Requirements: 3.1–3.5, 6.5_

- [x] 8. Checkpoint — core services
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. FastAPI application
  - Create `src/pregrader/api/app.py` — FastAPI app with lifespan context manager
  - Lifespan: iterate `settings.enabled_card_types`, call `registry.load()` for each; set `registry` on app state; return HTTP 503 for `/predict` until `registry.is_ready`
  - Implement `POST /predict` — accept `multipart/form-data` with `files: list[UploadFile]` and `card_type: CardType = CardType.pokemon`; delegate to `ImageIngestionService` → `PreprocessingService` → `GraderService`; return `list[GradeResult]`
  - Implement `GET /health` — always returns `{"status": "ok"}`
  - Implement `GET /ready` — returns `{"status": "ready"}` if `registry.is_ready`, else HTTP 503
  - Register exception handlers mapping domain exceptions to HTTP codes per design error table
  - All handlers return structured `{"error": str, "type": str}` body — never raw tracebacks
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 9.1 Write unit tests for FastAPI routes
    - Test HTTP 422 for invalid image format, resolution, and batch size > 50
    - Test HTTP 404 with structured body when `card_type` not in registry
    - Test HTTP 500 with structured body for mocked `InferenceError`
    - Test HTTP 503 when `registry.is_ready == False`
    - Test HTTP 200 with `list[GradeResult]` for valid request (mocked services)
    - _Requirements: 5.1–5.7_

- [x] 10. CLI (Typer)
  - Create `src/pregrader/cli.py` — Typer app with `predict` command
  - Command signature: `predict(images: list[Path], card_type: CardType = CardType.pokemon, output: Optional[Path] = None)`
  - Validate each path exists; print descriptive error to stderr and `raise typer.Exit(code=1)` for missing files
  - Reuse `ImageIngestionService` and `GraderService` directly (no HTTP layer)
  - Per-image `InferenceError`: log to stderr, continue remaining images
  - If `card_type` not in registry: print descriptive error to stderr, exit non-zero
  - Output: write JSON array to `output` file if flag set, else print to stdout
  - Register CLI entry point in `pyproject.toml` as `pregrader`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 10.1 Write unit tests for CLI
    - Test exit code 1 and stderr message for non-existent file path
    - Test exit code 1 and stderr message when `--card-type` has no loaded model
    - Test `--output` flag writes valid JSON to file
    - Test stdout output for valid single-image prediction (mocked grader)
    - _Requirements: 6.1–6.6_

- [x] 11. Checkpoint — API and CLI
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Training pipeline — ManifestLoader and DatasetBuilder
  - Create `src/pregrader/training/manifest.py` — `ManifestLoader` class
  - Implement `load(csv_path: Path) -> list[ManifestRow]`: parse CSV, validate each row as `ManifestRow`; skip rows with missing image files (log `WARNING` with row index and path); raise `ValidationError` and halt on any grade value outside 1–10
  - Create `src/pregrader/training/dataset.py` — `DatasetBuilder` class
  - Implement `build(rows: list[ManifestRow], config: TrainingConfig) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]` — returns (train, val, test) splits
  - Split logic: shuffle with fixed seed, slice by `train_ratio` / `val_ratio` / remainder; splits must be pairwise disjoint
  - Log dataset statistics (grade class distribution, total count) via `structlog` before returning
  - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.6_

  - [ ]* 12.1 Write property test for manifest missing-file skip (Property 11)
    - **Property 11: Manifest missing-file skip**
    - **Validates: Requirements 7.2**
    - Generate manifests of M rows where K rows reference non-existent paths; assert loaded dataset has exactly M-K samples

  - [ ]* 12.2 Write property test for dataset split partition invariant (Property 12)
    - **Property 12: Dataset split partition invariant**
    - **Validates: Requirements 7.5**
    - Generate datasets of random size N with valid split ratios; assert splits are pairwise disjoint, union equals full dataset, sizes approximate configured ratios within rounding tolerance

  - [ ]* 12.3 Write unit tests for ManifestLoader and DatasetBuilder
    - Test `ValidationError` raised and halted on manifest row with `overall_grade = 0`
    - Test missing-file rows are skipped with WARNING, not raised
    - Test dataset statistics are logged before return
    - _Requirements: 7.1–7.3, 7.5, 7.6_

- [x] 13. Training pipeline — AugmentationPipeline
  - Create `src/pregrader/training/augmentation.py` — `AugmentationPipeline` class
  - Implement `apply(image_tensor: tf.Tensor) -> tf.Tensor` with: random horizontal flip, random brightness ±20%, random rotation ±5°
  - Each transform applied independently via `tf.image` ops
  - _Requirements: 7.4_

  - [ ]* 13.1 Write property test for augmentation non-determinism (Property 13)
    - **Property 13: Augmentation non-determinism**
    - **Validates: Requirements 7.4**
    - Generate random input images; apply augmentation 100 times; assert ≥ 90 of 100 outputs differ from the original tensor

- [x] 14. Training pipeline — TrainingLoop and Evaluator
  - Create `src/pregrader/training/trainer.py` — `TrainingLoop` class
  - Implement `train(train_ds, val_ds, config: TrainingConfig) -> Path` — builds EfficientNetB0 backbone (ImageNet weights) + ordinal regression head (9 sigmoid outputs for cumulative probs); compiles with CORN loss or binary cross-entropy per threshold; logs loss/accuracy/MAE per epoch; saves `tf.saved_model` to `config.output_dir`; returns artifact path
  - Create `src/pregrader/training/evaluator.py` — `Evaluator` class
  - Implement `evaluate(model, test_ds, output_dir: Path) -> dict` — computes MAE, ±1 accuracy, 10×10 confusion matrix; writes JSON report to `output_dir / "eval_report.json"`; returns metrics dict
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.3, 9.4_

  - [ ]* 14.1 Write property test for MAE metric correctness (Property 14)
    - **Property 14: MAE metric correctness**
    - **Validates: Requirements 9.1**
    - Generate random lists of (predicted, actual) integer pairs; assert `Evaluator._compute_mae(pairs) == mean(|p-a|)` computed independently with numpy

  - [ ]* 14.2 Write property test for within-one accuracy metric correctness (Property 15)
    - **Property 15: Within-one accuracy metric correctness**
    - **Validates: Requirements 9.2**
    - Generate random (predicted, actual) pairs; assert `Evaluator._compute_within_one(pairs) == count(|p-a| ≤ 1) / total` computed independently

  - [ ]* 14.3 Write unit tests for TrainingLoop and Evaluator
    - Test SavedModel artifact exists at configured output path after training on a tiny synthetic dataset
    - Test evaluation JSON report is written and parseable
    - Test confusion matrix is 10×10
    - _Requirements: 8.4, 9.3, 9.4_

- [x] 15. Checkpoint — training pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Wire everything together
  - Create `src/pregrader/__init__.py` and `src/pregrader/services/__init__.py` — expose public API
  - Create `src/pregrader/api/dependencies.py` — FastAPI dependency injection for `ModelRegistry`, `PregraderSettings`, `ImageIngestionService`, `PreprocessingService`, `GraderService`
  - Create `src/pregrader/training/__init__.py` — expose `ManifestLoader`, `DatasetBuilder`, `AugmentationPipeline`, `TrainingLoop`, `Evaluator`
  - Verify CLI entry point `pregrader predict` resolves correctly via `pyproject.toml`
  - Verify `uvicorn src.pregrader.api.app:app` starts and `/health` returns 200
  - _Requirements: 5.1, 6.1, 10.1_

- [x] 17. Final checkpoint — full integration
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use `@settings(max_examples=100)` and the `"ci"` Hypothesis profile
- Each property test file includes a comment: `# Feature: pokemon-card-pregrader, Property N: <title>`
- The 300-line rule applies: `grader.py`, `trainer.py`, and `app.py` are the most likely candidates — flag for modularization if they approach the limit
- `PreprocessedCard` tensors are passed as `np.ndarray` between services in the hot path; Pydantic models serve as contract definitions only
