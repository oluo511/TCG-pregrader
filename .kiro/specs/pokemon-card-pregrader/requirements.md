# Requirements Document

## Introduction

A CNN-based pre-grading system for trading cards (TCG Pre-Grader) that predicts a grade (PSA 1–10 scale) from card photos before the user submits cards for official grading. The primary supported card type is Pokemon; One Piece and sports cards are planned future additions. The system accepts one or more card images, runs them through a trained model for the specified card type, and returns a predicted grade with subgrades (centering, corners, edges, surface) and a confidence score. The architecture must support both local CLI usage for experimentation and a REST API for future web/mobile integration.

## Glossary

- **TCG Pre-Grader**: The end-to-end system that accepts card images and returns predicted grades.
- **Pregrader**: Shorthand for TCG Pre-Grader.
- **Grader**: The CNN inference component responsible for producing grade predictions from preprocessed images.
- **Preprocessor**: The component responsible for image normalization, alignment, and augmentation.
- **GradeResult**: The structured output schema containing predicted grade, subgrades, and confidence.
- **PSA Scale**: The 1–10 integer grading scale used by Professional Sports Authenticator, where 10 is gem mint.
- **Subgrade**: A component score (centering, corners, edges, surface) contributing to the overall grade.
- **Training Pipeline**: The offline data pipeline responsible for sourcing, preprocessing, augmenting, and training the CNN model.
- **Serving Layer**: The runtime component that loads trained model artifacts and exposes predictions via API or CLI.
- **Card Region**: A cropped sub-image of a card used to evaluate a specific subgrade dimension.
- **CardType**: An enumerated value identifying the card game or sport category (`pokemon`, `one_piece`, `sports`).
- **ModelRegistry**: A map of `card_type → loaded model` that routes inference requests to the correct model artifact.

---

## Requirements

### Requirement 1: Image Ingestion

**User Story:** As a collector, I want to submit one or more card photos, so that I can receive a predicted grade without sending cards to a grading service.

#### Acceptance Criteria

1. THE Pregrader SHALL accept card images in JPEG and PNG formats.
2. THE Pregrader SHALL accept images with a minimum resolution of 300×420 pixels.
3. IF a submitted image is below the minimum resolution, THEN THE Pregrader SHALL return a descriptive error identifying the image and the resolution requirement.
4. IF a submitted image is not a valid JPEG or PNG, THEN THE Pregrader SHALL return a descriptive error identifying the file and the accepted formats.
5. THE Pregrader SHALL accept batches of up to 50 images per request.
6. IF a batch exceeds 50 images, THEN THE Pregrader SHALL return an error before processing any images.

---

### Requirement 2: Image Preprocessing

**User Story:** As a system operator, I want card images normalized and aligned before inference, so that the model receives consistent input regardless of photo conditions.

#### Acceptance Criteria

1. WHEN an image is received, THE Preprocessor SHALL resize it to a fixed input resolution of 224×312 pixels.
2. WHEN an image is received, THE Preprocessor SHALL normalize pixel values to the range [0.0, 1.0].
3. WHEN an image is received, THE Preprocessor SHALL apply perspective correction to align card borders to a standard orientation.
4. IF perspective correction fails to detect card borders, THEN THE Preprocessor SHALL log a warning and proceed with the uncorrected image.
5. THE Preprocessor SHALL extract four Card Regions (centering, corners, edges, surface) as separate crops for subgrade inference.

---

### Requirement 3: Grade Prediction

**User Story:** As a collector, I want the system to predict an overall PSA-scale grade and subgrades, so that I can prioritize which cards are worth submitting.

#### Acceptance Criteria

1. WHEN a preprocessed image is provided, THE Grader SHALL return a predicted overall grade on the PSA Scale (integer 1–10).
2. WHEN a preprocessed image is provided, THE Grader SHALL return a predicted Subgrade for each of the four dimensions: centering, corners, edges, and surface.
3. WHEN a preprocessed image is provided, THE Grader SHALL return a confidence score in the range [0.0, 1.0] for the overall grade prediction.
4. THE Grader SHALL complete inference for a single card within 2 seconds on CPU hardware.
5. IF the model artifact for the requested `card_type` is not found in the ModelRegistry, THEN THE Grader SHALL raise a `ModelNotFoundError` that includes the requested `card_type` in its message before accepting any inference requests for that type.

---

### Requirement 4: GradeResult Output Schema

**User Story:** As a developer integrating this system, I want a consistent, typed output schema, so that downstream consumers can reliably parse predictions.

#### Acceptance Criteria

1. THE Pregrader SHALL return a GradeResult containing: overall_grade (int, 1–10), subgrades (centering, corners, edges, surface as floats 1.0–10.0), confidence (float 0.0–1.0), and image_id (str).
2. THE Pregrader SHALL serialize GradeResult to JSON.
3. FOR ALL valid GradeResult objects, serializing then deserializing SHALL produce an equivalent object (round-trip property).
4. IF any predicted field falls outside its valid range, THEN THE Pregrader SHALL raise a ValidationError before returning the result.

---

### Requirement 5: REST API Serving Layer

**User Story:** As a developer, I want a REST API endpoint for grade prediction, so that web and mobile clients can integrate with the Pregrader without running local Python.

#### Acceptance Criteria

1. THE Serving Layer SHALL expose a POST `/predict` endpoint that accepts multipart/form-data with one or more image files and an optional `card_type` parameter (default: `"pokemon"`).
2. WHEN a valid request is received, THE Serving Layer SHALL return a JSON array of GradeResult objects, one per submitted image.
3. THE Serving Layer SHALL return HTTP 422 for validation errors (bad format, resolution, batch size).
4. THE Serving Layer SHALL return HTTP 500 with a structured error body for unexpected inference failures.
5. WHILE the model is loading at startup, THE Serving Layer SHALL return HTTP 503 for any incoming prediction requests.
6. THE Serving Layer SHALL load model artifacts once at startup for all enabled card types and reuse them across requests.
7. IF the requested `card_type` has no loaded model in the ModelRegistry, THEN THE Serving Layer SHALL return HTTP 404 with a descriptive error body that includes the requested `card_type`.

---

### Requirement 6: CLI Interface

**User Story:** As a data scientist, I want a CLI tool to run predictions on local image files, so that I can evaluate model performance without standing up a server.

#### Acceptance Criteria

1. THE Pregrader SHALL provide a CLI command that accepts one or more image file paths as arguments.
2. WHEN the CLI is invoked with valid image paths, THE Pregrader SHALL print a GradeResult as formatted JSON to stdout for each image.
3. WHEN the CLI is invoked with the `--output` flag and a file path, THE Pregrader SHALL write results to that file instead of stdout.
4. IF a provided file path does not exist, THEN THE Pregrader SHALL print a descriptive error to stderr and exit with a non-zero status code.
5. IF inference fails for one image in a batch, THEN THE Pregrader SHALL log the error for that image and continue processing remaining images.
6. THE CLI SHALL accept a `--card-type` flag (default: `"pokemon"`) to specify which model to use for inference; if the requested card type has no loaded model, THE CLI SHALL print a descriptive error to stderr and exit with a non-zero status code.

---

### Requirement 7: Training Data Pipeline

**User Story:** As an ML engineer, I want a reproducible data pipeline for sourcing and preprocessing training data, so that model training is auditable and re-runnable.

#### Acceptance Criteria

1. THE Training Pipeline SHALL accept a manifest CSV with columns: image_path, overall_grade, centering, corners, edges, surface.
2. IF a manifest row references a missing image file, THEN THE Training Pipeline SHALL log a warning and skip that row.
3. IF a manifest row contains a grade value outside the PSA Scale (1–10), THEN THE Training Pipeline SHALL raise a ValidationError and halt.
4. THE Training Pipeline SHALL apply data augmentation including: random horizontal flip, random brightness adjustment (±20%), and random rotation (±5 degrees).
5. THE Training Pipeline SHALL split data into train, validation, and test sets at a configurable ratio (default 70/15/15).
6. THE Training Pipeline SHALL log dataset statistics (class distribution per grade, total sample count) before training begins.

---

### Requirement 8: Model Training

**User Story:** As an ML engineer, I want a configurable training script, so that I can experiment with architectures and hyperparameters without modifying core logic.

#### Acceptance Criteria

1. THE Training Pipeline SHALL support configurable backbone architectures (default: EfficientNetB0) via a configuration file.
2. THE Training Pipeline SHALL support transfer learning by loading ImageNet pretrained weights when configured.
3. THE Training Pipeline SHALL log training metrics (loss, accuracy, MAE) per epoch to a configurable output directory.
4. WHEN training completes, THE Training Pipeline SHALL save the model artifact in TensorFlow SavedModel format to a configurable output path.
5. THE Training Pipeline SHALL treat grade prediction as an ordinal regression problem, not a flat 10-class classification, to preserve grade ordering.

---

### Requirement 9: Model Evaluation

**User Story:** As an ML engineer, I want standardized evaluation metrics, so that I can compare model versions objectively.

#### Acceptance Criteria

1. WHEN evaluation is run against a test set, THE Training Pipeline SHALL report Mean Absolute Error (MAE) between predicted and actual overall grades.
2. WHEN evaluation is run against a test set, THE Training Pipeline SHALL report the percentage of predictions within ±1 grade of the true grade.
3. WHEN evaluation is run against a test set, THE Training Pipeline SHALL generate a confusion matrix for overall grade predictions.
4. THE Training Pipeline SHALL save evaluation results to a JSON report file alongside the model artifact.

---

### Requirement 10: Configuration Management

**User Story:** As a system operator, I want all runtime and training parameters managed via a configuration file, so that deployments are reproducible and environment-specific values are not hardcoded.

#### Acceptance Criteria

1. THE Pregrader SHALL load configuration from environment variables and/or a `.env` file at startup.
2. THE Pregrader SHALL validate all configuration values against a Pydantic settings schema at startup.
3. IF a required configuration value is missing, THEN THE Pregrader SHALL raise a ConfigurationError with the name of the missing field before accepting requests.
4. THE Pregrader SHALL support the following configurable parameters: model artifact path, input image resolution, inference batch size limit, API host/port, log level.
