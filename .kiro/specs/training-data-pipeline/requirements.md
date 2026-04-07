# Requirements Document

## Introduction

The Training Data Pipeline is a standalone module (`TCG-pregrader/data_pipeline/`) that
automates collection, verification, and organization of labeled PSA slab photos for
training the TCG Pre-Grader CNN.

**Key architectural facts:**
- Training images are photos of PSA-graded slabs (the card inside the graded plastic
  case), not raw card scans. This is intentional — the model learns to assess card
  surface quality from slab photos, which is the same input it will receive at inference
  time.
- The PSA Public API is the label source only. It provides the authoritative grade and
  subgrades (centering, corners, edges, surface) for a given cert number. It does NOT
  supply images.
- Image sources are eBay sold listings and Card Ladder. The pipeline scrapes slab photos
  from these sources, extracts cert numbers from listing metadata, then calls the PSA API
  once per cert to retrieve verified labels.
- Validation images have the PSA label region masked. The white grade label at the bottom
  of a PSA slab is cropped before images enter the validation split, preventing the model
  from learning to read the label rather than assess card surface quality (data leakage).

The data flow is: scrape slab images (eBay / Card Ladder) → extract cert numbers →
call PSA API per cert → join image + label → write to manifest CSV. The pipeline must
reach a target of 500 images per PSA grade (1–10) while gracefully handling the
structural imbalance in rare grades (1–4).

## Glossary

- **Pipeline**: The `data_pipeline` module as a whole.
- **PSA_Client**: The component that calls `api.psacard.com` to retrieve cert label data
  (grade + subgrades). It is a label source, not an image source.
- **eBay_Scraper**: The component that scrapes eBay completed/sold listings for PSA slab
  photos.
- **CardLadder_Scraper**: The component that scrapes `cardladder.com` sales history for
  PSA slab photos.
- **Cert_Number**: The unique PSA certification number printed on a graded card slab.
- **Cert_Record**: A validated data record containing a cert number, overall grade, and
  four subgrades (centering, corners, edges, surface) sourced from the PSA API.
- **Slab_Photo**: A photograph of a PSA-graded card still inside its plastic slab case.
  All training and validation images are slab photos.
- **Label_Region**: The white PSA label at the bottom of a standard slab photo, occupying
  approximately the bottom 15% of the image, which displays the grade number.
- **Manifest**: The CSV file with columns `image_path, overall_grade, centering, corners,
  edges, surface` consumed by the existing `ManifestLoader`.
- **Deduplicator**: The component that ensures each cert number appears at most once
  across all sources.
- **Orchestrator**: The top-level component that coordinates all scrapers, the
  Deduplicator, and the Manifest_Builder.
- **Manifest_Builder**: The component that joins downloaded images with Cert_Records and
  writes the Manifest CSV.
- **Grade_Reporter**: The component that prints per-grade image counts and warnings.
- **Rate_Limiter**: The token-bucket component that enforces per-source request budgets.
- **Image_Downloader**: The shared utility that fetches and saves slab photos to disk.
- **Image_Preprocessor**: The component responsible for image transformations applied
  before images enter the dataset, including label region masking for validation images.
- **AugmentationPipeline**: The existing augmentation component, extended to support
  slab-specific artifact simulation for training images.
- **Crawl_Delay**: A configurable minimum wait between HTTP requests to a single host.

---

## Requirements

### Requirement 1: PSA API Client

**User Story:** As a pipeline operator, I want to retrieve verified grade labels from the
PSA Public API by cert number, so that every slab photo in the dataset has an
authoritative grade label independent of the image source.

#### Acceptance Criteria

1. WHEN a valid cert number is provided, THE PSA_Client SHALL call
   `GET /cert/GetByCertNumber/{certNumber}` and return a Cert_Record containing
   `overall_grade`, `centering`, `corners`, `edges`, and `surface`.
2. THE PSA_Client SHALL read the PSA API auth token exclusively from the `PSA_API_TOKEN`
   environment variable and SHALL raise a `ConfigurationError` if the variable is absent
   at startup.
3. THE PSA_Client SHALL enforce a daily quota of at most 100 API calls by persisting a
   call counter and reset timestamp to a local state file.
4. WHEN the daily quota is exhausted, THE PSA_Client SHALL raise a `QuotaExhaustedError`
   and SHALL NOT make further API calls until the 24-hour window resets.
5. WHEN an API call fails with a transient HTTP error (5xx or connection timeout), THE
   PSA_Client SHALL retry the request up to 3 times using exponential backoff with a base
   delay of 2 seconds.
6. IF the PSA API returns a non-retryable error (4xx excluding 429), THEN THE PSA_Client
   SHALL raise a `CertLookupError` containing the cert number and HTTP status code.
7. WHEN a rate-limit response (HTTP 429) is received, THE PSA_Client SHALL wait for the
   duration specified in the `Retry-After` header before retrying.

---

### Requirement 2: eBay Sold Listings Scraper

**User Story:** As a pipeline operator, I want to scrape eBay completed listings for PSA-
graded Pokemon cards across all grades, so that I can collect slab photos paired with
cert numbers for label lookup.

#### Acceptance Criteria

1. WHEN a search query of the form `"PSA [grade] pokemon"` is issued for any grade 1–10,
   THE eBay_Scraper SHALL retrieve completed/sold listing URLs from eBay's search results
   pages.
2. THE eBay_Scraper SHALL extract cert numbers from listing titles and descriptions using
   a regex pattern that matches PSA cert number formats (7–10 digit numeric strings
   preceded by "PSA" or "cert").
3. WHEN a cert number is extracted from a listing, THE eBay_Scraper SHALL pass it to the
   PSA_Client for grade label retrieval before adding the slab photo to the dataset.
4. THE eBay_Scraper SHALL respect `robots.txt` for `www.ebay.com` and SHALL NOT request
   any path disallowed by that file.
5. THE eBay_Scraper SHALL enforce a Crawl_Delay of at least 2 seconds between consecutive
   HTTP requests to the same host.
6. WHEN an image download fails, THE eBay_Scraper SHALL log a WARNING with the listing
   URL and cert number and SHALL continue processing remaining listings.
7. THE eBay_Scraper SHALL accept a configurable `max_listings_per_grade` parameter to
   bound the number of listings processed per PSA grade.

---

### Requirement 3: Card Ladder Scraper

**User Story:** As a pipeline operator, I want to scrape Card Ladder's full sales history
for all PSA grades 1–10, so that I can supplement eBay volume across every grade and
take advantage of Card Ladder's aggregated database spanning eBay, Goldin, Heritage, and
other platforms.

#### Acceptance Criteria

1. WHEN a grade filter of 1–10 is applied, THE CardLadder_Scraper SHALL retrieve sales
   records from `cardladder.com` that include slab photos and PSA grade data for the
   specified grade.
2. THE CardLadder_Scraper SHALL extract cert numbers from Card Ladder sale records where
   available and SHALL pass them to the PSA_Client for grade label retrieval.
3. THE CardLadder_Scraper SHALL respect `robots.txt` for `cardladder.com` and SHALL NOT
   request any path disallowed by that file.
4. THE CardLadder_Scraper SHALL enforce a Crawl_Delay of at least 3 seconds between
   consecutive HTTP requests to `cardladder.com`.
5. WHEN a Card Ladder record does not contain a cert number, THE CardLadder_Scraper SHALL
   use the grade value embedded in the sale record as the label and SHALL flag the row in
   the Manifest with `verified=False`.
6. WHEN an image download fails, THE CardLadder_Scraper SHALL log a WARNING with the
   record URL and SHALL continue processing remaining records.
7. THE CardLadder_Scraper SHALL accept a configurable `max_records_per_grade` parameter
   to bound the number of records processed per PSA grade.

---

### Requirement 4: Image Downloader

**User Story:** As a pipeline operator, I want all downloaded slab photos saved with the
cert number as the filename, so that every image is traceable back to its PSA record.

#### Acceptance Criteria

1. WHEN a slab photo is downloaded, THE Image_Downloader SHALL save it to the configured
   output directory using the cert number as the base filename
   (e.g., `{cert_number}.jpg`).
2. IF an image file with the same cert number already exists on disk, THEN THE
   Image_Downloader SHALL skip the download and log an INFO message indicating the file
   was already present.
3. THE Image_Downloader SHALL validate that the downloaded bytes represent a valid image
   (JPEG or PNG) before writing to disk, and SHALL raise an `InvalidImageError` if
   validation fails.
4. THE Image_Downloader SHALL retry a failed download up to 3 times with exponential
   backoff before raising a `DownloadError`.

---

### Requirement 5: Deduplication

**User Story:** As a pipeline operator, I want each cert number to appear at most once in
the final dataset, so that duplicate slab photos do not bias model training.

#### Acceptance Criteria

1. THE Deduplicator SHALL maintain an in-memory set of seen cert numbers across all
   scraper sources during a single pipeline run.
2. WHEN a cert number is encountered that already exists in the seen set, THE Deduplicator
   SHALL discard the duplicate record and log a DEBUG message with the cert number and
   source name.
3. THE Deduplicator SHALL persist the seen-cert-number set to a JSON state file at the
   end of each pipeline run so that subsequent runs do not re-download already-collected
   images.
4. WHEN the pipeline starts, THE Deduplicator SHALL load the persisted state file if it
   exists, so that incremental runs only collect new images.

---

### Requirement 6: Manifest Builder

**User Story:** As a pipeline operator, I want the pipeline to produce a manifest CSV
compatible with the existing `ManifestLoader`, so that collected slab photos can be used
for training without manual reformatting.

#### Acceptance Criteria

1. THE Manifest_Builder SHALL write a CSV file with the header
   `image_path,overall_grade,centering,corners,edges,surface` matching the schema
   expected by `ManifestLoader`.
2. WHEN a Cert_Record is joined with a downloaded slab photo, THE Manifest_Builder SHALL
   validate that `overall_grade` is an integer in [1, 10] and all subgrade values are
   floats in [1.0, 10.0] before writing the row.
3. IF a Cert_Record fails validation, THEN THE Manifest_Builder SHALL log an ERROR with
   the cert number and the failing field, and SHALL skip that row.
4. THE Manifest_Builder SHALL append to an existing manifest file rather than overwriting
   it, so that incremental pipeline runs accumulate rows.
5. THE Manifest_Builder SHALL write image paths as relative paths from the project root
   so that the manifest remains portable across machines.

---

### Requirement 7: Grade Distribution Reporter

**User Story:** As a pipeline operator, I want a grade distribution report after each
run, so that I can monitor progress toward the 500-images-per-grade target.

#### Acceptance Criteria

1. WHEN a pipeline run completes, THE Grade_Reporter SHALL print a table showing the
   count of images collected per PSA grade (1–10) across the full manifest.
2. WHEN a grade has fewer than 100 images in the manifest, THE Grade_Reporter SHALL log a
   WARNING with the grade value and current count.
3. WHEN a grade reaches or exceeds 500 images, THE Grade_Reporter SHALL log an INFO
   message indicating the target has been met for that grade.
4. THE Grade_Reporter SHALL compute counts by reading the manifest CSV rather than
   relying on in-memory state, so that the report reflects the true persisted dataset.

---

### Requirement 8: Rate Limiting and Polite Crawling

**User Story:** As a pipeline operator, I want all external requests to be rate-limited
and polite, so that the pipeline does not violate terms of service or get blocked.

#### Acceptance Criteria

1. THE Rate_Limiter SHALL implement a token-bucket algorithm with a configurable refill
   rate and bucket capacity per source.
2. WHEN a scraper requests a token and the bucket is empty, THE Rate_Limiter SHALL block
   the caller asynchronously until a token is available rather than raising an error.
3. THE Pipeline SHALL load per-source rate-limit configuration (requests per minute,
   crawl delay) from environment variables with documented defaults.
4. THE Pipeline SHALL check `robots.txt` for each scraped domain at startup and cache the
   result for the duration of the run.

---

### Requirement 9: Configuration and Secrets Management

**User Story:** As a pipeline operator, I want all secrets and tunable parameters
managed through environment variables, so that the pipeline is safe to commit and easy
to configure across environments.

#### Acceptance Criteria

1. THE Pipeline SHALL load all configuration from environment variables using a Pydantic
   `BaseSettings` model (`PipelineSettings`).
2. THE Pipeline SHALL require `PSA_API_TOKEN` to be set and SHALL raise a
   `ConfigurationError` at startup if it is absent.
3. THE Pipeline SHALL document all supported environment variables in an updated
   `.env.example` file.
4. THE Pipeline SHALL never log or print the value of `PSA_API_TOKEN` or any other
   secret field.

---

### Requirement 10: CLI Entry Point

**User Story:** As a pipeline operator, I want a CLI command to run the pipeline, so
that I can trigger data collection from the terminal or a scheduled job.

#### Acceptance Criteria

1. THE Pipeline SHALL expose a `data-pipeline run` CLI command that accepts
   `--grades`, `--max-per-grade`, `--output-dir`, and `--manifest-path` arguments.
2. WHEN `--grades` is omitted, THE Pipeline SHALL default to collecting all grades 1–10.
3. WHEN `--max-per-grade` is omitted, THE Pipeline SHALL default to 500.
4. THE Pipeline CLI SHALL print a progress summary to stdout on completion, including
   total images downloaded and the grade distribution table from the Grade_Reporter.

---

### Requirement 11: Validation Split Label Masking

**User Story:** As a pipeline operator, I want the PSA label region cropped from
validation images before they enter the dataset, so that the model cannot learn to read
the grade number from the slab label instead of assessing card surface quality.

#### Acceptance Criteria

1. WHEN an image is designated for the validation split, THE Image_Preprocessor SHALL
   crop the bottom 15% of the slab photo to remove the Label_Region before the image is
   written to the validation dataset directory.
2. THE Image_Preprocessor SHALL apply label masking exclusively to validation images and
   SHALL NOT modify training images, so that training images retain the full slab context.
3. WHEN label masking is applied, THE Image_Preprocessor SHALL preserve the original
   aspect ratio of the cropped image by resizing to the configured input dimensions after
   the crop.
4. THE Image_Preprocessor SHALL log an INFO message for each image masked, including the
   cert number and the pixel rows removed, to support auditability.
5. IF the source image height is less than 100 pixels, THEN THE Image_Preprocessor SHALL
   raise an `InvalidImageError` rather than applying the crop, as the label region
   percentage would be unreliable at that resolution.

---

### Requirement 12: Slab-Specific Augmentation

**User Story:** As a pipeline operator, I want the augmentation pipeline extended with
slab-specific artifact simulation, so that the model is robust to the glare, reflection,
and label occlusion commonly present in real-world slab photos.

#### Acceptance Criteria

1. THE AugmentationPipeline SHALL apply random glare simulation to training images by
   overlaying a semi-transparent elliptical highlight at a random position and intensity
   within the upper 85% of the image (above the Label_Region).
2. THE AugmentationPipeline SHALL apply random label region occlusion to training images
   by replacing the bottom 15% of the image with a solid fill at a configurable
   probability, simulating partial label obstruction.
3. WHEN slab augmentations are applied, THE AugmentationPipeline SHALL apply them after
   standard geometric augmentations (flip, rotate, crop) and before normalization, so
   that artifact simulation occurs in pixel space.
4. THE AugmentationPipeline SHALL expose `glare_probability` and
   `label_occlusion_probability` as configurable parameters in `PipelineSettings`,
   defaulting to 0.3 and 0.5 respectively.
5. THE AugmentationPipeline SHALL NOT apply slab-specific augmentations to validation
   images, so that validation metrics reflect real-world slab photo conditions without
   synthetic artifacts.

---

### Requirement 13: Image Quality Filtering

**User Story:** As a pipeline operator, I want low-quality slab photos automatically
rejected during collection, so that blurry, dark, or severely angled images do not
degrade model training.

#### Acceptance Criteria

1. WHEN a slab photo is downloaded, THE Image_Preprocessor SHALL compute a sharpness
   score using the Laplacian variance method and SHALL reject images with a score below a
   configurable `min_sharpness` threshold (default: 50.0), logging a WARNING with the
   cert number and computed score.
2. WHEN a slab photo is downloaded, THE Image_Preprocessor SHALL compute the mean
   luminance of the image and SHALL reject images with mean luminance below
   `min_luminance` (default: 30.0 on a 0–255 scale, indicating severe underexposure) or
   above `max_luminance` (default: 230.0, indicating severe overexposure/glare washout).
3. WHEN a slab photo is downloaded, THE Image_Preprocessor SHALL detect the slab
   orientation by checking the aspect ratio of the largest detected rectangle contour. IF
   the detected angle deviates more than `max_angle_degrees` (default: 15°) from
   vertical, THE Image_Preprocessor SHALL attempt perspective correction via the existing
   `PreprocessingService`. IF correction fails, the image SHALL be rejected with a
   WARNING.
4. WHEN an image is rejected by any quality filter, THE Image_Preprocessor SHALL log the
   rejection reason (sharpness, luminance, or angle), the cert number, and the measured
   value so the operator can tune thresholds if needed.
5. THE `min_sharpness`, `min_luminance`, `max_luminance`, and `max_angle_degrees`
   thresholds SHALL be configurable via `PipelineSettings` so the operator can relax
   filters for rare grades where volume is more important than image quality.
6. THE Image_Preprocessor SHALL track and report the total number of images rejected per
   quality filter type in the Grade_Reporter output at the end of each pipeline run.
