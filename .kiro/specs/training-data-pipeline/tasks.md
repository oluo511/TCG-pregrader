# Implementation Plan: Training Data Pipeline

## Overview

Incremental build-out from package skeleton → PSA client → scrapers → image processing → manifest/reporting → augmentation extension → orchestrator → CLI. Each phase wires into the previous so there is no orphaned code. All implementation is Python with Pydantic v2, httpx, BeautifulSoup4, OpenCV, and Hypothesis.

## Tasks

- [x] 1. Package scaffolding and configuration
  - Create `data_pipeline/` directory with `__init__.py`
  - Create `data_pipeline/exceptions.py` — full `PipelineError` hierarchy: `QuotaExhaustedError`, `CertLookupError(cert_number, status_code)`, `InvalidImageError`, `DownloadError`, `ConfigurationError`
  - Create `data_pipeline/config.py` — `PipelineSettings(BaseSettings)` with all fields from design: `psa_api_token: SecretStr`, `psa_daily_quota`, `psa_quota_state_path`, `psa_base_url`, `seen_certs_path`, crawl delays, `max_listings_per_grade`, `max_records_per_grade`, `max_concurrent_requests`, quality thresholds, augmentation probabilities, `output_dir`, `manifest_path`, `input_width`, `input_height`, `label_region_fraction`; load from `.env`
  - Add `data-pipeline = "data_pipeline.cli:app"` to `[project.scripts]` in `pyproject.toml`
  - Add pipeline dependencies to `pyproject.toml`: `httpx>=0.27.0`, `beautifulsoup4>=4.12.0`, `numpy>=1.26.0`
  - Create `.env.example` with all required keys including `PSA_API_TOKEN`
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [ ]* 1.1 Write property test for PipelineSettings secret redaction (Property 12)
    - **Property 12: Secret redaction**
    - **Validates: Requirements 9.4**
    - Use `hypothesis` `st.text(min_size=1)` to generate token values; assert `str(settings)`, `repr(settings)`, and `settings.model_dump()` never contain the raw token value

- [ ] 2. PSA API client
  - Create `data_pipeline/psa_client.py` — `PSAClient` class and `QuotaState(BaseModel)` with `calls_today: int`, `reset_at: datetime`
  - Implement `async get_cert(cert_number: str) -> CertRecord`: atomic quota check via `asyncio.Lock` → rate limiter token → httpx GET with retry → parse response → return `CertRecord`
  - Implement `_load_quota_state()` / `_persist_quota_state()`: read/write `.quota_state.json`; reset counter when `now() >= reset_at`
  - Implement `_retry_with_backoff(fn, max_retries=3, base_delay=2.0)`: 5xx/connection → retry with 2s/4s/8s delays; 429 → read `Retry-After` header, sleep, retry; 4xx (not 429) → raise `CertLookupError` immediately
  - Raise `QuotaExhaustedError` without HTTP call when quota exhausted; raise `ConfigurationError` at init if `PSA_API_TOKEN` absent
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 2.1 Write property test for PSA quota enforcement (Property 6)
    - **Property 6: PSA quota enforcement**
    - **Validates: Requirements 1.3, 1.4**
    - Mock httpx to count calls; generate sequences of N `get_cert` calls where N > `psa_daily_quota`; assert total HTTP requests ≤ quota and remaining calls raise `QuotaExhaustedError`

  - [ ]* 2.2 Write property test for retry count bound (Property 7)
    - **Property 7: Retry count bound**
    - **Validates: Requirements 1.5, 1.7**
    - Mock httpx to always return 5xx; assert total HTTP attempts ≤ `max_retries + 1` (default 4) for any error sequence

  - [ ]* 2.3 Write unit tests for PSAClient
    - Test `ConfigurationError` raised at init when `PSA_API_TOKEN` absent
    - Test 429 response reads `Retry-After` header and sleeps that duration
    - Test 4xx (not 429) raises `CertLookupError` with cert number and status code
    - Test quota counter persists across `PSAClient` instances via state file
    - Test quota resets when `reset_at` is in the past
    - _Requirements: 1.1–1.7_

- [ ] 3. Image downloader
  - Create `data_pipeline/downloader.py` — `ImageDownloader` class
  - Implement `async download(url: str, cert_number: str, output_dir: Path) -> Path`: skip if `{cert_number}.jpg` already exists (log INFO); fetch with retry/backoff (max 3 attempts, 1s/2s/4s delays); validate magic bytes (`FF D8 FF` for JPEG, `89 50 4E 47` for PNG) → raise `InvalidImageError` if neither; write to `{output_dir}/{cert_number}.{ext}`; return saved path
  - Raise `DownloadError` after exhausting retries
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 3.1 Write property test for retry count bound on downloads (Property 7 — downloader path)
    - **Property 7: Retry count bound**
    - **Validates: Requirements 4.4**
    - Mock httpx to always fail; assert total HTTP attempts ≤ 4 before `DownloadError` is raised

  - [ ]* 3.2 Write unit tests for ImageDownloader
    - Test skips download and logs INFO when file already exists on disk
    - Test `InvalidImageError` raised for random bytes with no valid magic prefix
    - Test `InvalidImageError` raised for GIF magic bytes (`47 49 46`)
    - Test successful download writes file with cert number as basename
    - _Requirements: 4.1–4.4_

- [ ] 4. Image preprocessor
  - Create `data_pipeline/preprocessor.py` — `ImagePreprocessor` class and `QualityReport` dataclass with `sharpness`, `mean_luminance`, `detected_angle`, `rejected`, `rejection_reason`
  - Implement `filter_quality(image_bytes: bytes, cert_number: str) -> tuple[np.ndarray | None, QualityReport]`: decode → Laplacian variance sharpness → PIL mean luminance → OpenCV contour angle → `PreprocessingService.correct()` if angle > threshold; return `(None, report)` on any rejection; log WARNING with cert number, filter name, and measured value
  - Implement `mask_label_region(image: np.ndarray, cert_number: str, target_size: tuple[int, int]) -> np.ndarray`: raise `InvalidImageError` if height < 100px; crop bottom `label_region_fraction`; letterbox-resize to `target_size`; log INFO with cert number and pixel rows removed
  - Track rejection counts per filter type for `GradeReporter`
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_

  - [ ]* 4.1 Write property test for quality filter monotonicity (Property 4)
    - **Property 4: Quality filter monotonicity**
    - **Validates: Requirements 13.1, 13.2, 13.3, 13.5**
    - Generate random images and threshold pairs; assert relaxing any threshold never causes a previously accepted image to be rejected

  - [ ]* 4.2 Write property test for label masking correctness (Property 10)
    - **Property 10: Label masking correctness**
    - **Validates: Requirements 11.1, 11.2, 11.3**
    - Generate images with height H ≥ 100; assert cropped height = `floor(H * (1 - label_region_fraction))` and final output shape matches `(input_height, input_width)`

  - [ ]* 4.3 Write unit tests for ImagePreprocessor
    - Test `InvalidImageError` raised when image height < 100px in `mask_label_region`
    - Test sharpness rejection logs WARNING with cert number and score
    - Test luminance rejection for underexposed (mean < 30) and overexposed (mean > 230) images
    - Test angle correction is attempted before rejection when angle > threshold
    - _Requirements: 11.1–11.5, 13.1–13.6_

- [ ] 5. Checkpoint — core utilities
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Deduplicator
  - Create `data_pipeline/deduplicator.py` — `Deduplicator` class
  - Implement `load()`: read `.seen_certs.json` if exists, populate `self._seen: set[str]`
  - Implement `is_seen(cert_number: str) -> bool`
  - Implement `mark_seen(cert_number: str, source: str) -> None`: add to in-memory set; log DEBUG if already present
  - Implement `persist()`: atomic write via temp file + `os.replace()` to avoid corrupt state on kill
  - State file format: `{"seen": ["12345678", ...]}`
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 6.1 Write property test for deduplication invariant (Property 1)
    - **Property 1: Deduplication invariant**
    - **Validates: Requirements 5.1, 5.2, 5.3**
    - Use `hypothesis` `st.lists(st.text(min_size=7, max_size=10, alphabet=st.characters(whitelist_categories=("Nd",))))` to generate cert sequences with duplicates; assert `dedup._seen == set(cert_numbers)` after marking all; assert persisted JSON round-trips to same set

  - [ ]* 6.2 Write unit tests for Deduplicator
    - Test `load()` populates set from existing state file
    - Test `persist()` writes atomically (temp file renamed, not written in-place)
    - Test `mark_seen` logs DEBUG on duplicate cert number
    - Test incremental run: load existing state, mark new certs, persist — old certs preserved
    - _Requirements: 5.1–5.4_

- [ ] 7. Manifest builder
  - Create `data_pipeline/manifest.py` — `ManifestBuilder` class
  - Implement `append_row(cert_record: CertRecord, image_path: Path) -> None`: validate via `ManifestRow` (raise `ValidationError` on bad grades, log ERROR with cert number + field, skip row); convert `image_path` to relative from `project_root`; create file with header if not exists; append CSV row
  - Header: `image_path,overall_grade,centering,corners,edges,surface`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 7.1 Write property test for manifest schema validity (Property 2)
    - **Property 2: Manifest schema validity**
    - **Validates: Requirements 6.1, 6.2, 6.3**
    - Use `hypothesis` `st.builds(CertRecord, overall_grade=st.integers(min_value=1, max_value=10), ...)` to generate valid cert records; assert every written row passes `ManifestRow` Pydantic validation on re-read

  - [ ]* 7.2 Write property test for manifest append invariant (Property 3)
    - **Property 3: Manifest append invariant**
    - **Validates: Requirements 6.4**
    - Generate existing manifest with N rows and M new valid records; assert manifest contains exactly N + M rows after appending, original rows unchanged

  - [ ]* 7.3 Write unit tests for ManifestBuilder
    - Test new manifest file is created with correct CSV header
    - Test absolute image path is written as relative path from project root
    - Test `ValidationError` is caught, ERROR is logged, and row is skipped (not raised)
    - _Requirements: 6.1–6.5_

- [ ] 8. Grade reporter
  - Create `data_pipeline/reporter.py` — `GradeReporter` class and `GradeReport(BaseModel)` with `counts_per_grade`, `rejection_counts`, `grades_below_warning`, `grades_at_target`, `total_images`
  - Implement `report(manifest_path: Path, rejection_counts: dict[str, int]) -> GradeReport`: read manifest CSV (not in-memory state); count rows per `overall_grade`; print table to stdout; log WARNING for grades < 100; log INFO for grades ≥ 500
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 8.1 Write property test for grade reporter accuracy (Property 5)
    - **Property 5: Grade reporter accuracy**
    - **Validates: Requirements 7.1, 7.4**
    - Use `hypothesis` `st.lists(st.integers(min_value=1, max_value=10))` to generate grade sequences; write manifest CSV with those grades; assert `report.counts_per_grade[g] == grades.count(g)` for all g in 1–10

  - [ ]* 8.2 Write unit tests for GradeReporter
    - Test WARNING logged for grade with count < 100
    - Test INFO logged for grade with count ≥ 500
    - Test report reads from CSV, not in-memory state (write manifest then instantiate fresh reporter)
    - _Requirements: 7.1–7.4_

- [ ] 9. Base scraper
  - Create `data_pipeline/scrapers/__init__.py` and `data_pipeline/scrapers/base.py` — `BaseScraper` ABC
  - Implement `__init__` wiring: `PipelineSettings`, `PSAClient`, `Deduplicator`, `ImageDownloader`
  - Implement `async scrape(grades: list[int]) -> list[ScrapedRecord]`: fan out across grades with `asyncio.Semaphore(max_concurrent_requests)`; enforce crawl delay via `_acquire_crawl_token()`; check `_check_robots()` before each request
  - Implement `async _check_robots(url: str) -> bool`: fetch and cache `robots.txt` per domain for the run duration; fail-open (return `True`) on fetch failure with WARNING log
  - Implement `async _acquire_crawl_token() -> None`: token-bucket rate limiter; block asynchronously until crawl delay elapsed
  - Define abstract methods `_fetch_listings(grade, page)` and `_extract_cert_number(listing)`
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ]* 9.1 Write unit tests for BaseScraper
    - Test `_check_robots` returns `False` for a disallowed path and `True` for an allowed path
    - Test `_check_robots` fetches robots.txt exactly once per domain per run (cached)
    - Test `_acquire_crawl_token` blocks until crawl delay has elapsed
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 10. eBay scraper
  - Create `data_pipeline/scrapers/ebay.py` — `EbayScraper(BaseScraper)`
  - Implement `async _fetch_listings(grade: int, page: int) -> list[RawListing]`: GET eBay completed listings search URL `https://www.ebay.com/sch/i.html?_nkw=PSA+{grade}+pokemon&LH_Complete=1&LH_Sold=1&_pgn={page}`; parse HTML with BeautifulSoup4; extract listing title + image URL; stop pagination when `max_listings_per_grade` reached
  - Implement `_extract_cert_number(listing: RawListing) -> str | None`: apply `CERT_PATTERN = re.compile(r"(?:PSA|cert)[^\d]*(\d{7,10})", re.IGNORECASE)` to title + description
  - Log WARNING with listing URL and cert number on image download failure; continue processing
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ]* 10.1 Write property test for cert number extraction completeness (Property 8 — eBay)
    - **Property 8: Cert number extraction completeness**
    - **Validates: Requirements 2.2**
    - Use `hypothesis` to generate HTML strings containing PSA cert numbers in valid formats (7–10 digits preceded by "PSA" or "cert"); assert `_extract_cert_number` returns a non-None value matching the embedded cert

  - [ ]* 10.2 Write property test for max-per-grade bound (Property 9 — eBay)
    - **Property 9: Max-per-grade bound**
    - **Validates: Requirements 2.7**
    - Mock `_fetch_listings` to return unlimited listings; assert scraper processes ≤ `max_listings_per_grade` listings per grade regardless of available supply

  - [ ]* 10.3 Write unit tests for EbayScraper
    - Test correct search URL is constructed for each grade 1–10
    - Test WARNING logged and processing continues when image download fails
    - Test cert number not extracted from listing with no PSA pattern → `verified=False` in record
    - _Requirements: 2.1–2.7_

- [ ] 11. Card Ladder scraper
  - Create `data_pipeline/scrapers/cardladder.py` — `CardLadderScraper(BaseScraper)`
  - Implement `async _fetch_listings(grade: int, page: int) -> list[RawListing]`: GET Card Ladder sales history filtered by PSA grade; parse HTML with BeautifulSoup4; extract sale record + image URL; stop pagination when `max_records_per_grade` reached
  - Implement `_extract_cert_number(listing: RawListing) -> str | None`: extract cert from Card Ladder record metadata; return `None` if not present (caller sets `verified=False`)
  - Log WARNING with record URL on image download failure; continue processing
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [ ]* 11.1 Write property test for cert number extraction completeness (Property 8 — CardLadder)
    - **Property 8: Cert number extraction completeness**
    - **Validates: Requirements 3.2**
    - Use `hypothesis` to generate Card Ladder record HTML with embedded cert numbers; assert `_extract_cert_number` returns the correct cert when present, `None` when absent

  - [ ]* 11.2 Write property test for max-per-grade bound (Property 9 — CardLadder)
    - **Property 9: Max-per-grade bound**
    - **Validates: Requirements 3.7**
    - Mock `_fetch_listings` to return unlimited records; assert scraper processes ≤ `max_records_per_grade` records per grade

  - [ ]* 11.3 Write unit tests for CardLadderScraper
    - Test `verified=False` set on record when cert number absent from sale record
    - Test WARNING logged and processing continues when image download fails
    - Test crawl delay ≥ 3 seconds enforced between requests to `cardladder.com`
    - _Requirements: 3.1–3.7_

- [ ] 12. Checkpoint — scrapers
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Augmentation extension — slab-specific transforms
  - Modify `src/pregrader/training/augmentation.py` — extend `AugmentationPipeline`
  - Add `glare_probability: float = 0.3` and `label_occlusion_probability: float = 0.5` parameters to `__init__`
  - Add `training: bool = True` parameter to `apply()` — existing callers unaffected (default `True`)
  - Implement `_apply_glare(image: tf.Tensor) -> tf.Tensor`: with probability `glare_probability`, sample random (x, y) in upper 85% of image, sample random ellipse axes and intensity in [0.3, 0.7], overlay semi-transparent white ellipse via `tf.tensor_scatter_nd_update`; applied after rotation, before normalization
  - Implement `_apply_label_occlusion(image: tf.Tensor) -> tf.Tensor`: with probability `label_occlusion_probability`, replace bottom `label_region_fraction` pixel rows with `tf.ones * mean_color` (solid fill)
  - Guard both new transforms with `if training:` so validation images are never augmented
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 13.1 Write property test for label occlusion determinism (Property 11)
    - **Property 11: Label occlusion determinism**
    - **Validates: Requirements 12.2, 12.5**
    - Generate random images; with `label_occlusion_probability=1.0` assert bottom 15% rows are uniform; with `label_occlusion_probability=0.0` assert bottom region unchanged; with `training=False` assert no augmentation applied

  - [ ]* 13.2 Write unit tests for slab augmentations
    - Test `apply(image, training=False)` returns tensor identical to input (no slab augmentations)
    - Test `apply(image, training=True)` with `label_occlusion_probability=1.0` → bottom 15% is uniform fill
    - Test existing augmentation tests still pass (flip, brightness, rotation unaffected by new params)
    - _Requirements: 12.1–12.5_

- [ ] 14. Orchestrator
  - Create `data_pipeline/orchestrator.py` — `Orchestrator` class
  - Implement `__init__(settings: PipelineSettings)`: wire all components — `PSAClient`, `Deduplicator`, `ImageDownloader`, `ImagePreprocessor`, `ManifestBuilder`, `GradeReporter`; instantiate `EbayScraper` and `CardLadderScraper` with shared components
  - Implement `async run(grades: list[int], max_per_grade: int) -> GradeReport`:
    1. Load deduplicator state
    2. `asyncio.TaskGroup`: run `EbayScraper.scrape()` + `CardLadderScraper.scrape()` concurrently
    3. For each scraped record: `PSAClient.get_cert()` → `ImageDownloader.download()` → `ImagePreprocessor.filter_quality()` → `ManifestBuilder.append_row()` if not rejected
    4. Catch `QuotaExhaustedError` → log ERROR, break PSA calls; catch `CertLookupError` / `InvalidImageError` / `DownloadError` per-record → log WARNING, continue
    5. Persist deduplicator state
    6. Run `GradeReporter.report()` and return `GradeReport`
  - _Requirements: 1.3, 1.4, 2.3, 3.2, 4.1, 5.1, 5.3, 6.1, 7.1, 8.1_

  - [ ]* 14.1 Write unit tests for Orchestrator
    - Test `QuotaExhaustedError` halts PSA calls but pipeline continues downloading remaining queued images
    - Test per-record `CertLookupError` is caught and remaining records are processed
    - Test deduplicator state is persisted even when an exception occurs mid-run
    - _Requirements: 1.3, 1.4, 5.3_

- [ ] 15. CLI entry point
  - Create `data_pipeline/cli.py` — Typer app
  - Implement `run` command with options: `--grades` (default 1–10), `--max-per-grade` (default 500), `--output-dir` (default `data/raw_slabs/`), `--manifest-path` (default `data/manifest.csv`)
  - Load `PipelineSettings`, override `output_dir` and `manifest_path` from CLI args, call `asyncio.run(Orchestrator(settings).run(...))`
  - Catch `ConfigurationError` at startup → print message to stderr, `raise typer.Exit(code=1)`
  - Print grade distribution table from `GradeReport` to stdout on completion
  - Verify `data-pipeline` entry point resolves correctly via `pyproject.toml`
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ]* 15.1 Write unit tests for CLI
    - Test `data-pipeline run` with no `--grades` defaults to grades 1–10
    - Test `data-pipeline run` with no `--max-per-grade` defaults to 500
    - Test exit code 1 and stderr message when `PSA_API_TOKEN` absent (`ConfigurationError`)
    - Test `--output-dir` and `--manifest-path` override `PipelineSettings` values
    - _Requirements: 10.1–10.4_

- [ ] 16. Final checkpoint — full integration
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use `@settings(max_examples=100)` and the `"ci"` Hypothesis profile (already configured in `conftest.py`)
- Each property test file includes a comment: `# Feature: training-data-pipeline, Property N: <title>`
- The 300-line rule applies: `orchestrator.py` and `psa_client.py` are the most likely candidates — flag for modularization if they approach the limit
- `data_pipeline` imports from `pregrader` (one-way dependency); `pregrader` must never import from `data_pipeline`
- Test files live under `TCG-pregrader/tests/unit/data_pipeline/` and `TCG-pregrader/tests/property/data_pipeline/`
