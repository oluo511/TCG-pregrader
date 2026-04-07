"""
ImagePreprocessor — quality filtering and label-region masking for slab photos.

Design pattern: Strategy chain — quality filters are applied sequentially;
any rejection short-circuits the chain and returns (None, QualityReport)
immediately. This avoids wasted computation on images that will be discarded.

Why PreprocessingService for angle correction?
The existing PreprocessingService._apply_perspective_correction() already
implements a robust homography-based warp. Reusing it here avoids duplicating
that logic and ensures the pipeline and the inference server apply identical
corrections — a critical consistency requirement for training data quality.

Shape convention throughout: (H, W, C) — Height × Width × Channels.
All array dimension comments use this ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import structlog
from PIL import Image

from data_pipeline._image_utils import detect_angle, letterbox_resize
from data_pipeline.config import PipelineSettings
from data_pipeline.exceptions import InvalidImageError
from pregrader.services.preprocessing import PreprocessingService

logger = structlog.get_logger(__name__)


@dataclass
class QualityReport:
    """Outcome of a single image's quality filter chain.

    All measured values are populated regardless of rejection so the operator
    can inspect borderline cases and tune thresholds without re-running.
    """

    sharpness: float
    mean_luminance: float
    detected_angle: float
    rejected: bool
    rejection_reason: str | None = None


class ImagePreprocessor:
    """Quality filtering and label-region masking for PSA slab photos.

    Responsibilities:
    1. filter_quality  — decode + sharpness + luminance + angle checks
    2. mask_label_region — crop label + letterbox-resize for validation split
    3. rejection_counts — per-filter tallies consumed by GradeReporter

    Thread safety: _rejection_counts is mutated in filter_quality. If the
    orchestrator ever calls this from concurrent tasks, wrap mutations in a
    threading.Lock. For the current async-but-single-threaded design this is
    not needed.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings
        # Reuse the existing perspective-correction logic from the inference
        # pipeline — ensures training and serving apply identical corrections.
        self._preprocessing_service = PreprocessingService()
        # Tracks per-filter rejection counts for GradeReporter (Req 13.6).
        self._rejection_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_quality(
        self,
        image_bytes: bytes,
        cert_number: str,
    ) -> tuple[np.ndarray | None, QualityReport]:
        """Run the quality filter chain on raw image bytes.

        Filters applied in order (any rejection returns immediately):
          1. Decode bytes → BGR numpy array
          2. Sharpness (Laplacian variance)
          3. Luminance (mean pixel intensity via PIL)
          4. Skew angle (largest contour minAreaRect; correction attempted)

        Args:
            image_bytes: Raw JPEG or PNG bytes.
            cert_number: Used in log messages for traceability.

        Returns:
            (array, report) where array is None if the image was rejected.
            array shape: (H, W, 3) — BGR uint8.

        Raises:
            InvalidImageError: If cv2.imdecode cannot decode the bytes.
        """
        s = self._settings

        # ----------------------------------------------------------------
        # Step 1 — Decode bytes to BGR numpy array (H, W, 3)
        # ----------------------------------------------------------------
        buf = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # shape: (H, W, 3) BGR uint8
        if bgr is None:
            raise InvalidImageError(
                f"cv2.imdecode returned None for cert {cert_number!r} — "
                "bytes are not a valid JPEG/PNG."
            )

        # ----------------------------------------------------------------
        # Step 2 — Sharpness: Laplacian variance (Req 13.1)
        # Higher variance = sharper image. Blurry images have low variance.
        # ----------------------------------------------------------------
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # shape: (H, W)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if sharpness < s.min_sharpness:
            logger.warning(
                "quality_filter_rejected",
                cert_number=cert_number,
                filter="sharpness",
                value=sharpness,
            )
            self._rejection_counts["sharpness"] = (
                self._rejection_counts.get("sharpness", 0) + 1
            )
            return None, QualityReport(
                sharpness=sharpness,
                mean_luminance=0.0,
                detected_angle=0.0,
                rejected=True,
                rejection_reason="sharpness",
            )

        # ----------------------------------------------------------------
        # Step 3 — Luminance: mean pixel intensity via PIL (Req 13.2, 13.3)
        # Convert BGR → RGB PIL Image → grayscale ('L') → mean.
        # PIL is used here (not cv2) to stay consistent with the inference
        # pipeline's luminance calculation.
        # ----------------------------------------------------------------
        # bgr shape: (H, W, 3) — convert to RGB for PIL
        rgb_arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # shape: (H, W, 3)
        pil_img = Image.fromarray(rgb_arr)
        mean_lum = float(np.array(pil_img.convert("L")).mean())

        if mean_lum < s.min_luminance:
            logger.warning(
                "quality_filter_rejected",
                cert_number=cert_number,
                filter="luminance_low",
                value=mean_lum,
            )
            self._rejection_counts["luminance_low"] = (
                self._rejection_counts.get("luminance_low", 0) + 1
            )
            return None, QualityReport(
                sharpness=sharpness,
                mean_luminance=mean_lum,
                detected_angle=0.0,
                rejected=True,
                rejection_reason="luminance_low",
            )

        if mean_lum > s.max_luminance:
            logger.warning(
                "quality_filter_rejected",
                cert_number=cert_number,
                filter="luminance_high",
                value=mean_lum,
            )
            self._rejection_counts["luminance_high"] = (
                self._rejection_counts.get("luminance_high", 0) + 1
            )
            return None, QualityReport(
                sharpness=sharpness,
                mean_luminance=mean_lum,
                detected_angle=0.0,
                rejected=True,
                rejection_reason="luminance_high",
            )

        # ----------------------------------------------------------------
        # Step 4 — Angle detection and correction (Req 13.4, 13.5)
        # Use largest contour minAreaRect to detect slab skew.
        # If angle > threshold, attempt perspective correction via the
        # existing PreprocessingService before re-checking.
        # ----------------------------------------------------------------
        angle = detect_angle(bgr)  # degrees in [0, 90]

        if angle > s.max_skew_angle:
            # Attempt correction: BGR → RGB PIL → _apply_perspective_correction
            # → back to BGR numpy array (H, W, 3).
            corrected_pil = self._preprocessing_service._apply_perspective_correction(
                pil_img, cert_number
            )
            # Convert corrected RGB PIL back to BGR numpy array (H, W, 3)
            corrected_bgr = cv2.cvtColor(
                np.array(corrected_pil), cv2.COLOR_RGB2BGR
            )  # shape: (H, W, 3)

            # Re-detect angle on the corrected image.
            angle_after = detect_angle(corrected_bgr)

            if angle_after > s.max_skew_angle:
                # Correction did not bring angle within threshold — reject.
                logger.warning(
                    "quality_filter_rejected",
                    cert_number=cert_number,
                    filter="angle",
                    value=angle_after,
                )
                self._rejection_counts["angle"] = (
                    self._rejection_counts.get("angle", 0) + 1
                )
                return None, QualityReport(
                    sharpness=sharpness,
                    mean_luminance=mean_lum,
                    detected_angle=angle_after,
                    rejected=True,
                    rejection_reason="angle",
                )

            # Correction succeeded — use the corrected array downstream.
            bgr = corrected_bgr  # shape: (H, W, 3)
            angle = angle_after

        # ----------------------------------------------------------------
        # All filters passed — return the (possibly corrected) BGR array.
        # ----------------------------------------------------------------
        return bgr, QualityReport(
            sharpness=sharpness,
            mean_luminance=mean_lum,
            detected_angle=angle,
            rejected=False,
        )

    def mask_label_region(
        self,
        image: np.ndarray,
        cert_number: str,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Crop the PSA label region and letterbox-resize for validation split.

        The PSA grade label occupies the bottom `label_region_fraction` of a
        standard slab photo. Removing it prevents the model from learning to
        read the label instead of assessing card surface quality (data leakage).

        Args:
            image: BGR numpy array of shape (H, W, C).
            cert_number: Used in log messages for auditability.
            target_size: (target_height, target_width) — output dimensions.

        Returns:
            Letterboxed BGR numpy array of shape (target_height, target_width, C).

        Raises:
            InvalidImageError: If image height < 100px (Req 11.5).
        """
        # image shape: (H, W, C)
        H, W, C = image.shape

        if H < 100:
            raise InvalidImageError(
                f"Image for cert {cert_number!r} has height {H}px — "
                "minimum 100px required for reliable label-region masking."
            )

        # ----------------------------------------------------------------
        # Crop: remove bottom label_region_fraction rows (Req 11.1)
        # ----------------------------------------------------------------
        rows_to_remove = int(H * self._settings.label_region_fraction)
        cropped = image[: H - rows_to_remove, :, :]  # shape: (H - rows_to_remove, W, C)
        cropped_H = cropped.shape[0]

        logger.info(
            "label_region_masked",
            cert_number=cert_number,
            rows_removed=rows_to_remove,
        )

        # Letterbox-resize to target_size — delegates to _image_utils (Req 11.3).
        target_height, target_width = target_size
        return letterbox_resize(cropped, target_height, target_width)

    @property
    def rejection_counts(self) -> dict[str, int]:
        """Return a copy of per-filter rejection tallies for GradeReporter."""
        return dict(self._rejection_counts)
