"""
PreprocessingService — normalizes raw card images into model-ready tensors.

Design pattern: Sequential transformation pipeline.
Each step is a discrete, testable transformation: decode → align →
resize → normalize → crop. Perspective correction is the only step that
can fail gracefully — all other failures are unrecoverable and raise
PreprocessingError.

Technical Debt: Perspective correction via contour detection is brittle
on low-contrast backgrounds or cards in sleeves. A production-grade
approach would use a dedicated card-detection model (e.g., fine-tuned
YOLO) as a preprocessing stage. For MVP, we log a WARNING and pass
through the uncorrected image if detection fails.

Technical Debt: full_tensor and region tensors are converted to nested
Python lists for Pydantic schema compliance. In the hot inference path,
pass np.ndarray directly between services and only coerce at the API
boundary to avoid the serialization overhead.
"""

import io
from typing import Final

import cv2
import numpy as np
from PIL import Image

from pregrader.exceptions import PreprocessingError
from pregrader.logging_config import get_logger
from pregrader.schemas import CardRegion, PreprocessedCard

# Target dimensions per Requirement 2.1 — must match the model's input shape.
# Height-first ordering follows numpy convention (H, W, C).
_TARGET_WIDTH: Final[int] = 224
_TARGET_HEIGHT: Final[int] = 312

# Region crop ratios — defined as fractions of the full image dimensions.
# These are the canonical subgrade evaluation windows used by human graders:
#   centering: the inner 80% captures the card face without border noise
#   surface:   the inner 60% isolates the print surface from edge artifacts
#   edges:     the outer 10% strips capture border wear and whitening
#   corners:   40×40px patches at each corner capture the most wear-prone areas
_CENTERING_RATIO: Final[float] = 0.80
_SURFACE_RATIO: Final[float] = 0.60
_EDGE_RATIO: Final[float] = 0.10
_CORNER_PATCH_SIZE: Final[int] = 40


class PreprocessingService:
    """Stateless image transformation pipeline.

    Converts raw image bytes into a PreprocessedCard containing:
    - full_tensor: the full resized, normalized card (H×W×C)
    - regions: four CardRegion crops for subgrade inference heads

    All methods are pure functions of their inputs — no mutable state is
    held between calls, making this safe for concurrent use.
    """

    def __init__(self) -> None:
        self._logger = get_logger(service="preprocessing")

    def preprocess(self, raw_bytes: bytes, image_id: str) -> PreprocessedCard:
        """Run the full preprocessing pipeline on a single card image.

        Steps (in order):
          1. Decode bytes → PIL Image
          2. Perspective correction via OpenCV contour/homography (non-fatal)
          3. Resize to 224×312
          4. Normalize pixel values to [0.0, 1.0]
          5. Extract four CardRegion crops

        Args:
            raw_bytes: Raw JPEG or PNG bytes from ingestion.
            image_id: Identifier used in log messages and the output schema.

        Returns:
            PreprocessedCard with image_id, full_tensor, and four regions.

        Raises:
            PreprocessingError: If decoding or normalization fails
                unrecoverably. Perspective correction failure is NOT raised —
                it logs a WARNING and continues with the uncorrected image.
        """
        # Step 1: Decode bytes to a PIL Image.
        # PIL is used here (not cv2.imdecode) because it handles a wider
        # range of JPEG sub-formats and gives cleaner error messages.
        pil_image = self._decode_bytes(raw_bytes, image_id)

        # Step 2: Attempt perspective correction.
        # Failure is non-fatal per Requirement 2.4 — log and continue.
        pil_image = self._apply_perspective_correction(pil_image, image_id)

        # Step 3: Resize to the fixed model input resolution.
        pil_image = pil_image.resize(
            (_TARGET_WIDTH, _TARGET_HEIGHT), resample=Image.LANCZOS
        )

        # Step 4: Convert to float32 numpy array and normalize to [0.0, 1.0].
        # Dividing by 255.0 maps uint8 [0, 255] → float32 [0.0, 1.0].
        arr = np.array(pil_image, dtype=np.float32) / 255.0

        # Ensure the array is always H×W×3 — convert grayscale or RGBA to RGB.
        arr = self._ensure_rgb(arr, image_id)

        # Step 5: Extract the four canonical card region crops.
        regions = self._extract_regions(arr)

        return PreprocessedCard(
            image_id=image_id,
            full_tensor=arr.tolist(),
            regions=regions,
        )

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    def _decode_bytes(self, raw_bytes: bytes, image_id: str) -> Image.Image:
        """Decode raw bytes to a PIL Image in RGB mode.

        Raises:
            PreprocessingError: If PIL cannot decode the bytes.
        """
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            # Force full decode now so corrupt data raises here, not later.
            img.load()
            return img.convert("RGB")
        except Exception as exc:
            self._logger.error(
                "image_decode_failed", image_id=image_id, error=str(exc)
            )
            raise PreprocessingError(
                f"Failed to decode image '{image_id}': {exc}"
            ) from exc

    def _apply_perspective_correction(
        self, pil_image: Image.Image, image_id: str
    ) -> Image.Image:
        """Attempt to detect card borders and apply a homography warp.

        Uses OpenCV contour detection on a Canny edge map to find the
        largest quadrilateral contour (the card boundary). If a valid
        four-corner contour is found, a perspective transform is applied
        to produce a front-facing, axis-aligned card image.

        On any failure (no contour found, degenerate geometry, OpenCV
        error), logs a WARNING with image_id and returns the original
        image unchanged — per Requirement 2.4.

        Args:
            pil_image: RGB PIL Image (pre-resize).
            image_id: Used in the WARNING log entry.

        Returns:
            Corrected PIL Image, or the original if correction failed.
        """
        try:
            # Convert to BGR numpy array for OpenCV processing.
            bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            # Blur before edge detection to reduce noise from card texture.
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

            # Dilate edges to close small gaps in the card border contour.
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                self._logger.warning(
                    "perspective_correction_no_contours",
                    image_id=image_id,
                )
                return pil_image

            # Take the largest contour by area — most likely the card boundary.
            largest = max(contours, key=cv2.contourArea)

            # Approximate the contour to a polygon. epsilon controls how
            # aggressively we simplify — 2% of perimeter is a common heuristic.
            epsilon = 0.02 * cv2.arcLength(largest, closed=True)
            approx = cv2.approxPolyDP(largest, epsilon, closed=True)

            # We need exactly 4 corners for a valid homography.
            if len(approx) != 4:
                self._logger.warning(
                    "perspective_correction_not_quadrilateral",
                    image_id=image_id,
                    corner_count=len(approx),
                )
                return pil_image

            # Order corners: top-left, top-right, bottom-right, bottom-left.
            src_pts = self._order_corners(approx.reshape(4, 2).astype(np.float32))

            h, w = bgr.shape[:2]
            dst_pts = np.array(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                dtype=np.float32,
            )

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_bgr = cv2.warpPerspective(bgr, M, (w, h))

            # Convert back to RGB PIL Image for the rest of the pipeline.
            warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(warped_rgb)

        except Exception as exc:
            # Any OpenCV or geometry error is non-fatal — log and continue.
            self._logger.warning(
                "perspective_correction_failed",
                image_id=image_id,
                error=str(exc),
            )
            return pil_image

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Sort four (x, y) points into [top-left, top-right, bottom-right, bottom-left].

        Uses the sum (x+y) and difference (x-y) trick:
        - top-left has the smallest sum
        - bottom-right has the largest sum
        - top-right has the smallest difference
        - bottom-left has the largest difference
        """
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]   # top-left
        ordered[2] = pts[np.argmax(s)]   # bottom-right
        diff = np.diff(pts, axis=1)
        ordered[1] = pts[np.argmin(diff)]  # top-right
        ordered[3] = pts[np.argmax(diff)]  # bottom-left
        return ordered

    def _ensure_rgb(self, arr: np.ndarray, image_id: str) -> np.ndarray:
        """Guarantee the array has shape (H, W, 3).

        PIL.convert("RGB") should always produce 3 channels, but this guard
        catches any edge case (e.g., a grayscale image that slipped through).
        """
        if arr.ndim == 2:
            # Grayscale — stack to 3 channels
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 4:
            # RGBA — drop alpha channel
            arr = arr[:, :, :3]
        elif arr.shape[2] != 3:
            raise PreprocessingError(
                f"Image '{image_id}' has unexpected channel count: {arr.shape[2]}. "
                f"Expected 3 (RGB)."
            )
        return arr

    def _extract_regions(self, arr: np.ndarray) -> list[CardRegion]:
        """Extract the four canonical card region crops from a normalized array.

        All crops are taken from the already-resized 312×224 array, so pixel
        coordinates are deterministic regardless of the original image size.

        Region definitions:
          centering — center 80% of width and height (inner crop)
          corners   — four 40×40px patches at each corner, stacked along axis 0
          edges     — top/bottom/left/right 10% border strips, stacked along axis 0
          surface   — center 60% of width and height (inner crop)

        Args:
            arr: Float32 numpy array of shape (312, 224, 3), values in [0.0, 1.0].

        Returns:
            List of four CardRegion objects in the order:
            [centering, corners, edges, surface].
        """
        h, w = arr.shape[:2]  # 312, 224

        regions: list[CardRegion] = []

        # --- centering: inner 80% crop ---
        # Captures the card face without the outermost border noise.
        regions.append(CardRegion(
            name="centering",
            tensor=self._center_crop(arr, _CENTERING_RATIO).tolist(),
        ))

        # --- corners: four 40×40px patches concatenated into one tensor ---
        # Each corner is the most wear-prone area; concatenating horizontally
        # gives the model a single 3D input (40, 160, 3) that sees all four
        # simultaneously while remaining compatible with the 3D tensor schema.
        corner_patches = self._extract_corner_patches(arr)
        # Concatenate along axis 1 (width) → shape (40, 160, 3)
        corners_tensor = np.concatenate(corner_patches, axis=1)
        regions.append(CardRegion(
            name="corners",
            tensor=corners_tensor.tolist(),
        ))

        # --- edges: four border strips concatenated into one tensor ---
        # Top/bottom strips are 10% of height; left/right strips are 10% of width.
        # Concatenating gives the model a unified 3D view of all four edges
        # while remaining compatible with the 3D tensor schema.
        edge_strips = self._extract_edge_strips(arr)
        # Pad strips to the same spatial size then concatenate along axis 1
        edges_tensor = self._concat_edge_strips(edge_strips, h, w)
        regions.append(CardRegion(
            name="edges",
            tensor=edges_tensor.tolist(),
        ))

        # --- surface: inner 60% crop ---
        # Isolates the print surface from edge artifacts and border wear.
        regions.append(CardRegion(
            name="surface",
            tensor=self._center_crop(arr, _SURFACE_RATIO).tolist(),
        ))

        return regions

    def _center_crop(self, arr: np.ndarray, ratio: float) -> np.ndarray:
        """Crop the center `ratio` fraction of both width and height.

        For a 312×224 image with ratio=0.80:
          crop_h = int(312 * 0.80) = 249
          crop_w = int(224 * 0.80) = 179
          top = (312 - 249) // 2 = 31
          left = (224 - 179) // 2 = 22
        """
        h, w = arr.shape[:2]
        crop_h = int(h * ratio)
        crop_w = int(w * ratio)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return arr[top : top + crop_h, left : left + crop_w, :]

    def _extract_corner_patches(self, arr: np.ndarray) -> list[np.ndarray]:
        """Extract four 40×40px patches from each corner of the image.

        Order: top-left, top-right, bottom-left, bottom-right.
        This order is consistent with how human graders inspect corners.
        """
        h, w = arr.shape[:2]
        p = _CORNER_PATCH_SIZE
        return [
            arr[0:p, 0:p, :],           # top-left
            arr[0:p, w - p : w, :],     # top-right
            arr[h - p : h, 0:p, :],     # bottom-left
            arr[h - p : h, w - p : w, :],  # bottom-right
        ]

    def _extract_edge_strips(self, arr: np.ndarray) -> list[np.ndarray]:
        """Extract four border strips (top, bottom, left, right) at 10% thickness.

        Strip thicknesses:
          top/bottom: 10% of height = int(312 * 0.10) = 31px
          left/right: 10% of width  = int(224 * 0.10) = 22px
        """
        h, w = arr.shape[:2]
        strip_h = int(h * _EDGE_RATIO)
        strip_w = int(w * _EDGE_RATIO)
        return [
            arr[0:strip_h, :, :],           # top strip  (strip_h × w × 3)
            arr[h - strip_h : h, :, :],     # bottom strip
            arr[:, 0:strip_w, :],           # left strip  (h × strip_w × 3)
            arr[:, w - strip_w : w, :],     # right strip
        ]

    def _concat_edge_strips(
        self,
        strips: list[np.ndarray],
        full_h: int,
        full_w: int,
    ) -> np.ndarray:
        """Pad and concatenate the four edge strips into a single (max_h, 4*max_w, 3) tensor.

        Top/bottom strips have shape (strip_h, full_w, 3).
        Left/right strips have shape (full_h, strip_w, 3).

        To concatenate them into a uniform 3D tensor we pad each strip to the
        same height (max_h), then concatenate along axis 1 (width).

        Why concatenate instead of stack? CardRegion.tensor is typed as
        list[list[list[float]]] (3D), so we must produce a 3D array.
        Concatenation along the width axis preserves 3D shape while keeping
        all four strips accessible in a single tensor.

        Why pad instead of resize? Padding preserves pixel values exactly —
        no interpolation artifacts that could confuse the edge-quality head.
        """
        strip_h = int(full_h * _EDGE_RATIO)
        strip_w = int(full_w * _EDGE_RATIO)

        # Target height: the tallest strip (left/right strips are full_h tall)
        target_h = max(strip_h, full_h)

        padded: list[np.ndarray] = []
        for strip in strips:
            sh = strip.shape[0]
            pad_h = target_h - sh
            # Pad with zeros (black) on the bottom — neutral padding
            # that won't introduce false edge signals.
            padded_strip = np.pad(
                strip,
                ((0, pad_h), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
            padded.append(padded_strip)

        # Concatenate along axis 1 (width) → (target_h, sum_of_widths, 3)
        return np.concatenate(padded, axis=1)
