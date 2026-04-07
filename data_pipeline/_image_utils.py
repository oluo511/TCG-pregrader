"""
_image_utils.py — stateless OpenCV helpers for the data pipeline.

Extracted from preprocessor.py to keep that module under 300 lines.
These are pure functions with no side effects — safe to unit test in isolation
and reuse across the pipeline without instantiating ImagePreprocessor.

Shape convention: (H, W, C) — Height × Width × Channels throughout.
"""

import cv2
import numpy as np


def detect_angle(bgr: np.ndarray) -> float:
    """Detect slab skew angle using the largest contour's minAreaRect.

    Why minAreaRect?
    It returns the minimum-area bounding rectangle of a contour, giving us
    the rotation angle of the dominant shape — the card slab boundary.
    cv2.minAreaRect returns angle in [-90, 0); we normalize to [0, 90].

    Args:
        bgr: BGR numpy array of shape (H, W, 3).

    Returns:
        Skew angle in degrees, in the range [0, 90].
        Returns 0.0 if no contours are detected.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)       # shape: (H, W)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)        # shape: (H, W)
    edges = cv2.Canny(blurred, 50, 150)                # shape: (H, W) binary edge map
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    _, _, angle = cv2.minAreaRect(largest)
    return abs(angle) if angle != 0 else 0.0


def letterbox_resize(
    image: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Uniformly scale image to fit target dimensions, padding remainder with black.

    Why letterbox over plain resize?
    Plain resize distorts aspect ratio. Letterboxing preserves it by scaling
    to the smaller axis and padding — the model sees undistorted card geometry.

    Args:
        image: BGR numpy array of shape (H, W, C).
        target_height: Output height in pixels.
        target_width: Output width in pixels.

    Returns:
        BGR numpy array of shape (target_height, target_width, C), dtype
        preserved from input.
    """
    H, W, C = image.shape  # (H, W, C)

    # Uniform scale — use the smaller axis to avoid overflow on either dimension.
    scale = min(target_width / W, target_height / H)
    new_w = int(W * scale)
    new_h = int(H * scale)

    # INTER_LINEAR: good quality/speed trade-off for downscaling.
    resized = cv2.resize(image, (new_w, new_h))  # shape: (new_h, new_w, C)

    # Black canvas at exact target dimensions.
    canvas = np.zeros((target_height, target_width, C), dtype=image.dtype)

    # Center the resized image on the canvas.
    pad_top = (target_height - new_h) // 2
    pad_left = (target_width - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w, :] = resized

    return canvas  # shape: (target_height, target_width, C)
