"""
Synthetic slab image generator for smoke-testing the training loop.

Usage:
    python -m data_pipeline.generate_synthetic_data \\
        --output-dir data/raw_slabs/ \\
        --manifest-path data/manifest.csv \\
        --images-per-grade 50

Why synthetic data?
  Real data collection takes days (PSA quota limits, scraping time). Synthetic
  data lets you verify the full training loop — ManifestLoader → DatasetBuilder
  → AugmentationPipeline → Trainer — in minutes.

Image shape: (Height=312, Width=224, Channels=3) — HWC convention.
Matches DatasetBuilder._IMG_HEIGHT / _IMG_WIDTH constants.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Image dimensions must match DatasetBuilder constants
_HEIGHT = 312
_WIDTH = 224
_LABEL_FRACTION = 0.15
_LABEL_HEIGHT = int(_HEIGHT * _LABEL_FRACTION)  # 46px
_CARD_HEIGHT = _HEIGHT - _LABEL_HEIGHT           # 266px

# Distinct card body colors per grade for visual debugging
_GRADE_COLORS: dict[int, tuple[int, int, int]] = {
    1:  (180, 60,  60),
    2:  (200, 100, 60),
    3:  (210, 140, 60),
    4:  (200, 180, 60),
    5:  (160, 190, 60),
    6:  (80,  180, 80),
    7:  (60,  180, 160),
    8:  (60,  140, 200),
    9:  (80,  80,  200),
    10: (140, 60,  200),
}

CSV_HEADER = ["image_path", "overall_grade", "centering", "corners", "edges", "surface"]


def _generate_slab_image(grade: int, variant: int) -> Image.Image:
    """Generate a synthetic PSA slab photo. Shape: (312, 224, 3) RGB."""
    img = Image.new("RGB", (_WIDTH, _HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    base = _GRADE_COLORS[grade]
    r = min(255, max(0, base[0] + (variant % 20) - 10))
    g = min(255, max(0, base[1] + (variant % 15) - 7))
    b = min(255, max(0, base[2] + (variant % 25) - 12))
    draw.rectangle([0, 0, _WIDTH, _CARD_HEIGHT], fill=(r, g, b))

    # Add noise so Laplacian variance > 0 (passes quality filter)
    arr = np.array(img)
    noise = np.random.randint(-15, 15, (_CARD_HEIGHT, _WIDTH, 3), dtype=np.int16)
    arr[:_CARD_HEIGHT] = np.clip(arr[:_CARD_HEIGHT].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Label region — white background with grade text
    draw.rectangle([0, _CARD_HEIGHT, _WIDTH, _HEIGHT], fill=(255, 255, 255))
    label = f"PSA {grade}"
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tx = (_WIDTH - (bbox[2] - bbox[0])) // 2
    ty = _CARD_HEIGHT + (_LABEL_HEIGHT - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
    return img


def _subgrades(grade: int) -> tuple[float, float, float, float]:
    """Generate plausible subgrades around the overall grade."""
    def clamp(v: float) -> float:
        return max(1.0, min(10.0, v))
    base = float(grade)
    return (
        clamp(base + random.uniform(-1.0, 1.0)),
        clamp(base + random.uniform(-1.0, 1.0)),
        clamp(base + random.uniform(-1.0, 1.0)),
        clamp(base + random.uniform(-1.0, 1.0)),
    )


def generate(output_dir: Path, manifest_path: Path, images_per_grade: int, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for grade in range(1, 11):
            for i in range(images_per_grade):
                cert = f"synth_{grade:02d}_{i:04d}"
                img_path = output_dir / f"{cert}.jpg"
                _generate_slab_image(grade, variant=i).save(img_path, format="JPEG", quality=90)

                c, co, e, s = _subgrades(grade)
                writer.writerow([str(img_path), grade, round(c, 1), round(co, 1), round(e, 1), round(s, 1)])
                total += 1

    print(f"Generated {total} synthetic images → {output_dir}")
    print(f"Manifest written → {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PSA slab images")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_slabs/"))
    parser.add_argument("--manifest-path", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--images-per-grade", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output_dir, args.manifest_path, args.images_per_grade, args.seed)
