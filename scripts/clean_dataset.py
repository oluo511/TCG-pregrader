"""
Dataset cleaning pipeline — run before training.

Two-pass filter:
  1. Dedup by image hash — removes same image with conflicting grade labels.
     Keeps the row whose grade matches the majority vote across all occurrences.
     Ties are broken by keeping the highest grade.

  2. Slab detector — removes non-slab images using CV heuristics tuned to
     the specific noise types observed in eBay scraping:
       - PSA storage boxes (landscape, dark red, no card)
       - Binder pages with multiple slabs (multiple rectangles, dark bg)
       - Card lots / loose cards (no white slab border)
       - Booster packs (portrait but colorful, no white border)
       - Tiny thumbnails < 100px (unreliable for any detection)

     A real PSA slab has:
       - Portrait orientation (taller than wide)
       - Significant white/near-white region (slab casing, >10% of pixels)
       - Single dominant rectangular contour (the slab itself)
       - Not predominantly dark (storage box filter)

Usage:
    python scripts/clean_dataset.py
    python scripts/clean_dataset.py --manifest data/manifest.csv --output data/manifest_clean.csv
    python scripts/clean_dataset.py --skip-slab-filter   # dedup only
"""

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Pass 1 — Deduplicate by image content hash
# ---------------------------------------------------------------------------

def _image_hash(path: Path) -> str | None:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return None


def dedup_by_hash(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep one row per unique image using majority-vote grade resolution."""
    hash_to_rows: dict[str, list[dict]] = defaultdict(list)
    no_hash: list[dict] = []

    for row in rows:
        h = _image_hash(Path(row["image_path"]))
        if h is None:
            no_hash.append(row)
        else:
            hash_to_rows[h].append(row)

    clean: list[dict] = []
    removed = 0

    for h, group in hash_to_rows.items():
        if len(group) == 1:
            clean.append(group[0])
            continue
        grade_counts = Counter(int(r["overall_grade"]) for r in group)
        max_count = max(grade_counts.values())
        candidates = [g for g, c in grade_counts.items() if c == max_count]
        winner_grade = max(candidates)
        winner = next(r for r in group if int(r["overall_grade"]) == winner_grade)
        clean.append(winner)
        removed += len(group) - 1

    return clean + no_hash, removed


# ---------------------------------------------------------------------------
# Pass 2 — Slab detector
# ---------------------------------------------------------------------------

def _is_slab(path: Path) -> tuple[bool, str]:
    """
    Return (is_slab, reason) using layered CV checks.

    Checks in order (fail-fast):
      1. Minimum size — thumbnails < 80px wide are unreliable
      2. Portrait orientation — slabs are taller than wide
      3. Not predominantly dark — filters storage boxes, dark backgrounds
      4. White border fraction — PSA casing is white/off-white
      5. Single dominant contour — filters multi-slab binder pages and lots
    """
    try:
        data = path.read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return False, "decode_failed"

    if img is None:
        return False, "decode_failed"

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return False, "zero_dimension"

    # --- Check 1: Minimum size ---
    # Thumbnails smaller than 80px wide don't have enough pixels for reliable
    # border detection. These are almost always noise from eBay's grid view.
    if w < 80 or h < 80:
        return False, f"too_small={w}x{h}"

    # --- Check 2: Portrait orientation ---
    # PSA slabs are always portrait (h > w). Storage boxes and card lots
    # are typically landscape or square.
    ratio = w / h
    if ratio > 0.92:
        return False, f"not_portrait={ratio:.2f}"

    # --- Check 3: Not predominantly dark ---
    # PSA storage boxes are dark red/maroon. Card lot photos often have
    # dark table backgrounds. A real slab photo has a bright white casing.
    # Check mean brightness of the full image.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness < 60:
        return False, f"too_dark={mean_brightness:.0f}"

    # --- Check 4: White border fraction ---
    # The PSA slab casing is white/off-white. We look for pixels with
    # high brightness (V > 160) and low saturation (S < 50) in HSV space.
    # Threshold is lower for smaller images where the border is fewer pixels.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 160)
    white_frac = float(white_mask.sum()) / (h * w)

    # Scale threshold by image size — small images have proportionally
    # thinner borders. At 105x140 a 3px border is ~4% of pixels.
    min_white = 0.06 if w < 150 else 0.10
    if white_frac < min_white:
        return False, f"no_white_border={white_frac:.3f}"

    # --- Check 5: Single dominant contour ---
    # A single slab should produce one large rectangular contour.
    # Binder pages (multiple slabs) and card lots produce many contours.
    # We threshold on the number of significant contours in the image.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count contours that are at least 5% of the image area
    min_area = h * w * 0.05
    significant = [c for c in contours if cv2.contourArea(c) >= min_area]

    # More than 4 significant contours = multiple objects (binder, lot, etc.)
    if len(significant) > 4:
        return False, f"multi_object={len(significant)}_contours"

    return True, "ok"


def filter_slabs(rows: list[dict], preview: int = 0) -> tuple[list[dict], int]:
    slab_rows: list[dict] = []
    rejected: list[tuple[dict, str]] = []
    total = len(rows)

    for i, row in enumerate(rows):
        if i % 200 == 0:
            print(f"  Checking {i}/{total}...", end="\r")
        is_slab, reason = _is_slab(Path(row["image_path"]))
        if is_slab:
            slab_rows.append(row)
        else:
            rejected.append((row, reason))

    print(f"  Checked {total}/{total}    ")

    # Rejection breakdown
    reasons = Counter(reason.split("=")[0] for _, reason in rejected)
    print("  Rejection reasons:")
    for reason, count in reasons.most_common():
        print(f"    {reason}: {count}")

    if preview > 0 and rejected:
        print(f"\nSample rejected (first {min(preview, len(rejected))}):")
        for row, reason in rejected[:preview]:
            print(f"  {Path(row['image_path']).name} g={row['overall_grade']} [{reason}]")

    return slab_rows, len(rejected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean training manifest")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifest_clean.csv"))
    parser.add_argument("--preview", type=int, default=10)
    parser.add_argument("--skip-slab-filter", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.manifest}...")
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    print(f"  {len(rows)} rows, {len(set(r['image_path'] for r in rows))} unique images\n")

    # Pass 1: Dedup
    print("Pass 1: Deduplicating by image hash...")
    rows, n_dedup = dedup_by_hash(rows)
    print(f"  Removed {n_dedup} duplicate rows -> {len(rows)} remaining\n")

    # Pass 2: Slab filter
    if not args.skip_slab_filter:
        print("Pass 2: Slab detection...")
        rows, n_filtered = filter_slabs(rows, preview=args.preview)
        print(f"  Removed {n_filtered} non-slabs -> {len(rows)} remaining\n")

    # Grade distribution
    grade_counts = Counter(int(r["overall_grade"]) for r in rows)
    print("Grade distribution after cleaning:")
    for g in range(1, 11):
        bar = "#" * (grade_counts[g] // 5)
        print(f"  Grade {g:>2}: {grade_counts[g]:>4}  {bar}")
    print(f"  TOTAL:  {sum(grade_counts.values())}\n")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written to {args.output}")
    print("\nNext step: train on the clean manifest:")
    print("  python scripts/smoke_test_training.py  (quick validation)")
    print("  # then update MANIFEST_PATH in smoke_test to data/manifest_clean.csv")


if __name__ == "__main__":
    main()
