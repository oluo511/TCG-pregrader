"""
Build manifest.csv from a folder of manually collected slab images.

Expected folder structure:
    data/raw_slabs/
        grade_1/   ← images of PSA 1 slabs
        grade_2/
        ...
        grade_10/

Usage:
    python scripts/build_manifest_from_folder.py
    python scripts/build_manifest_from_folder.py --input data/raw_slabs --output data/manifest.csv

Each image gets a row: image_path, overall_grade
Existing manifest.csv is overwritten.
"""

import csv
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def build_manifest(input_dir: Path, output_path: Path) -> None:
    rows: list[dict] = []

    for grade in range(1, 11):
        grade_dir = input_dir / f"grade_{grade}"
        if not grade_dir.exists():
            print(f"  [skip] {grade_dir} not found")
            continue

        images = [
            f for f in sorted(grade_dir.iterdir())
            if f.suffix.lower() in SUPPORTED_EXTS
        ]
        print(f"  grade {grade:>2}: {len(images)} images")

        for img_path in images:
            rows.append({
                "image_path": str(img_path.relative_to(input_dir.parent)),
                "overall_grade": grade,
            })

    if not rows:
        print("\nNo images found. Add images to data/raw_slabs/grade_X/ folders.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "overall_grade"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build manifest.csv from grade folders")
    parser.add_argument("--input", type=Path, default=Path("data/raw_slabs"))
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    args = parser.parse_args()

    print(f"Scanning {args.input}...\n")
    build_manifest(args.input, args.output)


if __name__ == "__main__":
    main()
