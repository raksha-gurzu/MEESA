"""One-time helper: pre-extract clean flat-lay images for every catalog product.

For each product in `products/catalog.json`, this script:
  1. Loads the raw product image from products/raw/{id}.jpg.
  2. Calls Gemini Stage-1 (extract_garment) to render a clean flat-lay.
  3. Saves the result at products/cleaned/{id}.png.

Idempotent: if products/cleaned/{id}.png already exists, that product is
skipped. Re-run after replacing raw/ images or after deleting a cleaned/ file.

Usage:
    uv run python scripts/clean_catalog.py

The cleaned/ PNGs should be committed so that other clones of the repo skip
this work entirely. Expected cost: ~$0.07 per product (~$0.70 for 10 products).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root is the parent of /scripts; make repo modules importable when this
# script is run directly via `python scripts/clean_catalog.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from catalog import CLEANED_DIR, load_catalog  # noqa: E402
from llm import extract_garment  # noqa: E402


def main() -> int:
    catalog = load_catalog()
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Catalog has {len(catalog)} products. Cleaned dir: {CLEANED_DIR}")

    skipped = 0
    cleaned_count = 0
    failed: list[str] = []

    for product in catalog:
        out_path = product.cleaned_path
        if out_path.exists():
            print(f"  [skip] {product.id}: already cleaned at {out_path}")
            skipped += 1
            continue
        if not product.raw_path.exists():
            print(f"  [MISS] {product.id}: raw image not found at {product.raw_path}")
            failed.append(product.id)
            continue

        print(f"  [run ] {product.id}: extracting from {product.raw_path} ...")
        try:
            raw = Image.open(product.raw_path)
        except Exception as e:
            print(f"  [FAIL] {product.id}: couldn't open raw image: {e}")
            failed.append(product.id)
            continue

        try:
            cleaned, status = extract_garment(raw)
        except Exception as e:
            print(f"  [FAIL] {product.id}: extract_garment crashed: {e}")
            failed.append(product.id)
            continue

        if status == "fallback":
            print(f"  [FAIL] {product.id}: extraction returned fallback; not saving")
            failed.append(product.id)
            continue

        try:
            cleaned.save(out_path, format="PNG")
            print(f"  [done] {product.id} -> {out_path} (status={status})")
            cleaned_count += 1
        except Exception as e:
            print(f"  [FAIL] {product.id}: save to {out_path} failed: {e}")
            failed.append(product.id)

    print()
    print(f"Summary: {cleaned_count} cleaned, {skipped} skipped, {len(failed)} failed")
    if failed:
        print(f"Failed IDs: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
