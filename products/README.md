# Catalog

This folder holds the 10-product catalog used by the **Catalog recommendations** tab.

## Files

- `catalog.json` — list of product metadata. One entry per product.
- `raw/{id}.jpg` — original product photos (one per product). **You provide these.**
- `cleaned/{id}.png` — pre-extracted flat-lay versions, used by the try-on pipeline. **Generated once by `scripts/clean_catalog.py`, then committed to the repo.**

## Schema for each catalog entry

```json
{
  "id": "P01",
  "name": "Floral Maxi Dress",
  "size": "M",
  "color": "pink",
  "image": "products/raw/P01.jpg"
}
```

- `id` — unique short code per product (used in filenames). Letters/digits only, no spaces.
- `name` — display name shown to users.
- `size` — single size string (e.g., `XS`, `S`, `M`, `L`, `XL`). Listed in the form's size dropdown.
- `color` — single color name (e.g., `pink`, `navy`, `mustard`). Listed in the form's color dropdown.
- `image` — relative path from repo root to the raw product image.

## Setup

1. Replace the 10 placeholder entries in `catalog.json` with your real products (or keep these and supply matching images).
2. Drop each product's photo at `products/raw/{id}.jpg`.
3. Run the one-time cleaner:

   ```bash
   uv run python scripts/clean_catalog.py
   ```

   This calls Gemini Stage-1 for each product and saves the cleaned flat-lay at `products/cleaned/{id}.png`. The script is idempotent — it skips products whose cleaned file already exists. Expected cost: ~$0.07 per product, ~$0.70 total.

4. Commit `products/cleaned/*.png` so other clones don't have to re-run the cleaner.
