"""Catalog loader, dropdown derivation, and rule-based fallback ranker.

The catalog is a small JSON file (`products/catalog.json`) listing the store's
products. Each entry has: id, name, size, color, image (path relative to repo root).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CATALOG_PATH = Path("products/catalog.json")
CLEANED_DIR = Path("products/cleaned")
REQUIRED_FIELDS = ("id", "name", "size", "color", "image")


class CatalogError(Exception):
    """Raised when the catalog file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    size: str
    color: str
    image: str  # relative path to raw product image

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def cleaned_path(self) -> Path:
        return CLEANED_DIR / f"{self.id}.png"

    @property
    def raw_path(self) -> Path:
        return Path(self.image)


def load_catalog(path: Path = CATALOG_PATH) -> list[Product]:
    if not path.exists():
        raise CatalogError(
            f"Catalog not found at {path}. See products/README.md for setup."
        )
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise CatalogError(f"{path} is not valid JSON: {e}") from e

    if not isinstance(raw, list):
        raise CatalogError(f"{path} must contain a JSON array at the top level.")

    products: list[Product] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise CatalogError(f"Entry {i} in {path} is not an object.")
        missing = [k for k in REQUIRED_FIELDS if k not in entry]
        if missing:
            raise CatalogError(
                f"Entry {i} in {path} is missing required fields: {missing}"
            )
        pid = str(entry["id"]).strip()
        if not pid:
            raise CatalogError(f"Entry {i} has an empty id.")
        if pid in seen_ids:
            raise CatalogError(f"Duplicate product id {pid!r} in {path}.")
        seen_ids.add(pid)
        products.append(
            Product(
                id=pid,
                name=str(entry["name"]).strip(),
                size=str(entry["size"]).strip(),
                color=str(entry["color"]).strip(),
                image=str(entry["image"]).strip(),
            )
        )

    if not products:
        raise CatalogError(f"{path} contains no products.")
    return products


def derive_sizes(catalog: list[Product]) -> list[str]:
    """Unique sizes in catalog order of first appearance (case-insensitive dedup)."""
    seen: dict[str, str] = {}
    for p in catalog:
        key = p.size.lower()
        if key not in seen:
            seen[key] = p.size
    return list(seen.values())


def derive_colors(catalog: list[Product]) -> list[str]:
    """Unique colors in catalog order of first appearance (case-insensitive dedup)."""
    seen: dict[str, str] = {}
    for p in catalog:
        key = p.color.lower()
        if key not in seen:
            seen[key] = p.color
    return list(seen.values())


def rule_fallback_top3(
    catalog: list[Product],
    user_attrs: dict,
) -> list[tuple[Product, str]]:
    """Deterministic top-3 picker used when the LLM ranker fails.

    Priority matches the LLM-ranker prompt (kind of outfit dominates):
      1. Style match (the user's `name` keyword appears in the product name).
      2. Size match (exact).
      3. Color match (exact).

    Each product is scored — style is weighted 1000, size 100, color 10 — and
    the top 3 are returned. Style match always outweighs size+color combined,
    so a style-matching wrong-size-wrong-color product beats an off-style
    perfect-size-perfect-color product. Ties break alphabetically by name.

    Returns: list of (Product, reason) pairs, exactly 3 entries (assuming the
    catalog has at least 3 products).
    """
    want_size = str(user_attrs.get("size", "")).strip().lower()
    want_color = str(user_attrs.get("color", "")).strip().lower()
    want_name = str(user_attrs.get("name", "")).strip().lower()
    name_tokens = [t for t in want_name.split() if t]

    def matches_size(p: Product) -> bool:
        return bool(want_size) and p.size.lower() == want_size

    def matches_color(p: Product) -> bool:
        return bool(want_color) and p.color.lower() == want_color

    def matches_style(p: Product) -> bool:
        if not name_tokens:
            return False
        product_name_lower = p.name.lower()
        return any(tok in product_name_lower for tok in name_tokens)

    def score(p: Product) -> int:
        s = 0
        if matches_style(p):
            s += 1000
        if matches_size(p):
            s += 100
        if matches_color(p):
            s += 10
        return s

    ranked = sorted(catalog, key=lambda p: (-score(p), p.name.lower()))

    picks: list[tuple[Product, str]] = []
    for p in ranked[:3]:
        style_ok = matches_style(p)
        size_ok = matches_size(p)
        color_ok = matches_color(p)
        if style_ok and size_ok and color_ok:
            reason = f"Matches your '{want_name}' query in size {p.size}, {p.color} color."
        elif style_ok and size_ok:
            reason = (
                f"Matches your '{want_name}' query in size {p.size}; "
                f"color is {p.color} (you asked for {user_attrs.get('color')})."
            )
        elif style_ok and color_ok:
            reason = (
                f"Matches your '{want_name}' query in {p.color}; "
                f"size is {p.size} (you asked for {user_attrs.get('size')})."
            )
        elif style_ok:
            reason = (
                f"Matches your '{want_name}' query (size {p.size}, color {p.color} "
                f"differ from your size/color preferences)."
            )
        elif size_ok and color_ok:
            reason = (
                f"No catalog item matches your '{want_name}' query, so showing "
                f"size {p.size}, {p.color} as the closest available."
            )
        elif size_ok:
            reason = (
                f"Off-style fallback — size {p.size} matches but "
                f"this isn't a '{want_name}' style."
            )
        elif color_ok:
            reason = (
                f"Off-style fallback — {p.color} color matches but "
                f"this isn't a '{want_name}' style."
            )
        else:
            reason = f"Closest available — size {p.size}, color {p.color}."
        picks.append((p, reason))

    return picks
