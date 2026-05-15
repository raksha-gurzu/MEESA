"""Gemini client wrapper for virtual try-on generation and catalog ranking.

Try-on pipeline (Nano Banana 2):
  1. extract_garment(outfit) -> clean flat-lay garment image (cached on disk)
  2. neutralize_person(person) -> body-canvas rendering (cached on disk)
  3. generate_tryon(person, outfit) -> final try-on result

Catalog recommender:
  - rank_products(catalog, user_attrs) -> top 3 (id + reason) via Flash multimodal
  - generate_tryon_batch(person, products) -> 3 parallel try-ons for the picks
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps

from catalog import Product
from prompt import (
    EXTRACT_GARMENT_INSTRUCTION,
    NEUTRALIZE_PERSON_INSTRUCTION,
    RANK_PRODUCTS_INSTRUCTION,
    TRY_ON_INSTRUCTION,
)

load_dotenv()

# Diagnostic print that always flushes — stdout is block-buffered under Streamlit.
log = functools.partial(print, flush=True)

MODEL_ID = "gemini-3.1-flash-image-preview"
# Ranker is a text-output multimodal model. Pinned for one-line swap to Pro
# (e.g. "gemini-2.5-pro") if Flash-tier ranking quality disappoints.
RANKER_MODEL_ID = "gemini-2.5-flash"
MAX_EDGE = 1536
TIMEOUT_MS = 120_000
CACHE_DIR = Path("images/cache")

_SUPPORTED_RATIOS: list[tuple[str, float]] = [
    ("21:9", 21 / 9),
    ("16:9", 16 / 9),
    ("3:2", 3 / 2),
    ("4:3", 4 / 3),
    ("1:1", 1.0),
    ("3:4", 3 / 4),
    ("2:3", 2 / 3),
    ("9:16", 9 / 16),
]

_SAFETY_FINISH_REASONS = {
    "SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII",
    "IMAGE_SAFETY", "IMAGE_PROHIBITED",
}
_NO_IMAGE_FINISH_REASONS = {
    "NO_IMAGE", "IMAGE_OTHER", "OTHER", "LANGUAGE", "MALFORMED_FUNCTION_CALL",
}


class TryOnError(Exception):
    """Friendly error carrying a user-facing message for st.error()."""


class RankerError(Exception):
    """Raised when the LLM ranker call fails after all retries.

    Caller is expected to fall back to catalog.rule_fallback_top3().
    """


@dataclass(frozen=True)
class RankingPick:
    product: Product
    reason: str


@dataclass(frozen=True)
class BatchResult:
    """One slot in a 3-up try-on batch. Exactly one of `image` / `error` is set."""
    product: Product
    reason: str
    image: Image.Image | None
    error: str | None  # user-facing error message; None on success


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise TryOnError("GEMINI_API_KEY not set. Add it to .env.")
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=TIMEOUT_MS),
        )
    return _client


def preprocess(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_EDGE:
        scale = MAX_EDGE / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _nearest_aspect_ratio(img: Image.Image) -> str:
    w, h = img.size
    target = w / h
    label, _ = min(_SUPPORTED_RATIOS, key=lambda r: abs(r[1] - target))
    return label


def _generate_image(
    prompt_text: str,
    images: list[Image.Image],
    aspect_ratio: str,
    label: str,
) -> Image.Image:
    """Single Gemini image-generation call. Returns the produced PIL image.

    Raises TryOnError with a user-facing message on any failure
    (network, safety block, no image, decode error).
    """
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt_text, *images],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
            ),
        )
    except Exception as e:
        log(f"[llm:{label}] Gemini call failed: {e!r}")
        raise TryOnError(
            "Couldn't reach the image service. Check your connection and try again."
        ) from e

    candidates = response.candidates or []
    if not candidates:
        raise TryOnError("The model didn't return an image. Try again, or try different inputs.")

    candidate = candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    finish_label = str(finish_reason).split(".")[-1].upper() if finish_reason else ""
    has_content = candidate.content is not None
    parts_raw = candidate.content.parts if has_content else None
    parts = parts_raw if parts_raw else []
    inline_count = sum(
        1 for p in parts
        if getattr(p, "inline_data", None) and getattr(p.inline_data, "data", None)
    )
    text_count = sum(1 for p in parts if getattr(p, "text", None))
    log(
        f"[llm:{label}] finish_reason={finish_label} content={has_content} "
        f"parts={len(parts)} inline_data_parts={inline_count} text_parts={text_count}"
    )

    if finish_label in _SAFETY_FINISH_REASONS:
        raise TryOnError(
            "This generation was blocked by content safety filters. Try a different photo or outfit."
        )

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data_len = len(inline.data) if hasattr(inline.data, "__len__") else "?"
            mime = getattr(inline, "mime_type", None)
            log(f"[llm:{label}] decoding image: mime={mime} bytes={data_len}")
            try:
                img = Image.open(io.BytesIO(inline.data))
                img.load()
                log(f"[llm:{label}] decoded: {img.size} {img.mode}")
                return img
            except Exception as e:
                log(f"[llm:{label}] Image.open/load failed: {e!r}")
                raise TryOnError(
                    f"Couldn't decode the image returned by the model: {e}"
                ) from e

    if finish_label in _NO_IMAGE_FINISH_REASONS:
        raise TryOnError(
            "The model didn't produce an image for these inputs. Try clearer photos — "
            "a real person photo and a clear outfit photo."
        )
    raise TryOnError("The model didn't return an image. Try again, or try different inputs.")


def extract_garment(outfit_img: Image.Image) -> tuple[Image.Image, str]:
    """Stage 1: extract just the garment as a clean flat-lay on white.

    Returns (cleaned_img, status) where status is one of:
      - 'cached'    — disk cache hit; no API call made.
      - 'extracted' — fresh Gemini call succeeded; result cached for next time.
      - 'fallback'  — Gemini call failed; cleaned_img is the preprocessed
                      original outfit so the caller can still proceed.
    """
    preprocessed = preprocess(outfit_img)
    cache_key = hashlib.sha256(preprocessed.tobytes()).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.png"

    if cache_path.exists():
        try:
            cached = Image.open(cache_path).convert("RGB")
            cached.load()
            log(f"[llm:extract] cache HIT {cache_path.name}")
            return cached, "cached"
        except Exception as e:
            log(f"[llm:extract] cache read failed ({e!r}); will re-extract")
            cache_path.unlink(missing_ok=True)

    aspect_ratio = _nearest_aspect_ratio(preprocessed)
    log(f"[llm:extract] cache MISS; calling Gemini (aspect_ratio={aspect_ratio})")
    try:
        cleaned = _generate_image(
            prompt_text=EXTRACT_GARMENT_INSTRUCTION,
            images=[preprocessed],
            aspect_ratio=aspect_ratio,
            label="extract",
        )
    except TryOnError as e:
        log(f"[llm:extract] FAILED ({e}); falling back to raw outfit")
        return preprocessed, "fallback"
    except Exception as e:
        log(f"[llm:extract] unexpected error ({e!r}); falling back to raw outfit")
        return preprocessed, "fallback"

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.save(cache_path, format="PNG")
        log(f"[llm:extract] cached at {cache_path.name}")
    except Exception as e:
        log(f"[llm:extract] cache write failed ({e!r}); continuing in-memory")

    return cleaned, "extracted"


def neutralize_person(person_img: Image.Image) -> tuple[Image.Image, str]:
    """Stage 2: render the person in plain form-fitting gray clothing with a neutral pose.

    Commits to the person's ACTUAL body shape, free from the original outfit's silhouette.

    Returns (neutralized_img, status) where status is one of:
      - 'cached'    — disk cache hit; no API call made.
      - 'extracted' — fresh Gemini call succeeded; result cached.
      - 'fallback'  — Gemini call failed; neutralized_img is the preprocessed
                      original person photo so the caller can still proceed.
    """
    preprocessed = preprocess(person_img)
    cache_key = hashlib.sha256(preprocessed.tobytes()).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.png"

    if cache_path.exists():
        try:
            cached = Image.open(cache_path).convert("RGB")
            cached.load()
            log(f"[llm:neutralize] cache HIT {cache_path.name}")
            return cached, "cached"
        except Exception as e:
            log(f"[llm:neutralize] cache read failed ({e!r}); will re-run")
            cache_path.unlink(missing_ok=True)

    aspect_ratio = _nearest_aspect_ratio(preprocessed)
    log(f"[llm:neutralize] cache MISS; calling Gemini (aspect_ratio={aspect_ratio})")
    try:
        neutralized = _generate_image(
            prompt_text=NEUTRALIZE_PERSON_INSTRUCTION,
            images=[preprocessed],
            aspect_ratio=aspect_ratio,
            label="neutralize",
        )
    except TryOnError as e:
        log(f"[llm:neutralize] FAILED ({e}); falling back to raw person photo")
        return preprocessed, "fallback"
    except Exception as e:
        log(f"[llm:neutralize] unexpected error ({e!r}); falling back to raw person photo")
        return preprocessed, "fallback"

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        neutralized.save(cache_path, format="PNG")
        log(f"[llm:neutralize] cached at {cache_path.name}")
    except Exception as e:
        log(f"[llm:neutralize] cache write failed ({e!r}); continuing in-memory")

    return neutralized, "extracted"


def generate_tryon(
    person_img: Image.Image,
    outfit_img: Image.Image,
) -> Image.Image:
    """Stage 3: compose the final try-on image.

    Args:
        person_img: typically the neutralized person (Stage 2 output).
        outfit_img: the cleaned outfit (Stage 1 output from extract_garment),
            or the raw outfit if Stage 1 fell back.

    Raises TryOnError with a user-facing message on any failure.
    """
    person_img = preprocess(person_img)
    outfit_img = preprocess(outfit_img)
    aspect_ratio = _nearest_aspect_ratio(person_img)
    return _generate_image(
        prompt_text=TRY_ON_INSTRUCTION,
        images=[person_img, outfit_img],
        aspect_ratio=aspect_ratio,
        label="tryon",
    )


# ---------------------------------------------------------------------------
# Catalog ranker (Flash multimodal -> structured JSON)
# ---------------------------------------------------------------------------

_RANKER_SCHEMA = {
    "type": "object",
    "properties": {
        "top_3": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "reason"],
            },
        },
    },
    "required": ["top_3"],
}


def _load_product_image(p: Product) -> Image.Image:
    """Open a product's raw image, preprocess, return RGB PIL.Image."""
    if not p.raw_path.exists():
        raise RankerError(
            f"Product image not found: {p.raw_path}. Add it to products/raw/ "
            "(see products/README.md)."
        )
    img = Image.open(p.raw_path)
    return preprocess(img)


def _call_ranker(
    catalog: list[Product],
    user_attrs: dict,
    product_images: list[Image.Image],
    stricter: bool,
) -> list[RankingPick]:
    """Single ranker attempt. Raises RankerError on any failure path."""
    client = _get_client()

    prompt_text = RANK_PRODUCTS_INSTRUCTION
    if stricter:
        prompt_text += (
            "\n\nIMPORTANT — your previous response could not be parsed. Return ONLY "
            "valid JSON matching the schema. The list MUST have exactly 3 items, and "
            "every id MUST exist in the catalog above."
        )

    contents: list = [
        prompt_text,
        "\n\nUSER REQUEST:\n" + json.dumps(user_attrs),
        "\n\nCATALOG (each entry's JSON is followed by its image):",
    ]
    for product, image in zip(catalog, product_images):
        contents.append("\n" + json.dumps(product.as_dict()))
        contents.append(image)

    try:
        response = client.models.generate_content(
            model=RANKER_MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RANKER_SCHEMA,
            ),
        )
    except Exception as e:
        log(f"[llm:rank] Gemini call failed: {e!r}")
        raise RankerError(f"Ranker call failed: {e}") from e

    text = (getattr(response, "text", None) or "").strip()
    log(f"[llm:rank] raw response len={len(text)}")
    if not text:
        raise RankerError("Ranker returned empty response.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        log(f"[llm:rank] JSON parse failed: {e!r}; text head={text[:200]!r}")
        raise RankerError(f"Ranker returned non-JSON: {e}") from e

    raw_picks = parsed.get("top_3")
    if not isinstance(raw_picks, list) or len(raw_picks) != 3:
        raise RankerError(
            f"Expected exactly 3 picks in top_3, got: {type(raw_picks).__name__} "
            f"with len={len(raw_picks) if isinstance(raw_picks, list) else 'n/a'}"
        )

    by_id = {p.id: p for p in catalog}
    picks: list[RankingPick] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw_picks):
        if not isinstance(entry, dict):
            raise RankerError(f"Pick {i} is not an object: {entry!r}")
        pid = str(entry.get("id", "")).strip()
        reason = str(entry.get("reason", "")).strip()
        if not pid or pid not in by_id:
            raise RankerError(f"Pick {i} has invalid id {pid!r} (not in catalog).")
        if pid in seen_ids:
            raise RankerError(f"Pick {i} is a duplicate of an earlier pick: {pid!r}.")
        if not reason:
            reason = "Recommended for you."
        seen_ids.add(pid)
        picks.append(RankingPick(product=by_id[pid], reason=reason))

    log(f"[llm:rank] valid picks: {[(p.product.id, p.reason[:40]) for p in picks]}")
    return picks


def rank_products(
    catalog: list[Product],
    user_attrs: dict,
) -> list[RankingPick]:
    """Ask Gemini Flash to pick the top 3 products from the catalog.

    Sends user_attrs JSON + each product's JSON interleaved with its image
    as a multimodal contents list. Forces structured JSON output via
    response_schema; validates returned IDs against the catalog.

    Retries once with a stricter prompt on any failure (parse error,
    missing/extra picks, hallucinated id, etc). Raises RankerError if both
    attempts fail — the caller should fall back to rule_fallback_top3.
    """
    log(f"[llm:rank] catalog={len(catalog)} products, user_attrs={user_attrs}")
    product_images = [_load_product_image(p) for p in catalog]
    log(f"[llm:rank] loaded {len(product_images)} product images for ranker call")

    try:
        return _call_ranker(catalog, user_attrs, product_images, stricter=False)
    except RankerError as first_err:
        log(f"[llm:rank] first attempt failed: {first_err}; retrying with stricter prompt")
    try:
        return _call_ranker(catalog, user_attrs, product_images, stricter=True)
    except RankerError as second_err:
        log(f"[llm:rank] retry also failed: {second_err}")
        raise


# ---------------------------------------------------------------------------
# Parallel try-on for top-3 catalog products
# ---------------------------------------------------------------------------

def _load_cleaned_catalog_image(product: Product) -> Image.Image:
    """Load the pre-extracted flat-lay for a catalog product. Falls back to
    the raw product image (with preprocessing) if cleaned/ is missing."""
    if product.cleaned_path.exists():
        img = Image.open(product.cleaned_path).convert("RGB")
        img.load()
        log(f"[llm:batch] using cleaned image for {product.id}")
        return img
    log(
        f"[llm:batch] cleaned/{product.id}.png missing; falling back to raw image. "
        "Run scripts/clean_catalog.py to pre-extract."
    )
    return _load_product_image(product)


def _tryon_one(
    person_neutralized: Image.Image,
    pick: RankingPick,
) -> BatchResult:
    """Worker for the ThreadPoolExecutor. Always returns a BatchResult; catches
    every exception so a single failure doesn't take down the whole batch."""
    try:
        cleaned = _load_cleaned_catalog_image(pick.product)
        image = generate_tryon(person_neutralized, cleaned)
        return BatchResult(product=pick.product, reason=pick.reason, image=image, error=None)
    except TryOnError as e:
        log(f"[llm:batch] TryOnError for {pick.product.id}: {e}")
        return BatchResult(product=pick.product, reason=pick.reason, image=None, error=str(e))
    except Exception as e:
        log(f"[llm:batch] unexpected error for {pick.product.id}: {e!r}")
        return BatchResult(
            product=pick.product,
            reason=pick.reason,
            image=None,
            error="Try-on couldn't be generated for this product. Try again.",
        )


def tryon_single_pick(
    person_neutralized: Image.Image,
    pick: RankingPick,
) -> BatchResult:
    """Run Stage 3 once for a single pick, given the already-neutralized person.

    Skips Stage 2 (caller supplies the neutralized image).
    Skips Stage 1 (uses products/cleaned/{id}.png; falls back to raw if missing).
    Never raises — wraps errors in BatchResult.error so a single failure doesn't
    take down the whole batch.

    Used directly for the per-card 'Regenerate this one' button, and via
    tryon_batch() for the initial 3-up generation.
    """
    return _tryon_one(person_neutralized, pick)


def tryon_batch(
    person_neutralized: Image.Image,
    picks: list[RankingPick],
) -> list[BatchResult]:
    """Run Stage 3 in parallel for each pick.

    Stage 2 is the caller's responsibility — pass an already-neutralized person
    image so this function only performs the Stage-3 fan-out.

    Returns: one BatchResult per pick (order matches input order).
    """
    log(f"[llm:batch] starting parallel try-on for {len(picks)} picks")
    with ThreadPoolExecutor(max_workers=max(1, len(picks))) as executor:
        results = list(
            executor.map(lambda pick: _tryon_one(person_neutralized, pick), picks)
        )
    log(
        f"[llm:batch] done: "
        f"{sum(1 for r in results if r.image is not None)}/{len(results)} succeeded"
    )
    return results
