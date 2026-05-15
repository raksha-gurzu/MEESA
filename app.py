"""Meesa Try-On — Streamlit app for virtual try-on via Gemini Nano Banana 2.

Two tabs:
  1. Manual try-on: upload a person photo + an outfit photo, run try-on.
  2. Catalog recommendations: pick size + color, type a query, upload a photo.
     Gemini ranks the catalog and runs try-on on the top 3 in parallel.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from catalog import (
    CatalogError,
    Product,
    derive_colors,
    derive_sizes,
    load_catalog,
    rule_fallback_top3,
)
from llm import (
    BatchResult,
    RankerError,
    RankingPick,
    TryOnError,
    extract_garment,
    generate_tryon,
    neutralize_person,
    rank_products,
    tryon_batch,
    tryon_single_pick,
)

load_dotenv()

UPLOAD_DIR = Path("images")
OUTPUT_DIR = Path("generated")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Meesa Try-On", page_icon="👗", layout="wide")

st.title("Meesa Try-On")
st.caption(
    "Use only photos you have permission to use. "
    "Results include an invisible SynthID watermark from Google."
)

if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "GEMINI_API_KEY not set. Add it to `.env` — see `.env.example`. "
        "Get a key from https://aistudio.google.com/apikey"
    )
    st.stop()


def _timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"


def _open_upload(uploaded_file) -> Image.Image:
    data = uploaded_file.getvalue()
    return Image.open(io.BytesIO(data))


def _save_png(img: Image.Image, path: Path) -> None:
    img.save(path, format="PNG")


# Session-state defaults. Manual-tab keys are unprefixed (existing flow).
# Catalog-tab keys are prefixed `cat_` to avoid collisions.
for key, default in [
    ("person_bytes", None),
    ("outfit_bytes", None),
    ("result_image", None),
    ("result_path", None),
    ("outfit_status", None),
    ("person_status", None),
    ("is_generating", False),
    # catalog tab
    ("cat_person_bytes", None),
    ("cat_results", None),  # list[BatchResult]
    ("cat_neutralized", None),  # PIL.Image used for per-card regenerate
    ("cat_neutralize_status", None),
    ("cat_ranker_status", None),  # 'ok' | 'fallback'
    ("cat_ts", None),
    ("cat_is_generating", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# Load catalog once at the top so both the recommendations tab and the
# browse tab can share it; tolerate failure so the manual tab still works.
try:
    _catalog = load_catalog()
    _catalog_error: str | None = None
except CatalogError as e:
    _catalog = None
    _catalog_error = str(e)

tab_manual, tab_catalog, tab_browse = st.tabs(
    ["Manual try-on", "Catalog recommendations", "Browse catalog"]
)


# ===========================================================================
# Tab 1 — Manual try-on (existing flow, unchanged)
# ===========================================================================

with tab_manual:
    upload_col, outfit_col, result_col = st.columns(3)

    with upload_col:
        st.subheader("Person photo")
        person_file = st.file_uploader(
            "Upload a photo of the person",
            type=["png", "jpg", "jpeg", "webp"],
            key="person_uploader",
            disabled=st.session_state.is_generating,
            label_visibility="collapsed",
        )
        st.caption(
            "Tip: a full-body or waist-up photo works best. "
            "For best results, just one person in frame."
        )
        if person_file is not None:
            new_bytes = person_file.getvalue()
            if new_bytes != st.session_state.person_bytes:
                st.session_state.person_bytes = new_bytes
                st.session_state.result_image = None
                st.session_state.result_path = None
                st.session_state.outfit_status = None
                st.session_state.person_status = None
        if st.session_state.person_bytes:
            try:
                st.image(st.session_state.person_bytes, width="stretch")
            except Exception:
                st.error("Couldn't read that image. Try a different file.")
                st.session_state.person_bytes = None

    with outfit_col:
        st.subheader("Outfit photo")
        outfit_file = st.file_uploader(
            "Upload a photo of the outfit",
            type=["png", "jpg", "jpeg", "webp"],
            key="outfit_uploader",
            disabled=st.session_state.is_generating,
            label_visibility="collapsed",
        )
        st.caption(
            "Tip: works best with a clear photo of the outfit. "
            "Either a product shot or someone wearing it is fine."
        )
        if outfit_file is not None:
            new_bytes = outfit_file.getvalue()
            if new_bytes != st.session_state.outfit_bytes:
                st.session_state.outfit_bytes = new_bytes
                st.session_state.result_image = None
                st.session_state.result_path = None
                st.session_state.outfit_status = None
                st.session_state.person_status = None
        if st.session_state.outfit_bytes:
            try:
                st.image(st.session_state.outfit_bytes, width="stretch")
            except Exception:
                st.error("Couldn't read that image. Try a different file.")
                st.session_state.outfit_bytes = None

    with result_col:
        st.subheader("Try-on result")
        if st.session_state.result_image is not None:
            st.image(st.session_state.result_image, width="stretch")
            fallback_notes = []
            if st.session_state.outfit_status == "fallback":
                fallback_notes.append("outfit preprocessing")
            if st.session_state.person_status == "fallback":
                fallback_notes.append("body-shape preprocessing")
            if fallback_notes:
                st.caption(
                    "ℹ️ "
                    + " and ".join(fallback_notes).capitalize()
                    + " was unavailable for this image; result quality may be reduced."
                )
        else:
            st.caption("Your generated try-on will appear here.")

    ready = bool(st.session_state.person_bytes and st.session_state.outfit_bytes)
    has_result = st.session_state.result_image is not None
    button_label = (
        "Generating…"
        if st.session_state.is_generating
        else ("Regenerate" if has_result else "Generate try-on")
    )

    button_col, download_col, _ = st.columns([1, 1, 4])

    with button_col:
        clicked = st.button(
            button_label,
            type="primary",
            disabled=(not ready) or st.session_state.is_generating,
            width="stretch",
            key="manual_generate_btn",
        )

    with download_col:
        if has_result and st.session_state.result_path:
            try:
                with open(st.session_state.result_path, "rb") as f:
                    st.download_button(
                        "Download",
                        f,
                        file_name=Path(st.session_state.result_path).name,
                        mime="image/png",
                        width="stretch",
                        key="manual_download_btn",
                    )
            except FileNotFoundError:
                pass

    if clicked:
        st.session_state.is_generating = True
        try:
            person_img = _open_upload(person_file or io.BytesIO(st.session_state.person_bytes))
        except Exception as e:
            st.session_state.is_generating = False
            print(f"[app] couldn't open person upload: {e!r}", flush=True)
            st.error("Couldn't read the person photo. Try a different file.")
            st.stop()
        try:
            outfit_img = _open_upload(outfit_file or io.BytesIO(st.session_state.outfit_bytes))
        except Exception as e:
            st.session_state.is_generating = False
            print(f"[app] couldn't open outfit upload: {e!r}", flush=True)
            st.error("Couldn't read the outfit photo. Try a different file.")
            st.stop()

        # Stage 1 — extract the garment as a clean flat-lay (cached by outfit hash).
        with st.spinner("Cleaning up the outfit image… (typically 15–30s)"):
            try:
                cleaned_outfit, outfit_status = extract_garment(outfit_img)
            except Exception as e:
                st.session_state.is_generating = False
                print(f"[app] unexpected error in extract_garment: {e!r}", flush=True)
                st.error("Something went wrong while preparing the outfit. Try again.")
                st.stop()

        # Stage 2 — neutralize the person to a body-canvas form (cached by person hash).
        with st.spinner("Reading your body shape… (typically 15–30s)"):
            try:
                neutralized_person, person_status = neutralize_person(person_img)
            except Exception as e:
                st.session_state.is_generating = False
                print(f"[app] unexpected error in neutralize_person: {e!r}", flush=True)
                st.error("Something went wrong while preparing the body canvas. Try again.")
                st.stop()

        # Stage 3 — main try-on using neutralized person + cleaned outfit.
        with st.spinner("Composing your try-on… (typically 15–30s)"):
            try:
                result = generate_tryon(neutralized_person, cleaned_outfit)
            except TryOnError as e:
                st.session_state.is_generating = False
                print(f"[app] TryOnError: {e}", flush=True)
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.session_state.is_generating = False
                print(f"[app] unexpected error in generate_tryon: {e!r}", flush=True)
                st.error("Something went wrong. Try again.")
                st.stop()

        try:
            ts = _timestamp()
            person_path = UPLOAD_DIR / f"{ts}_person.png"
            outfit_path = UPLOAD_DIR / f"{ts}_outfit.png"
            cleaned_outfit_path = UPLOAD_DIR / f"{ts}_outfit_cleaned.png"
            neutralized_person_path = UPLOAD_DIR / f"{ts}_person_neutralized.png"
            result_path = OUTPUT_DIR / f"{ts}_tryon.png"
            _save_png(person_img, person_path)
            _save_png(outfit_img, outfit_path)
            if outfit_status != "fallback":
                _save_png(cleaned_outfit, cleaned_outfit_path)
            if person_status != "fallback":
                _save_png(neutralized_person, neutralized_person_path)
            _save_png(result, result_path)
        except Exception as e:
            st.session_state.is_generating = False
            print(f"[app] save failed: {e!r}", flush=True)
            st.error(f"Generated but couldn't save the image: {e}")
            st.stop()

        st.session_state.result_image = result
        st.session_state.result_path = str(result_path)
        st.session_state.outfit_status = outfit_status
        st.session_state.person_status = person_status
        st.session_state.is_generating = False
        st.rerun()


# ===========================================================================
# Tab — Browse catalog (read-only product grid).
#
# Rendered HERE in script order, before tab_catalog, so the browse tab still
# shows its own error message if catalog loading fails (tab_catalog uses
# st.stop() on the catalog-missing path which would otherwise halt the
# script before this block runs).
# ===========================================================================

with tab_browse:
    if _catalog is None:
        st.error(f"Couldn't load catalog: {_catalog_error}")
    else:
        st.write(
            f"Showing all {len(_catalog)} products in the store catalog. "
            "Edit `products/catalog.json` to change the lineup."
        )
        cols_per_row = 3
        for row_start in range(0, len(_catalog), cols_per_row):
            row_products = _catalog[row_start : row_start + cols_per_row]
            row_cols = st.columns(cols_per_row)
            for col, product in zip(row_cols, row_products):
                with col:
                    if product.raw_path.exists():
                        st.image(str(product.raw_path), width="stretch")
                    else:
                        st.caption(
                            f"(image missing — add a file at `{product.raw_path}`)"
                        )
                    st.markdown(f"**{product.name}**")
                    st.markdown(
                        f"**Size:** {product.size} &nbsp;&nbsp; "
                        f"**Color:** {product.color}"
                    )
                    st.caption(f"ID: {product.id}")
            st.write("")  # vertical spacing between rows


# ===========================================================================
# Tab 2 — Catalog recommendations
# ===========================================================================

with tab_catalog:
    if _catalog is None:
        st.error(f"Couldn't load catalog: {_catalog_error}")
        st.stop()
    catalog = _catalog
    sizes = derive_sizes(catalog)
    colors = derive_colors(catalog)

    st.write(
        "Tell us what you're looking for. We'll pick the 3 closest matches "
        "from our catalog and show you how each looks on you."
    )

    form_col_a, form_col_b, form_col_c = st.columns(3)
    with form_col_a:
        cat_size = st.selectbox(
            "Size",
            options=sizes,
            key="cat_size_select",
            disabled=st.session_state.cat_is_generating,
        )
    with form_col_b:
        cat_color = st.selectbox(
            "Color",
            options=colors,
            key="cat_color_select",
            disabled=st.session_state.cat_is_generating,
        )
    with form_col_c:
        cat_name = st.text_input(
            "What kind of outfit?",
            placeholder="e.g., flowy summer dress",
            key="cat_name_input",
            disabled=st.session_state.cat_is_generating,
        )

    cat_person_file = st.file_uploader(
        "Your photo",
        type=["png", "jpg", "jpeg", "webp"],
        key="cat_person_uploader",
        disabled=st.session_state.cat_is_generating,
    )
    st.caption(
        "Tip: a full-body or waist-up photo works best. "
        "For best results, just one person in frame."
    )

    if cat_person_file is not None:
        new_bytes = cat_person_file.getvalue()
        if new_bytes != st.session_state.cat_person_bytes:
            st.session_state.cat_person_bytes = new_bytes
            # New photo invalidates previous results (different body, different neutralize).
            st.session_state.cat_results = None
            st.session_state.cat_neutralized = None
            st.session_state.cat_neutralize_status = None
            st.session_state.cat_ranker_status = None
            st.session_state.cat_ts = None

    if st.session_state.cat_person_bytes:
        try:
            st.image(
                st.session_state.cat_person_bytes,
                width=240,
                caption="Your photo",
            )
        except Exception:
            st.error("Couldn't read that image. Try a different file.")
            st.session_state.cat_person_bytes = None

    cat_ready = bool(
        st.session_state.cat_person_bytes
        and cat_size
        and cat_color
        and cat_name.strip()
    )
    cat_has_results = st.session_state.cat_results is not None
    cat_button_label = (
        "Finding your top 3…"
        if st.session_state.cat_is_generating
        else ("Re-rank & try on" if cat_has_results else "Find my top 3 & try them on")
    )

    cat_submit_clicked = st.button(
        cat_button_label,
        type="primary",
        disabled=(not cat_ready) or st.session_state.cat_is_generating,
        key="cat_submit_btn",
    )

    if cat_submit_clicked:
        st.session_state.cat_is_generating = True

        # Open the user's photo from the cached bytes (survives form reruns).
        try:
            person_img = Image.open(io.BytesIO(st.session_state.cat_person_bytes))
        except Exception as e:
            st.session_state.cat_is_generating = False
            print(f"[app] couldn't open catalog person upload: {e!r}", flush=True)
            st.error("Couldn't read your photo. Try a different file.")
            st.stop()

        user_attrs = {
            "size": cat_size,
            "color": cat_color,
            "name": cat_name.strip(),
        }
        ranker_status = "ok"

        # Phase 1 — rank the catalog.
        with st.spinner("Finding your top 3… (typically 5–10s)"):
            try:
                picks = rank_products(catalog, user_attrs)
            except RankerError as e:
                print(f"[app] RankerError, using rule fallback: {e}", flush=True)
                fallback_pairs = rule_fallback_top3(catalog, user_attrs)
                picks = [
                    RankingPick(product=p, reason=reason)
                    for p, reason in fallback_pairs
                ]
                ranker_status = "fallback"
            except Exception as e:
                st.session_state.cat_is_generating = False
                print(f"[app] unexpected error in rank_products: {e!r}", flush=True)
                st.error("Something went wrong while finding recommendations. Try again.")
                st.stop()

        if len(picks) < 3:
            st.session_state.cat_is_generating = False
            st.error(
                "Couldn't find enough products to recommend. "
                "Make sure your catalog has at least 3 products."
            )
            st.stop()

        # Phase 2 — neutralize the person photo (Stage 2). Cached by photo hash.
        with st.spinner("Reading your body shape… (typically 15–30s)"):
            try:
                # generate_tryon_batch runs Stage 2 internally, but we need the
                # neutralized image and its status here so we can save and reuse.
                # Calling neutralize_person directly lets us scope the spinner cleanly.
                neutralized, neutralize_status = neutralize_person(person_img)
            except Exception as e:
                st.session_state.cat_is_generating = False
                print(f"[app] unexpected error in neutralize_person (catalog): {e!r}", flush=True)
                st.error("Something went wrong while preparing the body canvas. Try again.")
                st.stop()

        # Phase 3 — parallel try-on for the 3 picks (Stage 3 only — Stage 2 already done).
        with st.spinner("Trying them on… (typically 30–60s)"):
            try:
                results: list[BatchResult] = tryon_batch(neutralized, picks)
            except Exception as e:
                st.session_state.cat_is_generating = False
                print(f"[app] unexpected error in tryon batch: {e!r}", flush=True)
                st.error("Something went wrong while generating the try-ons. Try again.")
                st.stop()

        # Save artifacts: person + neutralized + 3 results + ranker JSON.
        try:
            ts = _timestamp()
            person_path = UPLOAD_DIR / f"{ts}_catalog_person.png"
            _save_png(person_img, person_path)
            if neutralize_status != "fallback":
                neut_path = UPLOAD_DIR / f"{ts}_catalog_person_neutralized.png"
                _save_png(neutralized, neut_path)
            for i, res in enumerate(results, start=1):
                if res.image is not None:
                    out_path = OUTPUT_DIR / f"{ts}_catalog_rank{i}_{res.product.id}.png"
                    _save_png(res.image, out_path)
            ranking_path = OUTPUT_DIR / f"{ts}_catalog_ranking.json"
            ranking_path.write_text(
                json.dumps(
                    {
                        "user_attrs": user_attrs,
                        "ranker_status": ranker_status,
                        "neutralize_status": neutralize_status,
                        "top_3": [
                            {
                                "rank": i,
                                "id": r.product.id,
                                "name": r.product.name,
                                "size": r.product.size,
                                "color": r.product.color,
                                "reason": r.reason,
                                "image_generated": r.image is not None,
                                "error": r.error,
                            }
                            for i, r in enumerate(results, start=1)
                        ],
                    },
                    indent=2,
                )
            )
        except Exception as e:
            print(f"[app] catalog save failed: {e!r}", flush=True)
            # Soft failure — we still display, just lose the audit trail.

        st.session_state.cat_results = results
        st.session_state.cat_neutralized = neutralized
        st.session_state.cat_neutralize_status = neutralize_status
        st.session_state.cat_ranker_status = ranker_status
        st.session_state.cat_ts = ts
        st.session_state.cat_is_generating = False
        st.rerun()

    # -- Per-card regenerate handler (detects which card was clicked last rerun)
    if cat_has_results:
        results: list[BatchResult] = st.session_state.cat_results
        for i in range(len(results)):
            if st.session_state.get(f"cat_regen_btn_{i}_clicked"):
                st.session_state[f"cat_regen_btn_{i}_clicked"] = False
                neutralized = st.session_state.cat_neutralized
                if neutralized is None:
                    st.error("Body-canvas data was lost; please re-rank to refresh.")
                else:
                    with st.spinner(
                        f"Regenerating pick #{i + 1}… (typically 30–60s)"
                    ):
                        pick = RankingPick(
                            product=results[i].product,
                            reason=results[i].reason,
                        )
                        new_result = tryon_single_pick(neutralized, pick)
                    results[i] = new_result
                    # Persist updated list and save the new result to disk.
                    st.session_state.cat_results = results
                    if new_result.image is not None and st.session_state.cat_ts:
                        try:
                            out_path = (
                                OUTPUT_DIR
                                / f"{st.session_state.cat_ts}_catalog_rank{i + 1}_{new_result.product.id}_regen.png"
                            )
                            _save_png(new_result.image, out_path)
                        except Exception as e:
                            print(f"[app] regen save failed: {e!r}", flush=True)
                    st.rerun()

    # -- Render the 3-up results grid
    if cat_has_results:
        results = st.session_state.cat_results
        st.divider()
        if st.session_state.cat_ranker_status == "fallback":
            st.caption(
                "ℹ️ Recommendations were generated using simple matching this time "
                "(the AI ranker was unavailable)."
            )
        if st.session_state.cat_neutralize_status == "fallback":
            st.caption(
                "ℹ️ Body-shape preprocessing was unavailable for your photo; "
                "result quality may be reduced."
            )

        result_cols = st.columns(3)
        for i, (col, res) in enumerate(zip(result_cols, results)):
            with col:
                st.markdown(f"#### Pick #{i + 1} — {res.product.name}")
                if res.image is not None:
                    st.image(res.image, width="stretch")
                else:
                    # Show the raw product image as a stand-in, with error caption.
                    if res.product.raw_path.exists():
                        st.image(str(res.product.raw_path), width="stretch")
                    st.error(
                        res.error
                        or "Try-on couldn't be generated for this product."
                    )

                # Product thumbnail (small) + chips
                thumb_col, info_col = st.columns([1, 2])
                with thumb_col:
                    if res.product.raw_path.exists():
                        st.image(str(res.product.raw_path), width=80)
                with info_col:
                    st.markdown(
                        f"**Size**: {res.product.size} &nbsp;&nbsp; "
                        f"**Color**: {res.product.color}"
                    )
                st.markdown(f"**Why we picked this:** {res.reason}")

                # Per-card regenerate button — sets a flag the handler picks up next rerun.
                if st.button(
                    "Regenerate this one",
                    key=f"cat_regen_btn_{i}",
                    disabled=st.session_state.cat_is_generating,
                ):
                    st.session_state[f"cat_regen_btn_{i}_clicked"] = True
                    st.rerun()
    else:
        st.caption("Fill in the form above and submit to see your top 3 try-ons.")
