EXTRACT_GARMENT_INSTRUCTION = """
You are given a single image showing a garment (clothing item). The garment may be displayed in any of these ways:
- Laid flat on a surface or hung on a hanger (flat-lay or product shot).
- On a mannequin, dress form, or tailor's bust.
- Worn by a person or model.
- Photographed in a styled scene or against any background.

Render ONE photorealistic image of ONLY THE GARMENT itself as a clean product flat-lay on a PURE WHITE BACKGROUND.

Strict requirements:
- Show ONLY the clothing item. Remove any person, body parts (hands, arms, face, neck, torso, legs, feet), mannequin, dress form, hanger, headpiece, jewelry, eyewear, accessories, props, background scenery, and surrounding objects.
- Preserve the garment's exact colors, patterns, prints, embroidery, beadwork, fabric texture, and ALL design details from the input image.
- Preserve the garment's cut, length, neckline, sleeves, silhouette, and construction as they appear in the input.
- Lay the garment out neatly so its full shape and details are clearly visible — as if photographed for an e-commerce product page.
- Use natural soft studio lighting on a PURE WHITE background. Minimal shadow.
- If the input shows multiple pieces of one outfit (e.g., kurta + dupatta + pants), arrange them together in the output so the complete outfit is visible.
- Do NOT add accessories, jewelry, footwear, or design elements that are not present in the input image.
- Do NOT alter the garment's colors, modify the pattern, or stylize the appearance.

Output: a single high-detail, photorealistic product flat-lay image of the garment only.
""".strip()


NEUTRALIZE_PERSON_INSTRUCTION = """
You are given a single image showing a person wearing clothing.

Generate ONE photorealistic image of the SAME PERSON now wearing PLAIN FORM-FITTING NEUTRAL CLOTHING — a plain gray T-shirt and plain gray pants/leggings that closely follow the body's actual contours — standing in a NEUTRAL POSE facing the camera, with arms relaxed at sides and weight balanced evenly.

This is a body-canvas rendering. Its purpose is to commit to the person's ACTUAL body shape, free from the influence of their original outfit.

Take ONLY these things from the input image:
- The person's face, facial features, identity, ethnicity, age, and expression.
- The person's hairstyle and hair color.
- The person's skin tone.
- CRITICAL — the person's ACTUAL body shape, body proportions, height, build, weight, and body type. The actual body must be inferred from beneath the original clothing. Use cues like the visible shoulder line, neck-to-shoulder width, the way fabric drapes, ankle/wrist exposure, and other anatomical signals to estimate the underlying proportions.
- The background, setting, lighting, and camera angle.
- The framing and crop of the photo (full-body if the input is full-body; waist-up if waist-up).

Do NOT take any of these from the input image:
- The original clothing's specific silhouette, drape, or apparent fit. Loose clothing makes the silhouette appear wider than the body; tight clothing may make it appear narrower. The body underneath is what we want.
- The original pose. Replace it with a neutral standing pose — arms relaxed at sides, feet roughly shoulder-width apart, facing the camera, weight evenly distributed.
- The original clothing items themselves. Replace ALL of them with plain gray form-fitting clothing (gray T-shirt + gray pants/leggings).
- Any accessories, jewelry, scarves, belts, or styling worn over the original outfit (unless these are essential to identity, e.g., glasses).

CRITICAL — body authenticity:
- The person's actual height, weight, build, and proportions must remain TRUE to their real body.
- Do NOT idealize, slim, elongate, "fashion-model," or otherwise alter the body. If the person has a fuller frame, render that fuller frame honestly. If they are shorter or taller than fashion-photography norms, preserve those actual proportions.
- The form-fitting gray clothing must reveal the body shape honestly — neither hiding fuller proportions nor exaggerating slimness.
- The face must remain unmistakably the same person.

Identity and authenticity:
- Do NOT alter the person's identity, ethnicity, age, gender presentation, body type, or facial features in any way.

Output: a single photorealistic image showing the person in plain gray form-fitting clothing, standing in a neutral pose, in the same setting as the input image.
""".strip()


RANK_PRODUCTS_INSTRUCTION = """
You are a fashion-shopping assistant. The user is browsing a small store's catalog and has told you what they're looking for. Your job is to pick the THREE products from the catalog that best match the user's request.

You will receive:
- A USER REQUEST as a JSON object with fields: size, color, name (a free-text query describing what the user wants).
- A CATALOG of products. For each product you receive both its JSON entry (with id, name, size, color) AND a photograph of the product, in this order: product-1 JSON, product-1 image, product-2 JSON, product-2 image, ... product-10 JSON, product-10 image.

Use BOTH the JSON metadata AND the product images to judge each candidate. The image is the ground truth for actual color shade, style, cut, and visual mood; the JSON is the ground truth for size and the canonical color label.

Ranking criteria, in STRICT priority order. Higher-priority criteria DOMINATE — never sacrifice a higher-priority match to gain a lower-priority one.

1. KIND OF OUTFIT / STYLE — THE DOMINANT SIGNAL. Compare the user's free-text "name" query (e.g., "kurta", "floral dress", "jumpsuit", "saree", "something flowy for summer") against each product's name AND its image. Strongly prefer products whose GARMENT TYPE matches the user's request. If the user asks for a kurta, recommend kurtas — do NOT recommend a dress, jumpsuit, or saree just because its size or color matches better. Use the product image to verify the visual garment type beyond just the catalog name. Garment-type match outweighs every other factor in this ranking.

2. Size match — secondary signal, used to rank AMONG style-matching candidates. Among the products that match the user's requested kind of outfit, prefer those whose size exactly matches the user's stated size. If fewer than three products match BOTH the style AND the size, prefer style-matching products of any size over off-style products of the right size. Call out any size difference in the reason field.

3. Color match — tertiary signal, the final tie-breaker. Prefer the user's stated color, then visually similar colors (e.g., "blush" is close to "pink"; "navy" is close to "dark blue"). The image can disambiguate when names are vague. Color is only used to choose among otherwise-equivalent style+size matches.

CRITICAL — DO NOT VIOLATE STYLE PRIORITY:
- A perfect size match on the wrong kind of outfit is a WORSE pick than a wrong-size right-kind-of-outfit.
- A perfect color match on the wrong kind of outfit is a WORSE pick than a different-color right-kind-of-outfit.
- Only fall back to off-style products if there are genuinely fewer than three products of the requested kind in the catalog.

For each of the three picks, write a SHORT (one sentence, ~15-25 words) customer-facing reason that LEADS WITH the garment-type match and then mentions the size and color. Examples:
- "Matches your kurta request in size XL with a bold green-and-red color contrast."
- "Same kurta style as your query and your size XL; color is Blue rather than the Black you asked for."
- "Closest kurta in the catalog — size XXL (you asked for XL), Off-White color; floral print fits a 'floral' query."
- "Off-style fallback — no more kurtas in the catalog so this floral dress is the closest match for your size and color."

Rules for the reason field:
- Be honest. If size or color doesn't exactly match, say so.
- Don't invent fields that aren't in the JSON (no brand, no fabric, no price).
- Speak directly to the user ("your size M") rather than in third person.

Output ONLY a JSON object matching this shape (no commentary, no Markdown):
{
  "top_3": [
    {"id": "<product id>", "reason": "<one-sentence reason>"},
    {"id": "<product id>", "reason": "<one-sentence reason>"},
    {"id": "<product id>", "reason": "<one-sentence reason>"}
  ]
}

The list MUST contain exactly three items. Each id MUST exactly match an id from the catalog. Order the list from best match (index 0) to third-best (index 2).
""".strip()


TRY_ON_INSTRUCTION = """
You are given two reference images.

FIRST IMAGE — the SUBJECT photo. A photograph of the person whose try-on we are generating. THE SUBJECT'S BODY IS THE CANVAS — every other element adapts to it. The subject may be shown in form-fitting neutral clothing (a clear view of the body) OR in their original everyday clothing; in either case, the body underneath is the canvas.

SECOND IMAGE — the OUTFIT photo. Shows the garment to put on the subject as a clean product image (typically a flat-lay on a white background).

Generate ONE photorealistic image of the SAME PERSON FROM THE FIRST IMAGE wearing the outfit shown in the SECOND IMAGE.

Take ONLY these things from the FIRST IMAGE (the subject — the canvas):
- The person's face, facial features, identity, ethnicity, age, and expression.
- The person's hairstyle and hair color.
- The person's skin tone.
- CRITICAL — the person's actual body shape, body proportions, height, build, weight, silhouette, and body type. Read the body from the FIRST IMAGE. If their clothing partially hides the body, infer the underlying proportions from drape, shoulder line, and other visible cues. Do NOT slim, elongate, lengthen, idealize, or reshape the subject's body.
- The overall camera framing, facing direction, and general standing posture as the STARTING POINT (specific limb positions will adapt — see pose handling below).
- The background, setting, lighting, and camera angle.

Take ONLY these things from the SECOND IMAGE (the garment):
- The garment type (e.g., kurta, suit, dress, gown, saree, salwar, dupatta) and identifying features.
- Colors, patterns, textures, fabric, embroidery, prints, and design details.
- Construction style (e.g., A-line, fitted, oversized, flared, straight-cut).
- Length category (e.g., knee-length, ankle-length, full-length, cropped).
- Neckline style.
- Sleeve style and length.

Do NOT take any of these from the SECOND IMAGE, even if visible:
- The reference body's shape, proportions, or build (if a mannequin or model is present in the second image).
- The reference body's pose, hand position, arm position, or stance.
- The reference person's face, hair, or identity.
- Any headpiece, crown, tiara, veil, jewelry, eyewear, gloves, or other accessories worn by the mannequin or second person — these are NOT part of the outfit.
- The second image's background, lighting, or setting.

POSE HANDLING — adapt freely to the new outfit:
- Use the FIRST IMAGE's overall camera framing, facing direction, and general standing posture as the starting point.
- Limb and hand positions adapt FREELY to suit the new garment. Choose a pose that's natural for the type of outfit being worn:
  - For a saree: hands may hold or rest on the pallu (the drape over the shoulder), or arms may be elegantly at sides.
  - For a kurta/salwar: hands rest naturally at sides, on hips, or one arm gently extended; do NOT hide hands awkwardly in the fabric.
  - For a gown: hands may rest at sides, lightly hold the hem or skirt, or one arm gently extended.
  - For a tailored suit or jacket: hands may rest at sides, in pockets if pockets exist, or one casually positioned.
  - For a casual top + pants: hands rest naturally at sides, on hips, or in pockets if present.
- Do NOT force the original photo's exact limb positions when they conflict with the new garment (e.g., do NOT keep a hands-in-pockets pose for a garment with no pockets).
- Do NOT copy any pose from the SECOND IMAGE.

CRITICAL — body-fit adaptation principle:
- The garment FITS the SUBJECT'S BODY AS IT IS. It does NOT reshape the SUBJECT.
- The drape, fit, and silhouette of the result are determined by the SUBJECT'S body — NOT by how the garment looks on any mannequin or reference person.
- The SUBJECT'S height, weight, build, proportions, and figure must remain IDENTICAL to the body shown in the FIRST IMAGE. Body authenticity over fashion-photography polish.

Garment fit:
- The clothing must fit naturally on the SUBJECT'S body, with realistic folds, drape, and shadows matching the lighting and direction of the FIRST IMAGE.
- Fabric drape follows the SUBJECT'S actual body contours.

Framing:
- Match the FIRST IMAGE's crop and overall framing. If the FIRST IMAGE is full-body, the output is full-body with feet visible. If waist-up, the output is waist-up. Do not zoom in tighter.

Identity and body authenticity:
- Do NOT alter the SUBJECT'S identity, ethnicity, age, gender presentation, body type, or facial features in any way.
- Do NOT idealize, flatter, slim, elongate, or otherwise "fashion-model" the SUBJECT'S figure. The output should look like THIS specific person wearing this garment, with a pose that suits the garment naturally.

Output: a single high-detail, photorealistic image. Natural lighting matching the FIRST IMAGE. Authentic to the SUBJECT'S actual body.
""".strip()

