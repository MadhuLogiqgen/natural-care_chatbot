import base64
import io

from groq import Groq

from app.config import GROQ_API_KEY, VISION_MODEL
from app.models import UserProfile

# Groq accepts base64 images up to ~4 MB; we resize below this to stay safe.
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # reject anything absurdly large up front
MAX_DIMENSION = 1024  # longest edge after downscaling

VISION_SYSTEM_PROMPT = """You are a natural face and hair care assistant analyzing a user's photo.

Strict rules:
- This is for cosmetic, educational skincare/haircare guidance ONLY. You are NOT a doctor.
- Do NOT diagnose medical or dermatological conditions, name diseases, or suggest treatments.
- Do NOT identify the person, guess their identity, age precisely, ethnicity, or make judgments about appearance.
- Only describe visible cosmetic characteristics relevant to natural care, such as:
  skin: apparent oiliness/dryness, visible shine, texture, dryness/flaking, redness, dullness, visible blemishes.
  hair (if visible): apparent dryness, frizz, oiliness, curl pattern, breakage, scalp visibility.
- If the photo does not clearly show a face, skin, or hair, say so plainly and ask for a clearer photo.
- Be concise and factual. Output short bullet points of observations only — no product advice here.
- If anything looks like it may need a medical professional, gently suggest seeing one, without diagnosing."""


def _prepare_image(content: bytes) -> str:
    """Validate, downscale, and base64-encode an image as a JPEG data URL."""
    if not content:
        raise ValueError("No image data received.")
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large. Please upload a photo under 10 MB.")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ValueError(
            "Image support is not installed. Run: pip install pillow"
        ) from exc

    try:
        image = Image.open(io.BytesIO(content))
        image = image.convert("RGB")
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        jpeg_bytes = buffer.getvalue()
    except Exception as exc:
        raise ValueError(
            "Could not read that image. Please upload a valid JPG, PNG, or WEBP photo."
        ) from exc

    encoded = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def analyze_skin_hair_photo(
    content: bytes,
    note: str | None = None,
    profile: UserProfile | None = None,
) -> str:
    """Analyze a face/hair photo and return cosmetic observations as text.

    The observations are later fed into the RAG pipeline to generate grounded
    natural-product recommendations.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

    data_url = _prepare_image(content)

    prompt = (
        "Look at this photo and list the visible cosmetic skin/hair "
        "characteristics relevant to natural face and hair care. "
        "Respond with short bullet points of observations only."
    )
    if profile and (profile.skin_type or profile.hair_type):
        prompt += (
            f"\n\nFor reference, the user describes their skin as "
            f"'{profile.skin_type or 'unspecified'}' and hair as "
            f"'{profile.hair_type or 'unspecified'}'."
        )
    if note:
        prompt += f"\n\nThe user specifically asks: {note}"

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.3,
        max_tokens=600,
    )

    return (response.choices[0].message.content or "").strip()
