from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.models import Source, UserProfile

SYSTEM_PROMPT = """You are a helpful natural face and hair care assistant.

Rules:
- Provide educational information only. Do not diagnose conditions or prescribe medical treatment.
- Base your answers primarily on the provided context.
- When context comes from uploaded documents, prioritize that information.
- When context comes from web search, use it carefully and mention that it is general web information.
- When the context does not contain enough information, say so honestly and give only general safe guidance.
- Personalize advice when a user profile is provided (skin type, hair type, allergies, age, climate, routine).
- If the user has allergies, always warn about ingredients that may not suit them.
- Recommend natural products and routines when appropriate.
- Explain ingredients clearly when asked.
- Keep answers clear, practical, and friendly.
- End with a brief reminder that this is educational information, not medical advice."""


def _format_profile(profile: UserProfile | None) -> str:
    if not profile:
        return "No user profile provided."

    fields = {
        "Skin type": profile.skin_type,
        "Hair type": profile.hair_type,
        "Allergies": profile.allergies,
        "Age": profile.age,
        "Climate": profile.climate,
        "Current routine": profile.current_routine,
    }
    lines = [f"- {label}: {value}" for label, value in fields.items() if value]
    return "\n".join(lines) if lines else "No user profile provided."


def _format_context(sources: list[Source], used_web_fallback: bool) -> str:
    if not sources:
        return "No context available."

    if used_web_fallback:
        header = (
            "No relevant information was found in uploaded documents. "
            "The following web search results were used instead:"
        )
    else:
        header = "Context from uploaded documents:"

    blocks = [header]
    for i, source in enumerate(sources, start=1):
        if source.source_type == "web":
            url_part = f" ({source.url})" if source.url else ""
            blocks.append(
                f"[Web source {i}: {source.title or source.filename}{url_part}]\n"
                f"{source.excerpt}"
            )
        else:
            blocks.append(
                f"[Document {i}: {source.filename}, page {source.page}]\n"
                f"{source.excerpt}"
            )
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    sources: list[Source],
    profile: UserProfile | None = None,
    used_web_fallback: bool = False,
) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

    client = Groq(api_key=GROQ_API_KEY)

    user_message = f"""User profile:
{_format_profile(profile)}

{_format_context(sources, used_web_fallback)}

User question:
{question}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.4,
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""
