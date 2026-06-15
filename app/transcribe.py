from groq import Groq

from app.config import GROQ_API_KEY, WHISPER_MODEL

# Groq enforces a 25 MB upload limit on the free transcription tier.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def transcribe_audio(content: bytes, filename: str = "recording.wav") -> str:
    """Transcribe spoken audio to text using Groq's Whisper model."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

    if not content:
        raise ValueError("No audio data received.")

    if len(content) > MAX_AUDIO_BYTES:
        raise ValueError("Audio is too large. Please record a shorter clip (under 25 MB).")

    client = Groq(api_key=GROQ_API_KEY)

    transcription = client.audio.transcriptions.create(
        file=(filename, content),
        model=WHISPER_MODEL,
        response_format="text",
    )

    # response_format="text" returns the transcript directly as a string.
    text = transcription if isinstance(transcription, str) else getattr(transcription, "text", "")
    return (text or "").strip()
