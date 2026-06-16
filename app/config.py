import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", encoding="utf-8")


def _setting(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "chatbot.db"

GROQ_API_KEY = _setting("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL = _setting("WHISPER_MODEL", "whisper-large-v3-turbo")
# Multimodal model used for photo analysis. Must be a Groq vision model.
VISION_MODEL = _setting(
    "VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "natural_care"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
SIMILARITY_MAX_DISTANCE = float(_setting("SIMILARITY_MAX_DISTANCE", "0.65"))
WEB_SEARCH_MAX_RESULTS = int(_setting("WEB_SEARCH_MAX_RESULTS", "4"))

API_HOST = _setting("API_HOST", "127.0.0.1")
API_PORT = int(_setting("API_PORT", "8000"))
API_URL = _setting("API_URL") or f"http://{API_HOST}:{API_PORT}"

JWT_SECRET_KEY = _setting("JWT_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(_setting("JWT_EXPIRE_MINUTES", str(60 * 24 * 7)))
