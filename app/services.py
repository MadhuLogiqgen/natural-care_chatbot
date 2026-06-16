"""Shared business logic used by FastAPI and the Streamlit app."""

from __future__ import annotations

import json
from typing import Any

from app.auth import create_access_token, decode_access_token, hash_password, verify_password
from app.db.database import (
    add_message,
    create_user,
    delete_conversation,
    ensure_conversation,
    get_conversation,
    get_user_by_email,
    get_user_by_id,
    init_db,
    list_conversations,
    update_conversation_title_if_first_message,
)
from app.models import (
    AskRequest,
    AskResponse,
    AuthResponse,
    HealthResponse,
    IngestResponse,
    TranscriptionResponse,
    UserProfile,
    UserPublic,
)
from app.rag.generate import generate_answer
from app.rag.ingest import (
    get_indexed_chunk_count,
    ingest_pdfs,
    ingest_upload,
    list_document_names,
    list_pdfs,
)
from app.rag.retrieve import retrieve_context
from app.transcribe import transcribe_audio
from app.vision import analyze_skin_hair_photo

_initialized = False


class AppError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        init_db()
        _initialized = True


def require_user(access_token: str | None) -> dict[str, Any]:
    if not access_token:
        raise AppError(401, "Not authenticated. Please log in.")
    try:
        payload = decode_access_token(access_token)
        user_id = payload.get("sub")
        if not user_id:
            raise ValueError("Missing subject.")
    except ValueError as exc:
        raise AppError(401, str(exc)) from exc

    user = get_user_by_id(user_id)
    if not user:
        raise AppError(401, "User not found.")
    return user


def register(email: str, password: str) -> dict[str, Any]:
    ensure_initialized()
    try:
        user = create_user(email, hash_password(password))
    except ValueError as exc:
        raise AppError(400, str(exc)) from exc

    token = create_access_token(user["id"], user["email"])
    return AuthResponse(
        access_token=token,
        user=UserPublic(
            id=user["id"],
            email=user["email"],
            created_at=user["created_at"],
        ),
    ).model_dump()


def login(email: str, password: str) -> dict[str, Any]:
    ensure_initialized()
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        raise AppError(401, "Invalid email or password.")

    token = create_access_token(user["id"], user["email"])
    return AuthResponse(
        access_token=token,
        user=UserPublic(
            id=user["id"],
            email=user["email"],
            created_at=user["created_at"],
        ),
    ).model_dump()


def health_check() -> dict[str, Any]:
    ensure_initialized()
    documents = list_document_names()
    return HealthResponse(
        status="ok",
        pdf_count=len(documents),
        indexed_chunks=get_indexed_chunk_count(),
        documents=documents,
    ).model_dump()


def get_conversations(user: dict[str, Any]) -> list[dict[str, Any]]:
    return list_conversations(user["id"])


def get_conversation_detail(user: dict[str, Any], conversation_id: str) -> dict[str, Any]:
    try:
        return get_conversation(conversation_id, user["id"])
    except KeyError as exc:
        raise AppError(404, "Conversation not found.") from exc


def remove_conversation(user: dict[str, Any], conversation_id: str) -> dict[str, str]:
    try:
        delete_conversation(conversation_id, user["id"])
    except KeyError as exc:
        raise AppError(404, "Conversation not found.") from exc
    return {"message": "Conversation deleted."}


def upload_pdfs(
    user: dict[str, Any],
    files: list[tuple[str, bytes]],
) -> dict[str, Any]:
    del user  # auth only

    if not files:
        raise AppError(400, "No files provided.")

    total_chunks = 0
    saved_names: list[str] = []

    for filename, content in files:
        if not filename:
            raise AppError(400, "A file has no name.")

        if not filename.lower().endswith(".pdf"):
            raise AppError(
                400,
                f"'{filename}' is not a PDF. Only .pdf files are supported.",
            )

        try:
            saved_name, chunks = ingest_upload(filename, content)
        except ValueError as exc:
            raise AppError(400, str(exc)) from exc
        except Exception as exc:
            raise AppError(
                500,
                f"Failed to process '{filename}': {exc}",
            ) from exc

        saved_names.append(saved_name)
        total_chunks += chunks

    documents = list_document_names()
    return IngestResponse(
        files_processed=len(saved_names),
        chunks_indexed=total_chunks,
        message="PDFs uploaded and indexed successfully.",
        documents=documents,
    ).model_dump()


def reingest_documents(user: dict[str, Any]) -> dict[str, Any]:
    del user
    pdf_files = list_pdfs()
    if not pdf_files:
        raise AppError(
            400,
            "No documents yet. Upload PDFs from the chat interface.",
        )

    files_processed, chunks_indexed = ingest_pdfs()
    return IngestResponse(
        files_processed=files_processed,
        chunks_indexed=chunks_indexed,
        message="All documents re-indexed successfully.",
        documents=list_document_names(),
    ).model_dump()


def ask(
    user: dict[str, Any],
    question: str,
    profile: dict | None = None,
    conversation_id: str | None = None,
    web_search: bool = False,
) -> dict[str, Any]:
    request = AskRequest(
        question=question,
        profile=UserProfile(**profile) if profile else None,
        conversation_id=conversation_id,
        web_search=web_search,
    )

    sources, used_web_fallback = retrieve_context(
        request.question, force_web=request.web_search
    )

    if not sources:
        raise AppError(
            404,
            "No relevant documents found and web search returned no results. "
            "Try rephrasing your question or upload more PDFs.",
        )

    try:
        conv_id = ensure_conversation(
            user["id"],
            request.conversation_id,
            request.question,
            request.profile,
        )
        answer = generate_answer(
            request.question,
            sources,
            request.profile,
            used_web_fallback=used_web_fallback,
        )
        update_conversation_title_if_first_message(
            conv_id,
            user["id"],
            request.question,
        )
        add_message(conv_id, user["id"], "user", request.question)
        add_message(
            conv_id,
            user["id"],
            "assistant",
            answer,
            sources=sources,
            used_web_fallback=used_web_fallback,
        )
    except ValueError as exc:
        raise AppError(500, str(exc)) from exc
    except Exception as exc:
        raise AppError(502, f"Failed to generate answer: {exc}") from exc

    return AskResponse(
        answer=answer,
        sources=sources,
        used_web_fallback=used_web_fallback,
        conversation_id=conv_id,
    ).model_dump()


def analyze_photo(
    user: dict[str, Any],
    content: bytes,
    filename: str,
    mime: str | None,
    note: str | None = None,
    profile: dict | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:

    if mime and not mime.startswith("image/"):
        raise AppError(
            400,
            "Please upload an image file (JPG, PNG, or WEBP).",
        )

    profile_obj: UserProfile | None = UserProfile(**profile) if profile else None

    try:
        observations = analyze_skin_hair_photo(
            content, note=note, profile=profile_obj
        )
    except ValueError as exc:
        raise AppError(400, str(exc)) from exc
    except Exception as exc:
        raise AppError(502, f"Failed to analyze the photo: {exc}") from exc

    request_text = (note or "").strip() or (
        "Based on my photo, suggest the best natural products and a simple "
        "routine for my face and hair."
    )
    query = (
        f"{request_text}\n\nHere is an analysis of the user's uploaded photo:\n"
        f"{observations}\n\nStart by briefly summarizing what was observed, "
        f"then recommend specific natural products, ingredients, and a simple routine."
    )

    sources, used_web_fallback = retrieve_context(query, force_web=False)

    try:
        conv_id = ensure_conversation(
            user["id"],
            conversation_id,
            "Photo analysis",
            profile_obj,
        )
        answer = generate_answer(
            query,
            sources,
            profile_obj,
            used_web_fallback=used_web_fallback,
        )
        update_conversation_title_if_first_message(
            conv_id,
            user["id"],
            "Photo analysis",
        )
        add_message(
            conv_id,
            user["id"],
            "user",
            note.strip() if note and note.strip() else "[Uploaded a photo for analysis]",
        )
        add_message(
            conv_id,
            user["id"],
            "assistant",
            answer,
            sources=sources,
            used_web_fallback=used_web_fallback,
        )
    except ValueError as exc:
        raise AppError(500, str(exc)) from exc
    except Exception as exc:
        raise AppError(502, f"Failed to generate recommendations: {exc}") from exc

    return AskResponse(
        answer=answer,
        sources=sources,
        used_web_fallback=used_web_fallback,
        conversation_id=conv_id,
    ).model_dump()


def transcribe(user: dict[str, Any], content: bytes, filename: str) -> str:
    del user
    try:
        text = transcribe_audio(content, filename or "recording.wav")
    except ValueError as exc:
        raise AppError(400, str(exc)) from exc
    except Exception as exc:
        raise AppError(502, f"Failed to transcribe audio: {exc}") from exc

    if not text:
        raise AppError(
            422,
            "Could not understand the audio. Please try recording again.",
        )

    return TranscriptionResponse(text=text).text
