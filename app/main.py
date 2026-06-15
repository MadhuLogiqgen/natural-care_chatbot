import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth import create_access_token, hash_password, verify_password
from app.db.database import (
    add_message,
    create_conversation,
    create_user,
    delete_conversation,
    ensure_conversation,
    get_conversation,
    get_user_by_email,
    init_db,
    list_conversations,
    update_conversation_title_if_first_message,
)
from app.deps import get_current_user
from app.models import (
    AskRequest,
    AskResponse,
    AuthResponse,
    ConversationDetail,
    ConversationSummary,
    CreateConversationRequest,
    CreateConversationResponse,
    HealthResponse,
    IngestResponse,
    LoginRequest,
    RegisterRequest,
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Natural Care RAG API",
    description="Face and hair care Q&A powered by PDF knowledge and Groq Llama",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    documents = list_document_names()
    return HealthResponse(
        status="ok",
        pdf_count=len(documents),
        indexed_chunks=get_indexed_chunk_count(),
        documents=documents,
    )


@app.post("/auth/register", response_model=AuthResponse)
def register(request: RegisterRequest):
    try:
        user = create_user(request.email, hash_password(request.password))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = create_access_token(user["id"], user["email"])
    return AuthResponse(
        access_token=token,
        user=UserPublic(
            id=user["id"],
            email=user["email"],
            created_at=user["created_at"],
        ),
    )


@app.post("/auth/login", response_model=AuthResponse)
def login(request: LoginRequest):
    user = get_user_by_email(request.email)
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(user["id"], user["email"])
    return AuthResponse(
        access_token=token,
        user=UserPublic(
            id=user["id"],
            email=user["email"],
            created_at=user["created_at"],
        ),
    )


@app.get("/auth/me", response_model=UserPublic)
def me(current_user: dict = Depends(get_current_user)):
    return UserPublic(
        id=current_user["id"],
        email=current_user["email"],
        created_at=current_user["created_at"],
    )


@app.get("/conversations", response_model=list[ConversationSummary])
def get_conversations(current_user: dict = Depends(get_current_user)):
    return list_conversations(current_user["id"])


@app.post("/conversations", response_model=CreateConversationResponse)
def create_new_conversation(
    request: CreateConversationRequest,
    current_user: dict = Depends(get_current_user),
):
    conversation = create_conversation(
        user_id=current_user["id"],
        title=request.title,
        profile=request.profile,
    )
    return CreateConversationResponse(conversation=conversation)


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation_detail(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        return get_conversation(conversation_id, current_user["id"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc


@app.delete("/conversations/{conversation_id}")
def remove_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        delete_conversation(conversation_id, current_user["id"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    return {"message": "Conversation deleted."}


@app.post("/upload", response_model=IngestResponse)
async def upload_pdfs(
    files: list[UploadFile] = File(...),
    _: dict = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    total_chunks = 0
    saved_names: list[str] = []

    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="A file has no name.")

        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"'{upload.filename}' is not a PDF. Only .pdf files are supported.",
            )

        content = await upload.read()
        try:
            saved_name, chunks = ingest_upload(upload.filename, content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process '{upload.filename}': {exc}",
            ) from exc

        saved_names.append(saved_name)
        total_chunks += chunks

    documents = list_document_names()
    return IngestResponse(
        files_processed=len(saved_names),
        chunks_indexed=total_chunks,
        message="PDFs uploaded and indexed successfully.",
        documents=documents,
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    _: dict = Depends(get_current_user),
):
    content = await file.read()
    try:
        text = transcribe_audio(content, file.filename or "recording.wav")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to transcribe audio: {exc}",
        ) from exc

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Could not understand the audio. Please try recording again.",
        )

    return TranscriptionResponse(text=text)


@app.post("/ingest", response_model=IngestResponse)
def ingest(_: dict = Depends(get_current_user)):
    pdf_files = list_pdfs()
    if not pdf_files:
        raise HTTPException(
            status_code=400,
            detail="No documents yet. Upload PDFs from the chat interface.",
        )

    files_processed, chunks_indexed = ingest_pdfs()
    return IngestResponse(
        files_processed=files_processed,
        chunks_indexed=chunks_indexed,
        message="All documents re-indexed successfully.",
        documents=list_document_names(),
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    current_user: dict = Depends(get_current_user),
):
    sources, used_web_fallback = retrieve_context(
        request.question, force_web=request.web_search
    )

    if not sources:
        raise HTTPException(
            status_code=404,
            detail=(
                "No relevant documents found and web search returned no results. "
                "Try rephrasing your question or upload more PDFs."
            ),
        )

    try:
        conversation_id = ensure_conversation(
            current_user["id"],
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
            conversation_id,
            current_user["id"],
            request.question,
        )
        add_message(conversation_id, current_user["id"], "user", request.question)
        add_message(
            conversation_id,
            current_user["id"],
            "assistant",
            answer,
            sources=sources,
            used_web_fallback=used_web_fallback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to generate answer: {exc}",
        ) from exc

    return AskResponse(
        answer=answer,
        sources=sources,
        used_web_fallback=used_web_fallback,
        conversation_id=conversation_id,
    )


@app.post("/analyze-photo", response_model=AskResponse)
async def analyze_photo(
    file: UploadFile = File(...),
    note: str | None = Form(None),
    profile: str | None = Form(None),
    conversation_id: str | None = Form(None),
    current_user: dict = Depends(get_current_user),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file (JPG, PNG, or WEBP).",
        )

    content = await file.read()

    profile_obj: UserProfile | None = None
    if profile:
        try:
            profile_obj = UserProfile(**json.loads(profile))
        except Exception:
            profile_obj = None

    # 1) Vision: describe the visible cosmetic characteristics.
    try:
        observations = analyze_skin_hair_photo(content, note=note, profile=profile_obj)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to analyze the photo: {exc}",
        ) from exc

    # 2) Build a query from the observations and run it through the RAG pipeline
    #    so recommendations are grounded in the user's documents (or web).
    request = (note or "").strip() or (
        "Based on my photo, suggest the best natural products and a simple "
        "routine for my face and hair."
    )
    query = (
        f"{request}\n\nHere is an analysis of the user's uploaded photo:\n"
        f"{observations}\n\nStart by briefly summarizing what was observed, "
        f"then recommend specific natural products, ingredients, and a simple routine."
    )

    sources, used_web_fallback = retrieve_context(query, force_web=False)

    try:
        conversation_id = ensure_conversation(
            current_user["id"],
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
            conversation_id,
            current_user["id"],
            "Photo analysis",
        )
        add_message(
            conversation_id,
            current_user["id"],
            "user",
            note.strip() if note and note.strip() else "[Uploaded a photo for analysis]",
        )
        add_message(
            conversation_id,
            current_user["id"],
            "assistant",
            answer,
            sources=sources,
            used_web_fallback=used_web_fallback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to generate recommendations: {exc}",
        ) from exc

    return AskResponse(
        answer=answer,
        sources=sources,
        used_web_fallback=used_web_fallback,
        conversation_id=conversation_id,
    )
