import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import create_conversation
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
from app.services import (
    AppError,
    analyze_photo as analyze_photo_service,
    ask as ask_service,
    ensure_initialized,
    get_conversation_detail,
    get_conversations,
    health_check,
    login,
    register,
    reingest_documents,
    remove_conversation,
    transcribe as transcribe_service,
    upload_pdfs as upload_pdfs_service,
)


def _http_error(exc: AppError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_initialized()
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
    return health_check()


@app.post("/auth/register", response_model=AuthResponse)
def register_user(request: RegisterRequest):
    try:
        return register(request.email, request.password)
    except AppError as exc:
        raise _http_error(exc) from exc


@app.post("/auth/login", response_model=AuthResponse)
def login_user(request: LoginRequest):
    try:
        return login(request.email, request.password)
    except AppError as exc:
        raise _http_error(exc) from exc


@app.get("/auth/me", response_model=UserPublic)
def me(current_user: dict = Depends(get_current_user)):
    return UserPublic(
        id=current_user["id"],
        email=current_user["email"],
        created_at=current_user["created_at"],
    )


@app.get("/conversations", response_model=list[ConversationSummary])
def list_user_conversations(current_user: dict = Depends(get_current_user)):
    return get_conversations(current_user)


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
def conversation_detail(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        return get_conversation_detail(current_user, conversation_id)
    except AppError as exc:
        raise _http_error(exc) from exc


@app.delete("/conversations/{conversation_id}")
def delete_user_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        return remove_conversation(current_user, conversation_id)
    except AppError as exc:
        raise _http_error(exc) from exc


@app.post("/upload", response_model=IngestResponse)
async def upload_pdfs(
    files: list[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="A file has no name.")
        payloads.append((upload.filename, await upload.read()))

    try:
        return upload_pdfs_service(current_user, payloads)
    except AppError as exc:
        raise _http_error(exc) from exc


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    content = await file.read()
    try:
        text = transcribe_service(
            current_user, content, file.filename or "recording.wav"
        )
    except AppError as exc:
        raise _http_error(exc) from exc
    return TranscriptionResponse(text=text)


@app.post("/ingest", response_model=IngestResponse)
def ingest(current_user: dict = Depends(get_current_user)):
    try:
        return reingest_documents(current_user)
    except AppError as exc:
        raise _http_error(exc) from exc


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        return ask_service(
            current_user,
            request.question,
            profile=request.profile.model_dump() if request.profile else None,
            conversation_id=request.conversation_id,
            web_search=request.web_search,
        )
    except AppError as exc:
        raise _http_error(exc) from exc


@app.post("/analyze-photo", response_model=AskResponse)
async def analyze_photo(
    file: UploadFile = File(...),
    note: str | None = Form(None),
    profile: str | None = Form(None),
    conversation_id: str | None = Form(None),
    current_user: dict = Depends(get_current_user),
):
    content = await file.read()

    profile_obj: dict | None = None
    if profile:
        try:
            profile_obj = UserProfile(**json.loads(profile)).model_dump()
        except Exception:
            profile_obj = None

    try:
        return analyze_photo_service(
            current_user,
            content,
            file.filename or "photo.jpg",
            file.content_type,
            note=note,
            profile=profile_obj,
            conversation_id=conversation_id,
        )
    except AppError as exc:
        raise _http_error(exc) from exc
