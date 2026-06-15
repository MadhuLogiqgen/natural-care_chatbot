from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserProfile(BaseModel):
    skin_type: Optional[str] = None
    hair_type: Optional[str] = None
    allergies: Optional[str] = None
    age: Optional[str] = None
    climate: Optional[str] = None
    current_routine: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    profile: Optional[UserProfile] = None
    conversation_id: Optional[str] = None
    web_search: bool = False


class Source(BaseModel):
    filename: str
    page: int
    excerpt: str
    source_type: str = "document"
    url: Optional[str] = None
    title: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    used_web_fallback: bool = False
    conversation_id: str


class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    sources: list[Source] = []
    used_web_fallback: bool = False
    created_at: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    profile: Optional[dict] = None
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationDetail(BaseModel):
    id: str
    title: str
    profile: Optional[dict] = None
    created_at: str
    updated_at: str
    messages: list[ChatMessage]


class CreateConversationRequest(BaseModel):
    title: str = "New conversation"
    profile: Optional[UserProfile] = None


class CreateConversationResponse(BaseModel):
    conversation: ConversationDetail


class IngestResponse(BaseModel):
    files_processed: int
    chunks_indexed: int
    message: str
    documents: list[str] = []


class TranscriptionResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    pdf_count: int
    indexed_chunks: int
    documents: list[str] = []


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    created_at: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
