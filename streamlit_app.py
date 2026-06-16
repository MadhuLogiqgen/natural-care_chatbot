import os

# Must run before chromadb/protobuf-dependent imports on Streamlit Cloud.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import hashlib

import streamlit as st

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
    require_user,
    transcribe as transcribe_service,
    upload_pdfs as upload_pdfs_service,
)


@st.cache_resource
def init_backend() -> bool:
    ensure_initialized()
    return True


init_backend()


st.set_page_config(
    page_title="Natural Care Assistant",
    page_icon="🌿",
    layout="wide",
)

# --- ChatGPT-style theme (modern light look: white chat area, dark sidebar) ---
CHATGPT_CSS = """
<style>
:root {
    --ncc-bg: #ffffff;
    --ncc-sidebar-bg: #171717;
    --ncc-sidebar-text: #ececec;
    --ncc-user-bubble: #f4f4f4;
    --ncc-text: #0d0d0d;
    --ncc-border: #e5e5e5;
    --ncc-accent: #10a37f;
}

/* Base font to match ChatGPT */
html, body, [class*="css"], .stMarkdown, p, li {
    font-family: "Söhne", ui-sans-serif, -apple-system, "Segoe UI",
        Helvetica, Arial, sans-serif;
}

/* White app background, hide Streamlit chrome */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--ncc-bg);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Center and constrain the chat column like ChatGPT, full-height so the
   input bar can be pushed to the bottom of the screen */
[data-testid="stMain"] .block-container {
    max-width: 800px;
    padding-top: 2.5rem;
    padding-bottom: 0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* CRITICAL: Streamlit nests the real content one level deeper, inside the
   main vertical block. That block sizes to its content, so without this it
   stays short and `margin-top: auto` on the input bar has no free space to
   consume — leaving the bar stranded in the middle. Forcing it to grow to
   fill the 100vh container restores the space the auto-margin needs. */
[data-testid="stMain"] .block-container > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto;
}

/* Input bar pinned to the bottom — stays put while messages scroll behind it.
   `margin-top: auto` drops it to the bottom when the chat is short;
   `position: sticky; bottom: 0` keeps it glued there while scrolling. Because
   it stays in document flow, it reserves its own space, so messages are never
   hidden behind it. */
.st-key-chat_input_bar {
    margin-top: auto;
    position: sticky;
    bottom: 0;
    background: var(--ncc-bg);
    padding-top: 1rem;
    padding-bottom: 0.6rem;
    z-index: 99;
}
/* Soft fade so messages dissolve behind the bar instead of cutting off hard */
.st-key-chat_input_bar::before {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: -2.5rem;
    height: 2.5rem;
    background: linear-gradient(to top, var(--ncc-bg), rgba(255, 255, 255, 0));
    pointer-events: none;
}

/* Responsive: edge-to-edge with comfortable side padding on phones/tablets */
@media (max-width: 768px) {
    [data-testid="stMain"] .block-container {
        max-width: 100%;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
        max-width: 90%;
    }
}

/* Title block: tighter, centered */
[data-testid="stMain"] h1 {
    text-align: center;
    font-weight: 600;
    color: var(--ncc-text);
}
[data-testid="stMain"] [data-testid="stCaptionContainer"] {
    text-align: center;
}

/* --- Chat messages --- */
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 0.35rem 0;
    box-shadow: none;
}

/* Assistant: plain text, full width, no bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
[data-testid="stChatMessageContent"] {
    background: transparent;
    color: var(--ncc-text);
}

/* User: gray rounded bubble, pushed to the right */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] {
    background: var(--ncc-user-bubble);
    color: var(--ncc-text);
    border-radius: 1.4rem;
    padding: 0.65rem 1.05rem;
    max-width: 80%;
}

/* --- Chat input: rounded pill with soft shadow --- */
[data-testid="stChatInput"] {
    border-radius: 1.75rem;
    border: 1px solid var(--ncc-border);
    box-shadow: 0 2px 14px rgba(0,0,0,0.07);
    background: #ffffff;
}
[data-testid="stChatInput"] textarea { background: transparent; }

/* --- Dark sidebar --- */
section[data-testid="stSidebar"] {
    background: var(--ncc-sidebar-bg);
}
section[data-testid="stSidebar"] * {
    color: var(--ncc-sidebar-text);
}
section[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.12);
    color: var(--ncc-sidebar-text);
    border-radius: 0.6rem;
    text-align: left;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.2);
}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: rgba(255,255,255,0.12);
    border-color: rgba(255,255,255,0.25);
}
/* Inputs inside the dark sidebar */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06) !important;
    color: var(--ncc-sidebar-text) !important;
    border-color: rgba(255,255,255,0.12) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.04);
    border-color: rgba(255,255,255,0.15);
}
</style>
"""
st.markdown(CHATGPT_CSS, unsafe_allow_html=True)

if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "processed_upload_keys" not in st.session_state:
    st.session_state.processed_upload_keys = set()
if "processed_audio_keys" not in st.session_state:
    st.session_state.processed_audio_keys = set()
if "voice_prompt" not in st.session_state:
    st.session_state.voice_prompt = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "pending_web_search" not in st.session_state:
    st.session_state.pending_web_search = False
if "pending_photo" not in st.session_state:
    st.session_state.pending_photo = None
if "processed_photo_keys" not in st.session_state:
    st.session_state.processed_photo_keys = set()


def current_user() -> dict:
    return require_user(st.session_state.access_token)


def login_user(email: str, password: str) -> None:
    result = login(email, password)
    st.session_state.access_token = result["access_token"]
    st.session_state.user_email = result["user"]["email"]
    st.session_state.messages = []
    st.session_state.conversation_id = None


def register_user(email: str, password: str) -> None:
    result = register(email, password)
    st.session_state.access_token = result["access_token"]
    st.session_state.user_email = result["user"]["email"]
    st.session_state.messages = []
    st.session_state.conversation_id = None


def logout_user() -> None:
    st.session_state.access_token = None
    st.session_state.user_email = None
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.processed_upload_keys = set()
    st.session_state.processed_audio_keys = set()
    st.session_state.voice_prompt = None
    st.session_state.pending_prompt = None
    st.session_state.pending_web_search = False
    st.session_state.pending_photo = None
    st.session_state.processed_photo_keys = set()


def upload_file_key(file) -> str:
    return f"{file.name}:{file.size}"


def load_conversation(conversation_id: str) -> None:
    data = get_conversation_detail(current_user(), conversation_id)
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = [
        {
            "role": message["role"],
            "content": message["content"],
            "sources": message.get("sources", []),
            "used_web_fallback": message.get("used_web_fallback", False),
        }
        for message in data["messages"]
    ]


def start_new_chat() -> None:
    st.session_state.conversation_id = None
    st.session_state.messages = []


def render_sources(sources: list[dict], used_web_fallback: bool = False) -> None:
    if used_web_fallback:
        st.caption("No matching PDF content — answer supplemented from web search.")
    if not sources:
        return
    label = "Web sources" if used_web_fallback else "Sources"
    with st.expander(label):
        for src in sources:
            if src.get("source_type") == "web":
                title = src.get("title") or src.get("filename", "Web result")
                if src.get("url"):
                    st.markdown(f"**[{title}]({src['url']})**")
                else:
                    st.markdown(f"**{title}**")
            else:
                st.markdown(f"**{src['filename']}** (page {src['page']})")
            st.caption(src.get("excerpt", ""))


def show_auth_page() -> None:
    st.title("🌿 Natural Face & Hair Care Assistant")
    st.caption("Log in to save your chat history and get personalized advice.")

    tab_login, tab_register = st.tabs(["Log in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
            if submitted:
                if not email or not password:
                    st.error("Please enter email and password.")
                else:
                    try:
                        login_user(email, password)
                        st.success("Welcome back!")
                        st.rerun()
                    except AppError as exc:
                        st.error(exc.detail)
                    except Exception as exc:
                        st.error(str(exc))

    with tab_register:
        with st.form("register_form"):
            email = st.text_input("Email", key="register_email")
            password = st.text_input(
                "Password",
                type="password",
                key="register_password",
                help="At least 6 characters.",
            )
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
            if submitted:
                if not email or not password:
                    st.error("Please enter email and password.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        register_user(email, password)
                        st.success("Account created!")
                        st.rerun()
                    except AppError as exc:
                        st.error(exc.detail)
                    except Exception as exc:
                        st.error(str(exc))


if not st.session_state.access_token:
    show_auth_page()
    st.stop()

st.title("🌿 Natural Face & Hair Care Assistant")
st.caption("Educational tips from your natural care documents — not medical advice.")

with st.sidebar:
    st.write(f"Signed in as **{st.session_state.user_email}**")
    if st.button("Log out", use_container_width=True):
        logout_user()
        st.rerun()

    st.divider()
    st.header("Conversations")
    st.caption("Your chat history is saved to your account.")

    if st.button("New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    try:
        conversations = get_conversations(current_user())
        if conversations:
            for conversation in conversations:
                is_active = conversation["id"] == st.session_state.conversation_id
                label = conversation["title"]
                if conversation["message_count"]:
                    label = f"{label} ({conversation['message_count']})"
                if st.button(
                    label,
                    key=f"conv_{conversation['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    load_conversation(conversation["id"])
                    st.rerun()
        else:
            st.caption("No saved conversations yet.")
    except AppError as exc:
        if exc.status_code == 401:
            logout_user()
            st.rerun()
        st.caption(f"Could not load conversations: {exc.detail}")
    except Exception as exc:
        st.caption(f"Could not load conversations: {exc}")

    if st.session_state.conversation_id:
        if st.button("Delete this conversation", use_container_width=True):
            try:
                remove_conversation(current_user(), st.session_state.conversation_id)
                start_new_chat()
                st.rerun()
            except AppError as exc:
                st.error(exc.detail)
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.header("Your profile")
    st.caption("Optional — helps tailor routines and product ideas.")

    skin_type = st.selectbox(
        "Skin type",
        ["", "Normal", "Dry", "Oily", "Combination", "Sensitive"],
    )
    hair_type = st.selectbox(
        "Hair type",
        ["", "Straight", "Wavy", "Curly", "Coily", "Fine", "Thick"],
    )
    allergies = st.text_input("Allergies / sensitivities")
    age = st.text_input("Age range", placeholder="e.g. 25-35")
    climate = st.text_input("Climate", placeholder="e.g. humid tropical")
    current_routine = st.text_area(
        "Current routine",
        placeholder="What you use now for face and hair...",
        height=100,
    )

    st.divider()
    st.header("Knowledge base")
    st.caption("Upload your natural care PDFs — they are indexed automatically.")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Add guides, ingredient lists, or product info. You can upload multiple files.",
    )

    if uploaded_files:
        pending = [
            f
            for f in uploaded_files
            if upload_file_key(f) not in st.session_state.processed_upload_keys
        ]
        if pending:
            with st.spinner(
                f"Indexing {len(pending)} PDF{'s' if len(pending) != 1 else ''}..."
            ):
                try:
                    result = upload_pdfs_service(
                        current_user(),
                        [(f.name, f.getvalue()) for f in pending],
                    )
                    for f in pending:
                        st.session_state.processed_upload_keys.add(
                            upload_file_key(f)
                        )
                    st.success(result["message"])
                    st.caption(
                        f"Added {result['chunks_indexed']} searchable sections "
                        f"from {result['files_processed']} file(s)."
                    )
                    st.rerun()
                except AppError as exc:
                    st.error(exc.detail)
                except Exception as exc:
                    st.error(str(exc))

    try:
        health = health_check()
        st.success("Backend ready")
        st.write(f"Documents: **{health['pdf_count']}**")
        st.write(f"Indexed sections: **{health['indexed_chunks']}**")

        if health.get("documents"):
            with st.expander("View uploaded documents"):
                for name in health["documents"]:
                    st.markdown(f"- {name}")
        elif health["pdf_count"] == 0:
            st.info("Upload a PDF above to start asking questions.")
    except Exception as exc:
        st.error("Backend could not start.")
        st.caption(str(exc))
        health = None

    if health and health.get("pdf_count", 0) > 0:
        if st.button("Re-index all documents", use_container_width=True):
            with st.spinner("Rebuilding search index..."):
                try:
                    result = reingest_documents(current_user())
                    st.success(
                        f"Re-indexed {result['chunks_indexed']} sections "
                        f"from {result['files_processed']} document(s)."
                    )
                    st.rerun()
                except AppError as exc:
                    st.error(exc.detail)
                except Exception as exc:
                    st.error(str(exc))

def current_profile() -> dict:
    return {
        k: v
        for k, v in {
            "skin_type": skin_type or None,
            "hair_type": hair_type or None,
            "allergies": allergies or None,
            "age": age or None,
            "climate": climate or None,
            "current_routine": current_routine or None,
        }.items()
        if v
    }


def answer_pending_prompt() -> None:
    """Call the backend for the queued user message and store the reply.

    The user message is already in history; we render a 'thinking' bubble
    here (above the input bar) while the API call runs, then rerun so the
    saved answer renders through the history loop.
    """
    prompt = st.session_state.pending_prompt
    web_search = st.session_state.pending_web_search
    st.session_state.pending_prompt = None
    st.session_state.pending_web_search = False

    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("Searching the web..." if web_search else "Thinking..."):
            try:
                result = ask_service(
                    current_user(),
                    prompt,
                    profile=current_profile() or None,
                    conversation_id=st.session_state.conversation_id,
                    web_search=web_search,
                )
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "used_web_fallback": result.get("used_web_fallback", False),
                    }
                )
            except AppError as exc:
                if exc.status_code == 401:
                    logout_user()
                    st.rerun()
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {exc.detail}"}
                )
            except Exception as exc:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ Could not get an answer: {exc}"}
                )
    st.rerun()


def analyze_pending_photo() -> None:
    """Send the queued photo to the backend and store the recommendations.

    The user's photo is already shown in history; we render a 'thinking'
    bubble while the vision + RAG pipeline runs, then rerun so the saved
    answer renders through the history loop.
    """
    photo = st.session_state.pending_photo
    st.session_state.pending_photo = None

    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("Analyzing your photo and finding natural products..."):
            try:
                result = analyze_photo_service(
                    current_user(),
                    photo["bytes"],
                    photo["name"],
                    photo["mime"],
                    note=photo["note"],
                    profile=current_profile() or None,
                    conversation_id=st.session_state.conversation_id,
                )
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "used_web_fallback": result.get("used_web_fallback", False),
                    }
                )
            except AppError as exc:
                if exc.status_code == 401:
                    logout_user()
                    st.rerun()
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {exc.detail}"}
                )
            except Exception as exc:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ Could not analyze the photo: {exc}"}
                )
    st.rerun()


for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🌿"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("image"):
            st.image(message["image"], width=240)
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(
                message["sources"],
                used_web_fallback=message.get("used_web_fallback", False),
            )

# Answer a queued message right under the chat history, above the input bar.
if st.session_state.pending_photo:
    analyze_pending_photo()
elif st.session_state.pending_prompt:
    answer_pending_prompt()

# ChatGPT-style input row: text box with a voice-record button beside it.
st.markdown(
    """
    <style>
    /* Compact the voice recorder so it sits like a mic button next to the chat box */
    div[data-testid="stAudioInput"] {
        min-width: 0;
    }
    div[data-testid="stAudioInput"] > div {
        min-height: 2.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="chat_input_bar"):
    input_col, mic_col, photo_col = st.columns(
        [0.74, 0.13, 0.13], vertical_alignment="bottom"
    )

    with input_col:
        prompt = st.chat_input("Ask about natural face or hair care...")

    with mic_col:
        audio = st.audio_input(
            "Record voice",
            label_visibility="collapsed",
            key="voice_recorder",
        )

    with photo_col:
        with st.popover(
            "📷",
            use_container_width=True,
            help="Upload a photo for a natural product recommendation",
        ):
            st.markdown("**Analyze a face or hair photo**")
            st.caption("Get natural product suggestions based on your photo.")
            photo_file = st.file_uploader(
                "Upload a photo",
                type=["jpg", "jpeg", "png", "webp"],
                key="photo_uploader",
                label_visibility="collapsed",
            )
            camera_photo = st.camera_input(
                "Or take a photo",
                key="photo_camera",
                label_visibility="collapsed",
            )
            photo_note = st.text_input(
                "Anything specific? (optional)",
                key="photo_note",
                placeholder="e.g. focus on my hair frizz",
            )
            analyze_clicked = st.button(
                "Analyze photo",
                key="analyze_photo_btn",
                use_container_width=True,
                type="primary",
            )

    # Search toggle below the input — answers come from the web (ChatGPT-style).
    web_search = st.toggle(
        "🌐 Search the web",
        key="web_search",
        help="Answer from a live web search instead of only your uploaded PDFs.",
    )

if audio is not None:
    audio_bytes = audio.getvalue()
    audio_key = hashlib.md5(audio_bytes).hexdigest()
    if audio_key not in st.session_state.processed_audio_keys:
        st.session_state.processed_audio_keys.add(audio_key)
        with st.spinner("Transcribing..."):
            try:
                transcript = transcribe_service(
                    current_user(), audio_bytes, "recording.wav"
                )
                st.session_state.voice_prompt = transcript
                st.rerun()
            except AppError as exc:
                if exc.status_code == 401:
                    logout_user()
                    st.rerun()
                st.error(exc.detail)
            except Exception as exc:
                st.error(f"Could not transcribe audio: {exc}")

chosen_photo = camera_photo or photo_file
if analyze_clicked and chosen_photo is None:
    st.toast("Please upload or take a photo first.", icon="📷")
elif analyze_clicked and chosen_photo is not None:
    image_bytes = chosen_photo.getvalue()
    photo_key = hashlib.md5(image_bytes).hexdigest()
    if photo_key not in st.session_state.processed_photo_keys:
        st.session_state.processed_photo_keys.add(photo_key)
        note_text = (photo_note or "").strip()
        # Show the photo immediately (above the input bar) and queue the
        # analysis; the rerun renders it, then runs vision + RAG.
        st.session_state.messages.append(
            {
                "role": "user",
                "content": note_text or "Please analyze my photo and suggest natural products.",
                "image": image_bytes,
            }
        )
        st.session_state.pending_photo = {
            "bytes": image_bytes,
            "mime": chosen_photo.type or "image/jpeg",
            "name": getattr(chosen_photo, "name", "photo.jpg"),
            "note": note_text,
        }
        st.rerun()

if not prompt and st.session_state.voice_prompt:
    prompt = st.session_state.voice_prompt
    st.session_state.voice_prompt = None

if prompt:
    # Show the user message immediately (above the input bar) and queue the
    # reply; the rerun renders it through the history loop, then answers it.
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_prompt = prompt
    st.session_state.pending_web_search = web_search
    st.rerun()
