import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.config import DB_PATH
from app.models import Source, UserProfile


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def init_db() -> None:
    with get_connection() as conn:
        user_cols = _table_columns(conn, "users")
        conv_cols = _table_columns(conn, "conversations")

        if user_cols and "email" not in user_cols:
            conn.executescript(
                """
                DROP TABLE IF EXISTS messages;
                DROP TABLE IF EXISTS conversations;
                DROP TABLE IF EXISTS users;
                """
            )
            user_cols = set()
            conv_cols = set()

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        if not conv_cols:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    profile_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )
        elif "user_id" not in conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                used_web_fallback INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_user
                ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id);
            """
        )


def _profile_to_json(profile: Optional[UserProfile]) -> Optional[str]:
    if not profile:
        return None
    data = profile.model_dump(exclude_none=True)
    return json.dumps(data) if data else None


def _json_to_profile(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    return json.loads(raw)


def _sources_to_json(sources: Optional[list[Source]]) -> Optional[str]:
    if not sources:
        return None
    return json.dumps([source.model_dump() for source in sources])


def _json_to_sources(raw: Optional[str]) -> list[dict]:
    if not raw:
        return []
    return json.loads(raw)


def _conversation_title(question: str) -> str:
    title = " ".join(question.split())
    if len(title) > 60:
        title = f"{title[:57]}..."
    return title or "New conversation"


def create_user(email: str, password_hash: str) -> dict[str, Any]:
    now = _utc_now()
    user_id = str(uuid.uuid4())
    normalized_email = email.strip().lower()
    with get_connection() as conn:
        try:
            conn.execute(
                """
                INSERT INTO users (id, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, normalized_email, password_hash, now),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("An account with this email already exists.") from exc
    return get_user_by_id(user_id)


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    normalized_email = email.strip().lower()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "password_hash": row["password_hash"],
        "created_at": row["created_at"],
    }


def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "created_at": row["created_at"],
    }


def _conversation_owned_by_user(conversation_id: str, user_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
    return row is not None


def create_conversation(
    user_id: str,
    title: str = "New conversation",
    profile: Optional[UserProfile] = None,
) -> dict[str, Any]:
    now = _utc_now()
    conversation_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, user_id, title, profile_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                user_id,
                title,
                _profile_to_json(profile),
                now,
                now,
            ),
        )
    return get_conversation(conversation_id, user_id)


def list_conversations(user_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.profile_json, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.user_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """,
            (user_id,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "profile": _json_to_profile(row["profile_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": row["message_count"],
        }
        for row in rows
    ]


def get_conversation(conversation_id: str, user_id: str) -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        if not row:
            raise KeyError(conversation_id)

        messages = conn.execute(
            """
            SELECT id, role, content, sources_json, used_web_fallback, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        ).fetchall()

    return {
        "id": row["id"],
        "title": row["title"],
        "profile": _json_to_profile(row["profile_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": [
            {
                "id": message["id"],
                "role": message["role"],
                "content": message["content"],
                "sources": _json_to_sources(message["sources_json"]),
                "used_web_fallback": bool(message["used_web_fallback"]),
                "created_at": message["created_at"],
            }
            for message in messages
        ],
    }


def delete_conversation(conversation_id: str, user_id: str) -> None:
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(conversation_id)


def add_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    sources: Optional[list[Source]] = None,
    used_web_fallback: bool = False,
) -> dict[str, Any]:
    if not _conversation_owned_by_user(conversation_id, user_id):
        raise KeyError(conversation_id)

    now = _utc_now()
    message_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                id, conversation_id, role, content, sources_json,
                used_web_fallback, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                role,
                content,
                _sources_to_json(sources),
                int(used_web_fallback),
                now,
            ),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ? AND user_id = ?",
            (now, conversation_id, user_id),
        )

    return {
        "id": message_id,
        "role": role,
        "content": content,
        "sources": [source.model_dump() for source in sources or []],
        "used_web_fallback": used_web_fallback,
        "created_at": now,
    }


def ensure_conversation(
    user_id: str,
    conversation_id: Optional[str],
    question: str,
    profile: Optional[UserProfile] = None,
) -> str:
    if conversation_id and _conversation_owned_by_user(conversation_id, user_id):
        if profile:
            with get_connection() as conn:
                conn.execute(
                    """
                    UPDATE conversations
                    SET profile_json = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (_profile_to_json(profile), conversation_id, user_id),
                )
        return conversation_id

    conversation = create_conversation(
        user_id=user_id,
        title=_conversation_title(question),
        profile=profile,
    )
    return conversation["id"]


def update_conversation_title_if_first_message(
    conversation_id: str,
    user_id: str,
    question: str,
) -> None:
    with get_connection() as conn:
        count = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()["total"]
        if count == 0:
            conn.execute(
                """
                UPDATE conversations
                SET title = ?
                WHERE id = ? AND user_id = ?
                """,
                (_conversation_title(question), conversation_id, user_id),
            )
