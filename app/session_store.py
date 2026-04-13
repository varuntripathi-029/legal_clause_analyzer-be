"""In-memory chat session storage with TTL and size limits."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Any


@dataclass
class _SessionEntry:
    """Internal session record."""

    chat: Any
    created_at: float
    last_accessed_at: float


class ChatSessionStore:
    """Bounded in-memory store for Gemini chat sessions."""

    def __init__(self, ttl_seconds: int, max_sessions: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._items: dict[str, _SessionEntry] = {}
        self._lock = Lock()

    def _cleanup_locked(self, now: float) -> None:
        expired_ids = [
            session_id
            for session_id, entry in self._items.items()
            if now - entry.last_accessed_at > self.ttl_seconds
        ]
        for session_id in expired_ids:
            self._items.pop(session_id, None)

        while len(self._items) > self.max_sessions:
            oldest_session_id = min(
                self._items,
                key=lambda session_id: self._items[session_id].last_accessed_at,
            )
            self._items.pop(oldest_session_id, None)

    def set(self, session_id: str, chat: Any) -> None:
        """Add or replace a chat session."""
        now = time()
        with self._lock:
            self._items[session_id] = _SessionEntry(
                chat=chat,
                created_at=now,
                last_accessed_at=now,
            )
            self._cleanup_locked(now)

    def get(self, session_id: str) -> Any | None:
        """Return an active chat session if one exists."""
        now = time()
        with self._lock:
            self._cleanup_locked(now)
            entry = self._items.get(session_id)
            if entry is None:
                return None

            entry.last_accessed_at = now
            return entry.chat

    def clear(self) -> None:
        """Remove all stored sessions."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        """Return the number of active sessions."""
        now = time()
        with self._lock:
            self._cleanup_locked(now)
            return len(self._items)
