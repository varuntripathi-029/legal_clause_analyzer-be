"""Session storage backends for contract chat sessions."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
from threading import Lock
from time import time
from typing import Any, Protocol, TypedDict

from redis.asyncio import Redis
from redis.asyncio import from_url as redis_from_url

from app.settings import Settings

logger = logging.getLogger(__name__)


class ChatSessionState(TypedDict):
    """Serializable chat session payload stored in cache."""

    model_name: str
    chat_config: dict[str, Any]
    history: list[dict[str, Any]]
    created_at: float
    updated_at: float


class SessionStore(Protocol):
    """Contract for chat session persistence backends."""

    async def startup(self) -> None:
        """Initialize the backend connection if needed."""

    async def get(self, session_id: str) -> ChatSessionState | None:
        """Return a stored session if it exists."""

    async def set(self, session_id: str, state: ChatSessionState) -> None:
        """Persist a session state."""

    async def delete(self, session_id: str) -> None:
        """Delete a stored session."""

    async def close(self) -> None:
        """Release any backend resources."""

    async def clear_all(self) -> None:
        """Clear all stored sessions across the backend entirely."""


class ChatSessionStore:
    """Bounded in-memory store used when Redis is not configured."""

    def __init__(self, ttl_seconds: int, max_sessions: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._items: dict[str, ChatSessionState] = {}
        self._lock = Lock()

    @staticmethod
    def _clone_state(state: ChatSessionState) -> ChatSessionState:
        """Return a detached copy of the payload."""
        return deepcopy(state)

    def _cleanup_locked(self, now: float) -> None:
        expired_ids = [
            session_id
            for session_id, state in self._items.items()
            if now - state["updated_at"] > self.ttl_seconds
        ]
        for session_id in expired_ids:
            self._items.pop(session_id, None)

        while len(self._items) > self.max_sessions:
            oldest_session_id = min(
                self._items,
                key=lambda session_id: self._items[session_id]["updated_at"],
            )
            self._items.pop(oldest_session_id, None)

    async def startup(self) -> None:
        """No-op for in-memory storage."""

    async def get(self, session_id: str) -> ChatSessionState | None:
        """Return an active session if one exists."""
        now = time()
        with self._lock:
            self._cleanup_locked(now)
            state = self._items.get(session_id)
            if state is None:
                return None

            updated_state = self._clone_state(state)
            updated_state["updated_at"] = now
            self._items[session_id] = updated_state
            return self._clone_state(updated_state)

    async def set(self, session_id: str, state: ChatSessionState) -> None:
        """Add or replace a session."""
        now = time()
        with self._lock:
            payload = self._clone_state(state)
            payload.setdefault("created_at", now)
            payload["updated_at"] = now
            self._items[session_id] = payload
            self._cleanup_locked(now)

    async def delete(self, session_id: str) -> None:
        """Delete a session if it exists."""
        with self._lock:
            self._items.pop(session_id, None)

    async def close(self) -> None:
        """Clear all in-memory state."""
        await self.clear_all()

    async def clear_all(self) -> None:
        """Purge all sessions stored in memory."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        """Return the number of active sessions."""
        now = time()
        with self._lock:
            self._cleanup_locked(now)
            return len(self._items)


class RedisChatSessionStore:
    """Redis-backed session store for persistent multi-worker chat state."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        max_sessions: int,
        key_prefix: str,
        client: Redis | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._key_prefix = key_prefix.rstrip(":")
        self._redis = client or redis_from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        self._index_key = f"{self._key_prefix}:sessions:index"

    def _session_key(self, session_id: str) -> str:
        return f"{self._key_prefix}:session:{session_id}"

    async def startup(self) -> None:
        """Validate the Redis connection at application startup."""
        await self._redis.ping()

    async def _write(self, session_id: str, state: ChatSessionState) -> None:
        serialized = json.dumps(state, separators=(",", ":"), ensure_ascii=True)
        pipeline = self._redis.pipeline(transaction=False)
        pipeline.set(self._session_key(session_id), serialized, ex=self.ttl_seconds)
        pipeline.zadd(self._index_key, {session_id: state["updated_at"]})
        await pipeline.execute()
        await self._enforce_limit()

    async def _enforce_limit(self) -> None:
        overflow = await self._redis.zcard(self._index_key) - self.max_sessions
        if overflow <= 0:
            return

        stale_session_ids = await self._redis.zrange(self._index_key, 0, overflow - 1)
        if not stale_session_ids:
            return

        pipeline = self._redis.pipeline(transaction=False)
        for session_id in stale_session_ids:
            pipeline.delete(self._session_key(session_id))
            pipeline.zrem(self._index_key, session_id)
        await pipeline.execute()

    async def get(self, session_id: str) -> ChatSessionState | None:
        """Return a stored session and refresh its TTL."""
        raw_state = await self._redis.get(self._session_key(session_id))
        if raw_state is None:
            await self._redis.zrem(self._index_key, session_id)
            return None

        try:
            state = json.loads(raw_state)
        except json.JSONDecodeError:
            logger.warning("Dropping unreadable Redis chat session %s", session_id)
            await self.delete(session_id)
            return None

        state["updated_at"] = time()
        await self._write(session_id, state)
        return state

    async def set(self, session_id: str, state: ChatSessionState) -> None:
        """Persist a session with a refreshed TTL."""
        payload = deepcopy(state)
        now = time()
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        await self._write(session_id, payload)

    async def delete(self, session_id: str) -> None:
        """Delete a Redis-backed session."""
        pipeline = self._redis.pipeline(transaction=False)
        pipeline.delete(self._session_key(session_id))
        pipeline.zrem(self._index_key, session_id)
        await pipeline.execute()

    async def close(self) -> None:
        """Close the Redis client connection."""
        close = getattr(self._redis, "aclose", None)
        if close is not None:
            await close()
            return

        legacy_close = getattr(self._redis, "close", None)
        if legacy_close is not None:
            maybe_result = legacy_close()
            if hasattr(maybe_result, "__await__"):
                await maybe_result

    async def clear_all(self) -> None:
        """Drop all chat sessions across the Redis architecture globally."""
        pipeline = self._redis.pipeline(transaction=False)
        session_ids = await self._redis.zrange(self._index_key, 0, -1)
        for sid in session_ids:
            pipeline.delete(self._session_key(sid))
        pipeline.delete(self._index_key)
        await pipeline.execute()


def create_chat_session_store(settings: Settings) -> SessionStore:
    """Create the configured chat session storage backend."""
    if settings.redis_url:
        return RedisChatSessionStore(
            redis_url=settings.redis_url,
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.max_chat_sessions,
            key_prefix=settings.redis_key_prefix,
        )

    logger.warning(
        "REDIS_URL is not configured; falling back to in-memory chat sessions.",
    )
    return ChatSessionStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_sessions=settings.max_chat_sessions,
    )
