import asyncio
import unittest

from app.session_store import ChatSessionStore


class ChatSessionStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_expired_sessions_are_not_returned(self) -> None:
        store = ChatSessionStore(ttl_seconds=1, max_sessions=10)
        await store.set("session-1", {})

        await asyncio.sleep(1.1)

        self.assertIsNone(await store.get("session-1"))
        self.assertEqual(len(store), 0)

    async def test_store_respects_max_sessions(self) -> None:
        store = ChatSessionStore(ttl_seconds=60, max_sessions=2)
        await store.set("session-1", {})
        await store.set("session-2", {})
        await store.set("session-3", {})

        self.assertIsNone(await store.get("session-1"))
        self.assertIsNotNone(await store.get("session-2"))
        self.assertIsNotNone(await store.get("session-3"))


if __name__ == "__main__":
    unittest.main()
