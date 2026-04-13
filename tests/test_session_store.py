from __future__ import annotations

import time
import unittest

from app.session_store import ChatSessionStore


class ChatSessionStoreTests(unittest.TestCase):
    def test_expired_sessions_are_not_returned(self) -> None:
        store = ChatSessionStore(ttl_seconds=1, max_sessions=10)
        store.set("session-1", object())

        time.sleep(1.1)

        self.assertIsNone(store.get("session-1"))
        self.assertEqual(len(store), 0)

    def test_store_respects_max_sessions(self) -> None:
        store = ChatSessionStore(ttl_seconds=60, max_sessions=2)
        store.set("session-1", "a")
        store.set("session-2", "b")
        store.set("session-3", "c")

        self.assertIsNone(store.get("session-1"))
        self.assertEqual(store.get("session-2"), "b")
        self.assertEqual(store.get("session-3"), "c")


if __name__ == "__main__":
    unittest.main()
