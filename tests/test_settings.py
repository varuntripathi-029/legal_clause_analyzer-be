from __future__ import annotations

import unittest

from app.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_comma_separated_lists_are_parsed(self) -> None:
        settings = Settings(
            _env_file=None,
            debug=False,
            gemini_api_key="test-key",
            cors_origins="https://a.example.com,https://b.example.com",
            trusted_hosts="api.example.com,localhost",
        )

        self.assertEqual(
            settings.cors_origins,
            ["https://a.example.com", "https://b.example.com"],
        )
        self.assertEqual(
            settings.trusted_hosts,
            ["api.example.com", "localhost"],
        )

    def test_production_rejects_wildcard_cors(self) -> None:
        settings = Settings(
            _env_file=None,
            environment="production",
            debug=False,
            gemini_api_key="test-key",
            cors_origins=["*"],
            trusted_hosts=["api.example.com"],
        )

        with self.assertRaises(ValueError):
            settings.validate_for_runtime()


if __name__ == "__main__":
    unittest.main()
