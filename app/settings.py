"""Application settings for local and production deployments."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Legal Contract Analyzer"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "INFO"
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
        ]
    )
    trusted_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    allow_credentials: bool = True
    gzip_minimum_size: int = 1000
    analysis_concurrency: int = 5
    max_pdf_size_bytes: int = 10 * 1024 * 1024
    max_clauses_per_document: int = 100
    session_ttl_seconds: int = 60 * 60
    max_chat_sessions: int = 250

    @field_validator("cors_origins", "trusted_hosts", mode="before")
    @classmethod
    def _parse_list_setting(cls, value: Any) -> list[str] | Any:
        """Allow comma-separated env values for list settings."""
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
            return [item for item in items if item]
        return value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """Store log level consistently."""
        return value.upper()

    @field_validator("analysis_concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int) -> int:
        """Keep concurrency within a safe range."""
        return max(1, min(value, 20))

    @field_validator("max_pdf_size_bytes", "max_clauses_per_document", "session_ttl_seconds", "max_chat_sessions")
    @classmethod
    def _ensure_positive(cls, value: int) -> int:
        """Reject non-positive capacity settings."""
        if value <= 0:
            raise ValueError("Setting must be greater than zero.")
        return value

    @field_validator("trusted_hosts")
    @classmethod
    def _ensure_trusted_hosts(cls, value: list[str]) -> list[str]:
        """Provide a permissive fallback for local development."""
        return value or ["*"]

    @field_validator("cors_origins")
    @classmethod
    def _ensure_cors_origins(cls, value: list[str]) -> list[str]:
        """Provide a permissive fallback for local development."""
        return value or ["*"]

    @field_validator("gemini_api_key")
    @classmethod
    def _strip_api_key(cls, value: str) -> str:
        """Normalize whitespace around the API key."""
        return value.strip()

    @field_validator("allow_credentials")
    @classmethod
    def _coerce_bool(cls, value: bool) -> bool:
        """Keep explicit bool typing stable."""
        return bool(value)

    def validate_for_runtime(self) -> None:
        """Apply environment-specific safety checks."""
        if self.environment == "production":
            if not self.gemini_api_key or self.gemini_api_key == "your_api_key_here":
                raise ValueError("GEMINI_API_KEY must be configured in production.")
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS is not allowed in production.")
            if "*" in self.trusted_hosts:
                raise ValueError("Wildcard trusted hosts are not allowed in production.")


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object."""
    settings = Settings()
    settings.validate_for_runtime()
    return settings
