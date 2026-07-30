# -*- coding: utf-8 -*-
"""Credential for OpenAI-compatible remote TTS services."""
from typing import Literal, Type, TYPE_CHECKING
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, SecretStr, field_validator

from ._base import CredentialBase

if TYPE_CHECKING:
    from ..tts import TTSModelBase


class RemoteTTSCredential(CredentialBase):
    """Connection settings for an OpenAI-compatible TTS endpoint."""

    model_config = ConfigDict(title="Remote TTS")

    type: Literal["remote_tts_credential"] = "remote_tts_credential"
    """The credential type."""

    base_url: str = Field(
        description=(
            "Base URL of the remote TTS service. "
            "Example: http://127.0.0.1:8091/v1. "
            "Input text and reference audio are sent to this endpoint."
        ),
    )
    """Remote API base URL, including the ``/v1`` prefix."""

    api_key: SecretStr | None = Field(
        default=None,
        description="Optional bearer token for the remote TTS service.",
    )
    """Optional bearer token."""

    timeout: float = Field(
        default=120.0,
        gt=0,
        le=600,
        description="Request timeout in seconds.",
    )
    """Request timeout."""

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        """Accept only an HTTP(S) API base URL without embedded secrets."""
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(
                "base_url must be an absolute http or https URL",
            )
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain user credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if normalized.endswith("/audio/speech"):
            raise ValueError(
                "base_url must end at the API root (for example /v1), "
                "not /audio/speech",
            )
        return normalized

    @classmethod
    def get_chat_model_class(cls) -> Type:
        """Remote TTS does not provide chat models."""
        raise NotImplementedError(
            "RemoteTTSCredential does not support chat models.",
        )

    @classmethod
    def list_models(cls) -> list:
        """Remote TTS does not provide chat models."""
        return []

    @classmethod
    def get_tts_model_classes(cls) -> list[Type["TTSModelBase"]]:
        """Return the remote TTS adapter class."""
        from ..tts import RemoteTTSModel

        return [RemoteTTSModel]
