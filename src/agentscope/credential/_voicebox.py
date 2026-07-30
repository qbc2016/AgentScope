# -*- coding: utf-8 -*-
"""The Voicebox credential."""
from typing import Literal, Type, TYPE_CHECKING

from pydantic import ConfigDict, Field

from ._base import CredentialBase

if TYPE_CHECKING:
    from ..tts import TTSModelBase


class VoiceboxCredential(CredentialBase):
    """The Voicebox credential model.

    Voicebox is a local-first AI voice studio that exposes
    an MCP server for TTS and voice cloning. No API key is
    needed since it runs locally.
    """

    model_config = ConfigDict(
        title="Voicebox (Local)",
    )

    type: Literal["voicebox_credential"] = "voicebox_credential"
    """The credential type."""

    endpoint: str = Field(
        default="http://127.0.0.1:17493",
        description=(
            "Base URL of a running Voicebox 0.5.0+ server (do not append "
            "/mcp). The URL is accessed by the AgentScope backend, not by "
            "the browser. If AgentScope runs remotely or in a container, "
            "127.0.0.1 refers to that server or container."
        ),
    )
    """The Voicebox server endpoint URL."""

    @classmethod
    def get_chat_model_class(cls) -> Type:
        """Voicebox does not provide chat models."""
        raise NotImplementedError(
            "VoiceboxCredential does not support chat models.",
        )

    @classmethod
    def list_models(cls) -> list:
        """Voicebox does not provide chat models."""
        return []

    @classmethod
    def get_tts_model_classes(cls) -> list[Type["TTSModelBase"]]:
        """Return the Voicebox TTS model class."""
        from ..tts import VoiceboxTTSModel

        return [VoiceboxTTSModel]
