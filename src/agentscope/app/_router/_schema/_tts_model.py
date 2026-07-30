# -*- coding: utf-8 -*-
"""The TTS model configuration, used as DTO layer."""

from pydantic import BaseModel, Field

from ....tts import TTSModelCard


class ListTTSModelsResponse(BaseModel):
    """List the candidate TTS models response."""

    models: list[TTSModelCard] = Field(
        description="The candidate TTS models.",
    )
    total: int = Field(description="The total number of candidates.")


class ListTTSModelsRequest(BaseModel):
    """List the candidate TTS models request."""

    provider: str = Field(
        description="The provider type, e.g. dashscope_credential.",
    )


class VoiceboxClientSetupResponse(BaseModel):
    """Voicebox connection and per-user client-binding status."""

    client_id: str = Field(
        description="The client id to configure in Voicebox Settings -> MCP.",
    )
    endpoint: str = Field(description="The configured Voicebox base URL.")
    reachable: bool = Field(
        description="Whether the Voicebox binding endpoint is reachable.",
    )
    profile_id: str | None = Field(
        default=None,
        description="The Voicebox profile currently bound to this client.",
    )
    default_engine: str | None = Field(
        default=None,
        description="The optional engine configured on the binding.",
    )
    error: str | None = Field(
        default=None,
        description="An actionable connection or compatibility error.",
    )
