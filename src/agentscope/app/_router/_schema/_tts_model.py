# -*- coding: utf-8 -*-
"""The TTS model configuration, used as DTO layer."""

from pydantic import BaseModel, Field, model_validator

from ....tts import TTSModelCard


class ListTTSModelsResponse(BaseModel):
    """List the candidate TTS models response."""

    models: list[TTSModelCard] = Field(
        description="The candidate TTS models.",
    )
    total: int = Field(description="The total number of candidates.")


class ListTTSModelsRequest(BaseModel):
    """List the candidate TTS models request."""

    provider: str | None = Field(
        default=None,
        description="The provider type, e.g. dashscope_credential.",
    )

    credential_id: str | None = Field(
        default=None,
        description=(
            "A concrete credential used for endpoint-specific discovery."
        ),
    )

    @model_validator(mode="after")
    def _require_lookup_scope(self) -> "ListTTSModelsRequest":
        """Require either a provider type or a concrete credential."""
        if self.provider is None and self.credential_id is None:
            raise ValueError("provider or credential_id is required")
        return self
