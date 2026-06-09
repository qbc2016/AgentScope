# -*- coding: utf-8 -*-
"""The TTS model configuration, used as DTO layer."""

from pydantic import BaseModel, Field

from ....model import ModelCard


class ListTTSModelsResponse(BaseModel):
    """List the candidate TTS models response."""

    models: list[ModelCard] = Field(
        description="The candidate TTS models.",
    )
    total: int = Field(description="The total number of candidates.")


class ListTTSModelsRequest(BaseModel):
    """List the candidate TTS models request."""

    provider: str = Field(
        description="The provider type, e.g. dashscope_credential.",
    )
