# -*- coding: utf-8 -*-
"""The voice profile storage model."""
import base64
import binascii
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ...._utils._common import _generate_id
from ._base import _RecordBase


_ENGINE_TYPE = Literal[
    "cosyvoice",
    "dashscope_tts",
    "openai_tts",
    "gemini_tts",
    "kokoro",
    "chatterbox",
    "luxtts",
    "tada",
    "remote_tts",
]

_SOURCE_TYPE = Literal["api", "local"]
_MAX_REFERENCE_AUDIO_BYTES = 10 * 1024 * 1024
_MAX_REFERENCE_AUDIO_BASE64_CHARS = ((_MAX_REFERENCE_AUDIO_BYTES + 2) // 3) * 4
_REFERENCE_AUDIO_MEDIA_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/m4a",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "audio/opus",
    "audio/wav",
    "audio/webm",
    "audio/x-wav",
}

ENGINE_TO_CREDENTIAL_TYPE: dict[str, str] = {
    "cosyvoice": "dashscope_credential",
    "dashscope_tts": "dashscope_credential",
    "openai_tts": "openai_credential",
    "gemini_tts": "gemini_credential",
    "kokoro": "local_tts_credential",
    "chatterbox": "local_tts_credential",
    "luxtts": "local_tts_credential",
    "tada": "local_tts_credential",
    "remote_tts": "remote_tts_credential",
}

ENGINE_SOURCE: dict[str, _SOURCE_TYPE] = {
    "cosyvoice": "api",
    "dashscope_tts": "api",
    "openai_tts": "api",
    "gemini_tts": "api",
    "kokoro": "local",
    "chatterbox": "local",
    "luxtts": "local",
    "tada": "local",
    "remote_tts": "api",
}

ENGINE_GPU_REQUIREMENT: dict[str, str | None] = {
    "cosyvoice": None,
    "dashscope_tts": None,
    "openai_tts": None,
    "gemini_tts": None,
    "kokoro": None,
    "chatterbox": "CUDA recommended",
    "luxtts": "<1 GB VRAM",
    "tada": "CUDA recommended",
    "remote_tts": None,
}

ENGINE_VOICE_CLONING: dict[str, bool] = {
    "cosyvoice": True,
    "dashscope_tts": True,
    "openai_tts": True,
    "gemini_tts": False,
    "kokoro": False,
    "chatterbox": True,
    "luxtts": True,
    "tada": True,
    "remote_tts": True,
}


class VoiceProfileData(BaseModel):
    """The voice profile data model."""

    name: str = Field(
        description="Display name for this voice profile.",
        title="Name",
    )

    engine: _ENGINE_TYPE | None = Field(
        default=None,
        description=(
            "TTS engine: cosyvoice, dashscope_tts, "
            "openai_tts, gemini_tts (API) or kokoro, "
            "chatterbox, luxtts, tada (local), "
            "or remote_tts (OpenAI-compatible API)."
        ),
        title="Engine",
    )

    model: str | None = Field(
        default=None,
        description=(
            "Specific TTS model name for synthesis "
            "(e.g. 'qwen3-tts-flash', 'cosyvoice-v3-flash')."
        ),
        title="Model",
    )

    credential_id: str | None = Field(
        default=None,
        description=(
            "Credential ID used for voice cloning and "
            "synthesis. Ensures the same API key is used "
            "for both operations."
        ),
        title="Credential ID",
    )

    source: _SOURCE_TYPE | None = Field(
        default=None,
        description=(
            "Deployment source: 'api' for cloud-based "
            "or 'local' for on-device inference."
        ),
        title="Source",
    )

    voice: str | None = Field(
        default=None,
        description=(
            "Voice identifier: preset name (e.g. 'alloy')"
            " or cloned voice ID from API."
        ),
        title="Voice",
    )

    metadata: dict | None = Field(
        default=None,
        description="Engine-specific extra configuration.",
        title="Metadata",
    )

    @field_validator("metadata")
    @classmethod
    def _validate_reference_audio(cls, metadata: dict | None) -> dict | None:
        """Reject malformed or oversized inline reference audio."""
        if metadata is None:
            return None
        media_type = metadata.get("reference_audio_media_type")
        if media_type is not None:
            if (
                not isinstance(media_type, str)
                or media_type not in _REFERENCE_AUDIO_MEDIA_TYPES
            ):
                raise ValueError(
                    "reference_audio_media_type must be a supported "
                    "audio MIME type",
                )
        audio_base64 = metadata.get("reference_audio_base64")
        if audio_base64 is None:
            return metadata
        if not isinstance(audio_base64, str):
            raise ValueError("reference_audio_base64 must be a string")
        if len(audio_base64) > _MAX_REFERENCE_AUDIO_BASE64_CHARS:
            raise ValueError("reference audio must not exceed 10 MiB")
        try:
            audio = base64.b64decode(audio_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError(
                "reference_audio_base64 must be valid base64",
            ) from error
        if len(audio) > _MAX_REFERENCE_AUDIO_BYTES:
            raise ValueError("reference audio must not exceed 10 MiB")
        return metadata

    @model_validator(mode="after")
    def _derive_source_from_engine(self) -> "VoiceProfileData":
        """Keep the persisted source consistent with the engine."""
        if self.engine is not None:
            self.source = ENGINE_SOURCE[self.engine]
        return self


def get_missing_voice_profile_binding_fields(
    data: VoiceProfileData,
) -> list[str]:
    """Return missing fields required to authorize and use a voice profile.

    Both profile CRUD validation and runtime TTS validation must use this
    definition so a profile accepted at write time is also usable later.
    """
    binding_fields = {
        "engine": data.engine,
        "credential_id": data.credential_id,
        "model": data.model,
    }
    # Reference-audio profiles do not use a provider voice id. Their ownership
    # is still bound to the exact engine, credential and model.
    if data.engine not in {
        "chatterbox",
        "luxtts",
        "tada",
        "remote_tts",
    }:
        binding_fields["voice"] = data.voice
    return [
        name
        for name, value in binding_fields.items()
        if not isinstance(value, str) or not value.strip()
    ]


class VoiceProfileRecord(_RecordBase):
    """The voice profile ORM model."""

    user_id: str = Field(
        default_factory=_generate_id,
    )
    """The owner user id."""

    data: VoiceProfileData
    """The voice profile data."""
