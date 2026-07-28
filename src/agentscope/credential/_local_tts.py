# -*- coding: utf-8 -*-
"""The Local TTS credential."""
from typing import Literal, Type, TYPE_CHECKING

from pydantic import ConfigDict, Field

from ._base import CredentialBase

if TYPE_CHECKING:
    from ..tts import TTSModelBase


class LocalTTSCredential(CredentialBase):
    """The Local TTS credential model.

    Used by local TTS engines (Kokoro, Chatterbox, LuxTTS, TADA)
    that run on the same machine. No API key is needed; only the
    preferred compute device is stored.
    """

    model_config = ConfigDict(
        title="Local TTS",
    )

    type: Literal["local_tts_credential"] = "local_tts_credential"
    """The credential type."""

    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Compute device for local TTS inference.",
    )
    """The preferred compute device."""

    @classmethod
    def get_chat_model_class(cls) -> Type:
        """Local TTS does not provide chat models."""
        raise NotImplementedError(
            "LocalTTSCredential does not support chat models.",
        )

    @classmethod
    def list_models(cls) -> list:
        """Local TTS does not provide chat models."""
        return []

    @classmethod
    def get_tts_model_classes(
        cls,
    ) -> list[Type["TTSModelBase"]]:
        """Return classes whose optional engine dependencies exist."""
        from ..tts import (
            KokoroTTSModel,
            ChatterboxTTSModel,
            LuxTTSModel,
            TadaTTSModel,
        )
        from ..tts._local._utils import is_local_tts_engine_available

        classes: list[Type["TTSModelBase"]] = []
        if is_local_tts_engine_available("kokoro"):
            classes.append(KokoroTTSModel)
        if is_local_tts_engine_available("chatterbox"):
            classes.append(ChatterboxTTSModel)
        if is_local_tts_engine_available("luxtts"):
            classes.append(LuxTTSModel)
        if is_local_tts_engine_available("tada"):
            classes.append(TadaTTSModel)
        return classes
