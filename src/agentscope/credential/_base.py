# -*- coding: utf-8 -*-
"""The credential base class."""
import uuid
from typing import TYPE_CHECKING, Type

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..model import ChatModelBase, ModelCard
    from ..tts import TTSModelBase


class CredentialBase(BaseModel):
    """The credential base class."""

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="The credential id",
    )

    name: str = Field(
        default="",
        description="User-facing display name for this credential.",
    )

    @classmethod
    def get_chat_model_class(cls) -> Type["ChatModelBase"]:
        """Return the :class:`ChatModelBase` subclass that consumes this
        credential. Subclasses must override this method to return the
        corresponding chat model class.

        Returns:
            `Type[ChatModelBase]`:
                The chat model class that uses this credential.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement ``get_chat_model_class``.",
        )

    @classmethod
    def get_tts_model_class(cls) -> Type["TTSModelBase"] | None:
        """Return the :class:`TTSModelBase` subclass that consumes this
        credential, or ``None`` if this provider does not support TTS.

        Returns:
            `Type[TTSModelBase] | None`:
                The TTS model class, or ``None``.
        """
        return None

    @classmethod
    def list_tts_models(cls) -> list["ModelCard"]:
        """List the candidate TTS models available under this credential.

        Returns:
            `list[ModelCard]`:
                A list of TTS model cards, or empty if TTS is not supported.
        """
        tts_cls = cls.get_tts_model_class()
        if tts_cls is None:
            return []
        return tts_cls.list_models()

    @classmethod
    def list_models(cls) -> list["ModelCard"]:
        """List the candidate chat models that are available under this
        credential. The default implementation delegates to the
        :meth:`ChatModelBase.list_models` of the class returned by
        :meth:`get_chat_model_class`.

        Returns:
            `list[ModelCard]`:
                A list of candidate models described by their model cards.
        """
        return cls.get_chat_model_class().list_models()
