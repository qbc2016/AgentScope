# -*- coding: utf-8 -*-
"""TTS model service: builds a TTSModelBase from stored credential + config."""
from fastapi import HTTPException, status

from ..storage import StorageBase, TTSModelConfig
from ...credential import CredentialFactory
from ...tts import TTSModelBase


async def get_tts_model(
    user_id: str,
    config: TTSModelConfig,
    storage: StorageBase,
) -> TTSModelBase:
    """Get the TTS model instance from the configuration and storage.

    Args:
        user_id (`str`):
            The user id.
        config (`TTSModelConfig`):
            The TTS model configuration.
        storage (`StorageBase`):
            The storage instance.

    Returns:
        `TTSModelBase`:
            The TTS model instance.
    """
    credential_record = await storage.get_credential(
        user_id,
        config.credential_id,
    )
    if credential_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Credential {config.credential_id!r} not found.",
        )

    credential = CredentialFactory.from_dict(credential_record.data)
    tts_cls = credential.get_tts_model_class()
    if tts_cls is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider {config.type!r} does not support TTS models.",
        )

    parameters = (
        tts_cls.Parameters(**config.parameters) if config.parameters else None
    )
    return tts_cls(
        credential=credential,
        model=config.model,
        parameters=parameters,
    )
