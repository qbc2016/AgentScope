# -*- coding: utf-8 -*-
"""TTS model service: builds a TTSModelBase from stored credential + config."""
from typing import Type

from fastapi import HTTPException, status

from ._access import ResourceAccessService
from ..storage import StorageBase, TTSModelConfig
from ...credential import CredentialFactory
from ...tts import TTSModelBase


async def get_tts_model(
    user_id: str,
    config: TTSModelConfig,
    access: ResourceAccessService,
    storage: StorageBase | None = None,
) -> TTSModelBase:
    """Build a TTS model instance from a stored credential and config.

    Args:
        user_id (`str`):
            The viewer's user id (may differ from the credential owner
            for shared credentials).
        config (`TTSModelConfig`):
            The TTS model configuration.
        access (`ResourceAccessService`):
            Injected resource access service.
        storage (`StorageBase | None`, defaults to `None`):
            Storage backend for loading voice profile metadata.

    Returns:
        `TTSModelBase`:
            The TTS model instance.
    """
    credential_record = await access.resolve_credential(
        user_id,
        config.credential_id,
    )

    credential = CredentialFactory.from_dict(credential_record.data)
    tts_classes = credential.get_tts_model_classes()
    if not tts_classes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Credential {credential_record.data.get('type', '?')!r}"
                f" does not support TTS models."
            ),
        )

    tts_cls = _resolve_tts_class(
        tts_classes,
        config.model,
        allow_single_fallback=(credential.type != "local_tts_credential"),
    )
    params = dict(config.parameters) if config.parameters else {}

    # Load reference audio from voice profile metadata
    profile_id = params.pop("_voice_profile_id", None)
    if profile_id and storage is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A storage backend is required to use a voice profile.",
        )
    if profile_id:
        assert storage is not None
        params = await _enrich_from_profile(
            storage,
            user_id,
            profile_id,
            params,
        )

    # Strip remaining internal keys
    params = {k: v for k, v in params.items() if not k.startswith("_")}
    parameters = tts_cls.Parameters(**params) if params else None
    return tts_cls(
        credential=credential,
        model=config.model,
        parameters=parameters,
    )


async def _enrich_from_profile(
    storage: StorageBase,
    user_id: str,
    profile_id: str,
    params: dict,
) -> dict:
    """Merge voice profile metadata into TTS params.

    Reads the voice profile from storage and copies
    engine-specific synthesis parameters from its metadata into
    ``params``.

    Local TTS models accept ``reference_audio_base64`` directly
    in their Parameters and handle decoding internally during
    synthesis. The audio is stored as base64 (not as a file
    path) so that multi-node deployments can access it from
    any machine without shared filesystem.

    Values explicitly set in the TTS config parameters take
    precedence over the profile metadata.
    """
    profile = await storage.get_voice_profile(user_id, profile_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice profile {profile_id!r} not found.",
        )
    if profile.data.metadata:
        meta = profile.data.metadata
        for key in (
            "reference_audio_base64",
            "reference_text",
            "lang_code",
            "speed",
        ):
            if key in meta and meta[key] is not None:
                params.setdefault(key, meta[key])
    return params


def _resolve_tts_class(
    classes: list[Type[TTSModelBase]],
    model: str,
    allow_single_fallback: bool = True,
) -> Type[TTSModelBase]:
    """Pick the TTS class that lists the given model name.

    When allowed, falls back to the single class for credentials
    exposing only one TTS class. Otherwise raises 400 when no class
    lists the model, to avoid silently using the wrong engine.
    """
    for cls in classes:
        if any(card.name == model for card in cls.list_models()):
            return cls
    if len(classes) == 1 and allow_single_fallback:
        return classes[0]
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unknown TTS model {model!r} for this credential.",
    )
