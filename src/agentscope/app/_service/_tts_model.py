# -*- coding: utf-8 -*-
"""TTS model service: builds a TTSModelBase from stored credential + config."""
from typing import Type

from fastapi import HTTPException, status
from pydantic import ValidationError

from ._access import CredentialView, ResourceAccessService
from ..storage import StorageBase, TTSModelConfig
from ..storage._model import (
    ENGINE_TO_CREDENTIAL_TYPE,
    get_missing_voice_profile_binding_fields,
)
from ...credential import CredentialBase, CredentialFactory
from ...tts import TTSModelBase, TTSModelCard


def redact_credential_view(view: CredentialView) -> CredentialView:
    """Remove Remote TTS bearer tokens from a public credential view."""
    if view.data.get("type") != "remote_tts_credential":
        return view
    redacted = view.model_copy(deep=True)
    redacted.data.pop("api_key", None)
    return redacted


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
            Owner-scoped storage used to validate custom voice profiles and
            load their engine-specific synthesis parameters.

    Returns:
        `TTSModelBase`:
            The TTS model instance.
    """
    credential, tts_cls = await _validate_and_resolve_tts_config(
        user_id,
        config,
        access,
        storage,
    )

    params = dict(config.parameters) if config.parameters else {}
    if config.voice_profile_id:
        if storage is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A storage backend is required to use a voice profile.",
            )
        params = await _enrich_from_profile(
            storage,
            user_id,
            config.voice_profile_id,
            params,
        )
    params = {
        key: value for key, value in params.items() if not key.startswith("_")
    }
    parameters = tts_cls.Parameters(**params) if params else None

    return tts_cls(
        credential=credential,
        model=config.model,
        parameters=parameters,
    )


async def validate_tts_model_config(
    user_id: str,
    config: TTSModelConfig | None,
    access: ResourceAccessService,
    storage: StorageBase,
) -> None:
    """Validate a TTS config before it is persisted on a session."""
    if config is None:
        return
    await _validate_and_resolve_tts_config(
        user_id,
        config,
        access,
        storage,
    )


async def _validate_and_resolve_tts_config(
    user_id: str,
    config: TTSModelConfig,
    access: ResourceAccessService,
    storage: StorageBase | None,
) -> tuple[CredentialBase, Type[TTSModelBase]]:
    """Resolve the provider and enforce custom voice ownership."""
    credential_record = await access.resolve_credential(
        user_id,
        config.credential_id,
    )
    credential_type = credential_record.data.get("type")
    if credential_type != config.type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"TTS config type {config.type!r} does not match credential "
                f"type {credential_type!r}."
            ),
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
    card = _find_model_card(tts_cls, config.model)
    _validate_tts_parameters(tts_cls, config)
    await _validate_voice_binding(
        user_id=user_id,
        config=config,
        credential_owner_id=credential_record.user_id,
        storage=storage,
        card=card,
    )
    return credential, tts_cls


def _validate_tts_parameters(
    tts_cls: Type[TTSModelBase],
    config: TTSModelConfig,
) -> None:
    """Validate provider parameters."""
    params = {
        key: value
        for key, value in config.parameters.items()
        if not key.startswith("_")
    }
    try:
        tts_cls.Parameters(**params)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters for TTS model {config.model!r}: {e}",
        ) from e


async def _validate_voice_binding(
    *,
    user_id: str,
    config: TTSModelConfig,
    credential_owner_id: str,
    storage: StorageBase | None,
    card: TTSModelCard | None,
) -> None:
    """Validate preset voices or an exact owner-scoped custom voice profile.

    Profile CRUD performs the write-time counterpart in
    ``_router._voice_profile._validate_voice_profile_data``. Runtime
    validation remains here as defense against stale or externally persisted
    data, while both layers share the required binding-field definition.
    """
    voice = config.parameters.get("voice")
    preset_voices = _get_preset_voices(card)
    if config.voice_profile_id is None:
        if (
            config.type == "local_tts_credential"
            and config.model == "chatterbox"
            and config.parameters.get("variant", "turbo")
            in {"turbo", "multilingual"}
        ):
            # Chatterbox is a hybrid model: Turbo (the default) and
            # Multilingual can use their built-in voice, while English is
            # clone-only and continues through the profile requirement below.
            return
        if config.type == "remote_tts_credential":
            # Phase 1 has no remote voice-discovery endpoint. A free-form
            # provider voice id is therefore valid and must not be mistaken
            # for an AgentScope-owned cloned voice. Reference-audio cloning
            # continues to require an owner-scoped Voice Profile.
            return
        if _requires_voice_profile(card, preset_voices):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"TTS model {config.model!r} requires a cloned voice "
                    "profile."
                ),
            )
        if voice is None:
            return
        if isinstance(voice, str) and voice in preset_voices:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=("A non-preset voice requires an owned voice_profile_id."),
        )

    if storage is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A storage backend is required to use a voice profile.",
        )
    profile = await storage.get_voice_profile(
        user_id,
        config.voice_profile_id,
    )
    if profile is None:
        # Owner-scoped lookup intentionally hides whether another tenant owns
        # the supplied profile id.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice profile {config.voice_profile_id!r} not found.",
        )

    data = profile.data
    missing = get_missing_voice_profile_binding_fields(data)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Voice profile {profile.id!r} is incomplete: "
                + ", ".join(missing)
                + "."
            ),
        )

    assert data.engine is not None
    assert data.credential_id is not None
    if credential_owner_id != user_id and data.engine != "remote_tts":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Custom voices cannot use a shared credential.",
        )

    expected_type = ENGINE_TO_CREDENTIAL_TYPE[data.engine]
    bindings = {
        "credential_id": (config.credential_id, data.credential_id),
        "provider type": (config.type, expected_type),
        "model": (config.model, data.model),
        "voice": (voice, data.voice),
    }
    mismatches = [
        name
        for name, (actual, expected) in bindings.items()
        if actual != expected
    ]
    if mismatches:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"TTS config does not match voice profile {profile.id!r}: "
                + ", ".join(mismatches)
                + "."
            ),
        )


def _find_model_card(
    tts_cls: Type[TTSModelBase],
    model: str,
) -> TTSModelCard | None:
    """Return the exact model card used to classify preset voices."""
    return next(
        (card for card in tts_cls.list_models() if card.name == model),
        None,
    )


def _get_preset_voices(card: TTSModelCard | None) -> set[str]:
    """Return the provider-declared preset voices for a model card."""
    if card is None:
        return set()
    properties = card.parameter_schema.get("properties", {})
    voice_schema = properties.get("voice", {})
    voices = voice_schema.get("enum", [])
    if not isinstance(voices, list):
        return set()
    return {voice for voice in voices if isinstance(voice, str)}


def _requires_voice_profile(
    card: TTSModelCard | None,
    preset_voices: set[str] | None = None,
) -> bool:
    """Return whether a model can only synthesize with a cloned voice.

    A voice-cloning model with no provider-declared preset voices is treated
    as clone-only. In particular, omitting ``voice`` must not fall through to
    a shared parameter class default that is invalid for that model.
    """
    if card is None or not card.voice_cloning:
        return False
    if preset_voices is None:
        preset_voices = _get_preset_voices(card)
    return not preset_voices


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
            "reference_audio_media_type",
            "reference_text",
            "language",
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
