# -*- coding: utf-8 -*-
"""TTS model service: builds a TTSModelBase from stored credential + config."""
import asyncio
from functools import partial
from time import monotonic
from typing import Type

from fastapi import HTTPException, status
from pydantic import ValidationError

from ._access import CredentialView, ResourceAccessService
from ..storage import CredentialRecord, StorageBase, TTSModelConfig
from ..storage._model import (
    ENGINE_TO_CREDENTIAL_TYPE,
    get_missing_voice_profile_binding_fields,
)
from ...credential import CredentialBase, CredentialFactory
from ...tts import TTSModelBase, TTSModelCard

_DISCOVERY_CACHE_TTL_SECONDS = 60.0
_DISCOVERY_CACHE_MAX_ENTRIES = 256
_DiscoveryCacheKey = tuple[str, str, str]
_discovery_cache: dict[
    _DiscoveryCacheKey,
    tuple[float, list[TTSModelCard]],
] = {}
_discovery_inflight: dict[
    _DiscoveryCacheKey,
    asyncio.Task[list[TTSModelCard]],
] = {}


async def discover_tts_models(
    credential_record: CredentialRecord,
    credential: CredentialBase | None = None,
) -> list[TTSModelCard]:
    """Discover model cards with credential-scoped short-lived caching.

    The cache key contains the credential owner, ID, and ``updated_at``
    revision. Shared callers can therefore reuse public capability metadata
    without mixing different credentials or retaining results after an edit.
    """
    credential = credential or CredentialFactory.from_dict(
        credential_record.data,
    )
    if credential.type != "remote_tts_credential":
        return await _discover_tts_models_uncached(credential)

    cache_key = (
        credential_record.user_id,
        credential_record.id,
        credential_record.updated_at.isoformat(),
    )
    now = monotonic()
    _prune_discovery_cache(now, cache_key)
    cached = _discovery_cache.get(cache_key)
    if cached is not None and cached[0] > now:
        return cached[1]

    task = _discovery_inflight.get(cache_key)
    if task is None:
        task = asyncio.create_task(
            _discover_tts_models_uncached(credential),
        )
        _discovery_inflight[cache_key] = task
        task.add_done_callback(
            partial(_cache_discovery_result, cache_key),
        )
    try:
        models = await asyncio.shield(task)
    finally:
        if task.done() and _discovery_inflight.get(cache_key) is task:
            _discovery_inflight.pop(cache_key, None)

    _discovery_cache[cache_key] = (
        monotonic() + _DISCOVERY_CACHE_TTL_SECONDS,
        models,
    )
    return models


def _cache_discovery_result(
    cache_key: _DiscoveryCacheKey,
    task: asyncio.Task[list[TTSModelCard]],
) -> None:
    """Store a completed discovery even if its HTTP caller disconnected."""
    if _discovery_inflight.get(cache_key) is task:
        _discovery_inflight.pop(cache_key, None)
    if task.cancelled():
        return
    try:
        models = task.result()
    except Exception:  # pylint: disable=broad-exception-caught
        return
    now = monotonic()
    _prune_discovery_cache(now, cache_key)
    _discovery_cache[cache_key] = (
        now + _DISCOVERY_CACHE_TTL_SECONDS,
        models,
    )


async def _discover_tts_models_uncached(
    credential: CredentialBase,
) -> list[TTSModelCard]:
    """Run provider discovery without caching."""
    models: list[TTSModelCard] = []
    for tts_cls in credential.get_tts_model_classes():
        models.extend(await tts_cls.discover_models(credential))
    return models


def _prune_discovery_cache(
    now: float,
    current_key: _DiscoveryCacheKey,
) -> None:
    """Remove expired and obsolete credential revisions from the cache."""
    owner_id, credential_id, _ = current_key
    for key, (expires_at, _) in list(_discovery_cache.items()):
        same_credential = key[:2] == (owner_id, credential_id)
        if expires_at <= now or (same_credential and key != current_key):
            _discovery_cache.pop(key, None)
    if len(_discovery_cache) >= _DISCOVERY_CACHE_MAX_ENTRIES:
        oldest_key = min(
            _discovery_cache,
            key=lambda key: _discovery_cache[key][0],
        )
        _discovery_cache.pop(oldest_key, None)


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
        allow_incomplete_remote_model=True,
    )


async def _validate_and_resolve_tts_config(
    user_id: str,
    config: TTSModelConfig,
    access: ResourceAccessService,
    storage: StorageBase | None,
    *,
    allow_incomplete_remote_model: bool = False,
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
    remote_model = config.parameters.get("model")
    normalized_remote_model = (
        remote_model.strip() if isinstance(remote_model, str) else ""
    )
    if (
        credential.type == "remote_tts_credential"
        and config.model == "remote-tts"
        and not allow_incomplete_remote_model
        and normalized_remote_model in {"", "remote-tts"}
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Remote TTS requires a concrete model ID. Enter the model "
                "served by the endpoint in the TTS parameters."
            ),
        )
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
    if credential.type == "remote_tts_credential":
        discovered_cards = await discover_tts_models(
            credential_record,
            credential,
        )
        card_name = (
            normalized_remote_model
            if config.model == "remote-tts" and normalized_remote_model
            else config.model
        )
        card = next(
            (
                candidate
                for candidate in discovered_cards
                if candidate.name == card_name
            ),
            next(
                (
                    candidate
                    for candidate in discovered_cards
                    if candidate.name == config.model
                ),
                None,
            ),
        )
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
            card is not None
            and card.reference_audio_required
            and not _has_reference_audio(config.parameters)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"TTS model {config.model!r} requires reference audio."
                ),
            )
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
            # Remote endpoints may expose discovered presets or accept a
            # provider-specific free-form voice ID. Neither should be treated
            # as an AgentScope-owned cloned voice.
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
    if (
        card is not None
        and card.reference_audio_required
        and not _has_reference_audio(config.parameters)
        and not _has_reference_audio(data.metadata or {})
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Voice profile {profile.id!r} requires reference audio "
                f"for TTS model {config.model!r}."
            ),
        )


def _has_reference_audio(values: dict) -> bool:
    """Return whether a parameter or metadata mapping contains audio."""
    for key in ("reference_audio_base64", "reference_audio_path"):
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


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
