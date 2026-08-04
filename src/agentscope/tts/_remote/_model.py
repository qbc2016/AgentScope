# -*- coding: utf-8 -*-
"""OpenAI-compatible remote TTS model implementation."""
import asyncio
import base64
import copy
import json
import uuid
from typing import Any, AsyncGenerator, Literal

import httpx
from pydantic import BaseModel, Field

from .._tts_base import TTSModelBase
from .._tts_model_card import TTSModelCard
from .._tts_response import TTSResponse
from ...credential import CredentialBase, RemoteTTSCredential
from ...message import Base64Source, DataBlock


_DEFAULT_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
    "opus": "audio/opus",
    "aac": "audio/aac",
}
_MAX_DISCOVERED_MODELS = 200
_MAX_DISCOVERED_VOICES = 500
_MAX_VOICE_DISCOVERY_MODELS = 32
_MAX_CONCURRENT_VOICE_REQUESTS = 8
_DISCOVERY_DEADLINE_SECONDS = 15.0
_DISCOVERY_REQUEST_TIMEOUT_SECONDS = 10.0
_DISCOVERY_UNSUPPORTED_STATUS = {404, 405, 501}


class RemoteTTSError(RuntimeError):
    """A safe, actionable error returned by a remote TTS endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        status_code: int | None,
        request_id: str | None,
        remote_message: str,
    ) -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        self.request_id = request_id
        self.remote_message = remote_message
        super().__init__(
            "Remote TTS request failed: "
            f"endpoint={endpoint!r}, status={status_code!r}, "
            f"request_id={request_id!r}, message={remote_message!r}.",
        )


class RemoteTTSModel(TTSModelBase):
    """Adapter for a binary ``POST /v1/audio/speech`` endpoint."""

    class Parameters(BaseModel):
        """Frontend-exposed remote synthesis parameters."""

        model: str | None = Field(
            default=None,
            title="Model",
            description=(
                "Model identifier understood by the remote TTS service. "
                "Required when using the generic Remote TTS option."
            ),
        )

        voice: str | None = Field(
            default=None,
            title="Voice",
            description=(
                "Preset voice or remote voice identifier. Required unless "
                "the selected remote model does not use a voice."
            ),
        )

        response_format: Literal[
            "wav",
            "mp3",
            "flac",
            "pcm",
            "opus",
            "aac",
        ] = Field(
            default="wav",
            title="Response Format",
            description=(
                "Requested audio encoding. The selected remote model must "
                "support this format."
            ),
        )

        speed: float = Field(
            default=1.0,
            ge=0.25,
            le=4.0,
            title="Speed",
            description=(
                "Speech speed multiplier passed to the remote service."
            ),
        )

        language: str | None = Field(
            default=None,
            title="Language",
            description=(
                "Optional language code or name understood by the remote "
                "service."
            ),
        )

        instructions: str | None = Field(
            default=None,
            title="Instructions",
            description=(
                "Optional synthesis instructions. Support depends on the "
                "selected remote model."
            ),
        )

        task_type: str | None = Field(
            default=None,
            title="Task Type",
            description=(
                "Optional remote task type, for example CustomVoice, "
                "VoiceDesign, or Base."
            ),
        )

        reference_audio_base64: str | None = Field(
            default=None,
            title="Reference Audio",
            description=(
                "Base64-encoded reference audio for voice cloning or voice "
                "conditioning."
            ),
        )

        reference_audio_media_type: Literal[
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
        ] = Field(
            default="audio/wav",
            title="Reference Audio Media Type",
            description="MIME media type of the reference audio.",
        )

        reference_text: str | None = Field(
            default=None,
            title="Reference Text",
            description=(
                "Optional transcript of the reference audio, required by "
                "some voice-cloning models."
            ),
        )

    type: Literal["remote_tts"] = "remote_tts"
    """The type of the TTS model."""

    realtime: bool = False

    def __init__(
        self,
        credential: RemoteTTSCredential,
        model: str,
        parameters: "RemoteTTSModel.Parameters | None" = None,
        stream: bool = False,
    ) -> None:
        """Initialize the remote adapter."""
        super().__init__(
            credential=credential,
            model=model,
            parameters=parameters,
            stream=stream,
        )
        self._base_url = credential.base_url.rstrip("/")
        self._speech_url = f"{self._base_url}/audio/speech"

    def _build_payload(self, text: str) -> dict[str, Any]:
        """Build an explicit allowlist of remote request fields."""
        remote_model = (self.parameters.model or self.model).strip()
        if remote_model == "remote-tts" or not remote_model:
            raise ValueError(
                "Remote TTS requires a concrete model ID. Configure the "
                "Model field in the TTS parameters.",
            )
        voice = (
            self.parameters.voice.strip()
            if isinstance(self.parameters.voice, str)
            else ""
        )
        voice_optional = bool(self.parameters.reference_audio_base64) or (
            self.parameters.task_type == "VoiceDesign"
        )
        if not voice and not voice_optional:
            raise ValueError(
                "Remote TTS requires a concrete voice ID. Select a voice "
                "discovered from the endpoint or enter one in the TTS "
                "parameters.",
            )
        payload: dict[str, Any] = {
            "model": remote_model,
            "input": text,
            "response_format": self.parameters.response_format,
            "speed": self.parameters.speed,
        }
        if voice:
            payload["voice"] = voice
        if self.parameters.language:
            payload["language"] = self.parameters.language
        if self.parameters.instructions:
            payload["instructions"] = self.parameters.instructions
        if self.parameters.task_type:
            payload["task_type"] = self.parameters.task_type
        if self.parameters.reference_audio_base64:
            payload["ref_audio"] = (
                f"data:{self.parameters.reference_audio_media_type};base64,"
                f"{self.parameters.reference_audio_base64}"
            )
        if self.parameters.reference_text:
            payload["ref_text"] = self.parameters.reference_text
        return payload

    @classmethod
    async def discover_models(
        cls,
        credential: CredentialBase,
    ) -> list[TTSModelCard]:
        """Return endpoint-specific model cards when discovery is available.

        Discovery is an optional convenience. Any network, authorization,
        protocol, or compatibility failure falls back to the generic card so
        callers can still enter a concrete model ID manually.
        """
        generic_cards = cls.list_models()
        if not generic_cards:
            return []
        generic = generic_cards[0]
        if not isinstance(credential, RemoteTTSCredential):
            return [generic]
        headers: dict[str, str] = {}
        if credential.api_key is not None:
            token = credential.api_key.get_secret_value()
            headers["Authorization"] = f"Bearer {token}"

        cards: list[TTSModelCard] = []
        try:
            async with asyncio.timeout(_DISCOVERY_DEADLINE_SECONDS):
                async with httpx.AsyncClient(
                    timeout=min(
                        credential.timeout,
                        _DISCOVERY_REQUEST_TIMEOUT_SECONDS,
                    ),
                    follow_redirects=False,
                ) as client:
                    models = await cls._discover_model_items(
                        client,
                        credential.base_url.rstrip("/"),
                        headers,
                    )
                    if not models:
                        return [generic]

                    voice_candidates: list[tuple[TTSModelCard, str]] = []
                    for item in models[:_MAX_DISCOVERED_MODELS]:
                        card = cls._model_card_from_remote(
                            generic,
                            item,
                        )
                        if card is None or card.name == generic.name:
                            continue
                        cards.append(card)
                        if (
                            "voices" not in item
                            and not card.reference_audio_required
                            and len(voice_candidates)
                            < _MAX_VOICE_DISCOVERY_MODELS
                        ):
                            voice_candidates.append((card, card.name))

                    semaphore = asyncio.Semaphore(
                        _MAX_CONCURRENT_VOICE_REQUESTS,
                    )
                    async with asyncio.TaskGroup() as task_group:
                        for card, model_id in voice_candidates:
                            task_group.create_task(
                                cls._enrich_card_voices(
                                    client,
                                    credential.base_url.rstrip("/"),
                                    headers,
                                    card,
                                    model_id,
                                    semaphore,
                                ),
                            )
        except TimeoutError:
            # Cards are built before optional voice enrichment. A slow voice
            # endpoint therefore degrades to free-form voice input instead of
            # discarding already discovered model IDs.
            return [*cards, generic]
        except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError):
            return [*cards, generic] if cards else [generic]
        return [*cards, generic]

    @classmethod
    async def _enrich_card_voices(
        cls,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict[str, str],
        card: TTSModelCard,
        model_id: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Add optional voice choices without blocking other model cards."""
        async with semaphore:
            voices = await cls._discover_voice_ids(
                client,
                base_url,
                headers,
                model_id,
            )
        if not voices:
            return
        properties = card.parameter_schema.get("properties", {})
        voice_property = properties.get("voice")
        if isinstance(voice_property, dict):
            voice_property["enum"] = voices
            voice_property["default"] = voices[0]

    @classmethod
    async def _discover_model_items(
        cls,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Query capability discovery, then fall back to OpenAI models."""
        for path in ("/audio/models", "/models"):
            try:
                response = await client.get(
                    f"{base_url}{path}",
                    headers=headers,
                )
                if response.status_code in _DISCOVERY_UNSUPPORTED_STATUS:
                    continue
                if not response.is_success:
                    continue
                items = cls._extract_items(response.json())
                normalized = [
                    item
                    for item in items
                    if isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                    and item["id"].strip()
                ]
                if normalized:
                    return normalized
            except (
                httpx.HTTPError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ):
                # Malformed or unsupported capability discovery must not
                # prevent the standard /models fallback.
                continue
        return []

    @staticmethod
    def _extract_items(payload: Any) -> list[Any]:
        """Extract a list from common OpenAI and capability response shapes."""
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        for key in ("data", "models", "voices"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []

    @classmethod
    def _model_card_from_remote(
        cls,
        generic: TTSModelCard,
        item: dict[str, Any],
    ) -> TTSModelCard | None:
        """Convert remote capability metadata into an AgentScope model card."""
        model_id = item["id"].strip()
        if not model_id:
            return None

        schema = copy.deepcopy(generic.parameter_schema)
        properties = schema.setdefault("properties", {})
        # A discovered model card already identifies the concrete provider
        # model. Only the generic Remote TTS card needs a manual Model field.
        properties.pop("model", None)

        formats = cls._string_list(item.get("response_formats"))
        if formats:
            supported_formats = [
                value for value in formats if value in _DEFAULT_MEDIA_TYPES
            ]
            if supported_formats and "response_format" in properties:
                properties["response_format"]["enum"] = supported_formats
                properties["response_format"]["default"] = supported_formats[0]
        else:
            supported_formats = ["wav"]

        languages = cls._string_list(item.get("languages"))
        if languages and "language" in properties:
            properties["language"]["enum"] = languages

        task_types = cls._string_list(item.get("task_types"))
        if task_types and "task_type" in properties:
            properties["task_type"]["enum"] = task_types

        voices = cls._voice_ids(item.get("voices"))
        reference_required = item.get("reference_audio_required") is True
        if voices and "voice" in properties:
            properties["voice"]["enum"] = voices
            properties["voice"]["default"] = voices[0]
        elif reference_required:
            properties.pop("voice", None)

        output_types = [
            _DEFAULT_MEDIA_TYPES[value]
            for value in supported_formats
            if value in _DEFAULT_MEDIA_TYPES
        ]
        return TTSModelCard(
            name=model_id,
            label=(
                item.get("label")
                if isinstance(item.get("label"), str)
                else model_id
            ),
            status="active",
            input_types=["text/plain"],
            output_types=output_types or ["audio/wav"],
            # ``TTSModelCard.realtime`` means streaming *input*. A remote
            # endpoint advertising streaming output does not change the
            # adapter's input lifecycle.
            realtime=False,
            voice_cloning=item.get("voice_cloning") is True,
            reference_audio_required=reference_required,
            parameter_schema=schema,
            parameters_overrides=generic.parameters_overrides,
        )

    @classmethod
    async def _discover_voice_ids(
        cls,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict[str, str],
        model_id: str,
    ) -> list[str]:
        """Discover optional voice choices for one model."""
        try:
            response = await client.get(
                f"{base_url}/audio/voices",
                headers=headers,
                params={"model": model_id},
            )
            if not response.is_success:
                return []
            return cls._voice_ids(cls._extract_items(response.json()))
        except (
            httpx.HTTPError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            return []

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        """Return a bounded, de-duplicated list of non-empty strings."""
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                normalized = item.strip()
                if normalized not in result:
                    result.append(normalized)
        return result[:_MAX_DISCOVERED_VOICES]

    @classmethod
    def _voice_ids(cls, value: Any) -> list[str]:
        """Extract voice IDs from strings or voice metadata objects."""
        if not isinstance(value, list):
            return []
        raw_ids = [
            item
            if isinstance(item, str)
            else item.get("id")
            if isinstance(item, dict)
            else None
            for item in value
        ]
        return cls._string_list(raw_ids)

    def _headers(
        self,
        request_id: str,
        idempotency_key: str,
    ) -> dict[str, str]:
        """Build request headers without exposing the API key elsewhere."""
        headers = {
            "X-Request-ID": request_id,
            "Idempotency-Key": idempotency_key,
        }
        if self.credential.api_key is not None:
            token = self.credential.api_key.get_secret_value()
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _redact_secret(self, message: str) -> str:
        """Remove the configured token if a remote server echoes it."""
        if self.credential.api_key is None:
            return message
        token = self.credential.api_key.get_secret_value()
        return message.replace(token, "***") if token else message

    @staticmethod
    def _remote_error_message(response: httpx.Response) -> str:
        """Extract an OpenAI-style error message, falling back to text."""
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            return response.text[:1000] or response.reason_phrase

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and isinstance(
                error.get("message"),
                str,
            ):
                return error["message"]
            if isinstance(error, str):
                return error
            if isinstance(body.get("detail"), str):
                return body["detail"]
        return response.text[:1000] or response.reason_phrase

    async def synthesize(
        self,
        text: str | None = None,
        **kwargs: Any,
    ) -> TTSResponse | AsyncGenerator[TTSResponse, None]:
        """Synthesize text and return the endpoint's binary audio."""
        if not text:
            return TTSResponse(content=None)

        request_id = f"req_{uuid.uuid4().hex}"
        idempotency_key = f"tts_{uuid.uuid4().hex}"
        headers = self._headers(request_id, idempotency_key)
        payload = self._build_payload(text)

        try:
            async with httpx.AsyncClient(
                timeout=self.credential.timeout,
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    self._speech_url,
                    json=payload,
                    headers=headers,
                )
        except httpx.HTTPError as error:
            raise RemoteTTSError(
                endpoint=self._speech_url,
                status_code=None,
                request_id=request_id,
                remote_message=self._redact_secret(str(error)),
            ) from error

        remote_request_id = response.headers.get(
            "x-request-id",
            request_id,
        )
        if not response.is_success:
            raise RemoteTTSError(
                endpoint=self._speech_url,
                status_code=response.status_code,
                request_id=remote_request_id,
                remote_message=self._redact_secret(
                    self._remote_error_message(response),
                ),
            )

        if not response.content:
            raise RemoteTTSError(
                endpoint=self._speech_url,
                status_code=response.status_code,
                request_id=remote_request_id,
                remote_message="The remote service returned no audio data",
            )
        media_type = response.headers.get(
            "content-type",
            _DEFAULT_MEDIA_TYPES[self.parameters.response_format],
        ).strip()
        normalized_media_type = media_type.split(";", 1)[0].strip().lower()
        if normalized_media_type == "application/octet-stream":
            media_type = _DEFAULT_MEDIA_TYPES[self.parameters.response_format]
        elif not normalized_media_type.startswith("audio/"):
            raise RemoteTTSError(
                endpoint=self._speech_url,
                status_code=response.status_code,
                request_id=remote_request_id,
                remote_message=(
                    "Expected a binary audio response, received "
                    f"Content-Type {media_type!r}"
                ),
            )
        audio_base64 = base64.b64encode(response.content).decode("ascii")
        return TTSResponse(
            content=DataBlock(
                source=Base64Source(
                    data=audio_base64,
                    media_type=media_type,
                ),
            ),
            metadata={
                "request_id": remote_request_id,
                "idempotency_key": idempotency_key,
            },
            is_last=True,
        )
