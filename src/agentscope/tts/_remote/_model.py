# -*- coding: utf-8 -*-
"""OpenAI-compatible remote TTS model implementation."""
import base64
import json
import uuid
from typing import Any, AsyncGenerator, Literal

import httpx
from pydantic import BaseModel, Field

from .._tts_base import TTSModelBase
from .._tts_response import TTSResponse
from ...credential import RemoteTTSCredential
from ...message import Base64Source, DataBlock


_DEFAULT_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
    "opus": "audio/opus",
    "aac": "audio/aac",
}


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

        voice: str = Field(
            default="default",
            title="Voice",
            description="Preset voice or remote voice identifier.",
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
        )

        speed: float = Field(
            default=1.0,
            ge=0.25,
            le=4.0,
            title="Speed",
        )

        language: str | None = Field(
            default=None,
            title="Language",
        )

        instructions: str | None = Field(
            default=None,
            title="Instructions",
        )

        reference_audio_base64: str | None = Field(
            default=None,
            title="Reference Audio",
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
        )

        reference_text: str | None = Field(
            default=None,
            title="Reference Text",
        )

    type: Literal["remote_tts"] = "remote_tts"
    """The type of the TTS model."""

    realtime: bool = False

    def __init__(
        self,
        credential: RemoteTTSCredential,
        model: str = "remote-tts",
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
        remote_model = (
            self.credential.served_model
            if self.model == "remote-tts" and self.credential.served_model
            else self.model
        )
        payload: dict[str, Any] = {
            "model": remote_model,
            "input": text,
            "voice": self.parameters.voice,
            "response_format": self.parameters.response_format,
            "speed": self.parameters.speed,
        }
        if self.parameters.language:
            payload["language"] = self.parameters.language
        if self.parameters.instructions:
            payload["instructions"] = self.parameters.instructions
        if self.parameters.reference_audio_base64:
            payload["ref_audio"] = (
                f"data:{self.parameters.reference_audio_media_type};base64,"
                f"{self.parameters.reference_audio_base64}"
            )
        if self.parameters.reference_text:
            payload["ref_text"] = self.parameters.reference_text
        return payload

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
