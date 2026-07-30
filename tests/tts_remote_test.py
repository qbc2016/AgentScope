# -*- coding: utf-8 -*-
"""Required protocol tests for the Phase 1 Remote TTS adapter."""
import base64
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from pydantic import ValidationError

from agentscope.app._service import (
    CredentialView,
    get_tts_model,
    redact_credential_view,
)
from agentscope.app.storage import CredentialRecord, TTSModelConfig
from agentscope.credential import CredentialFactory, RemoteTTSCredential
from agentscope.tts import RemoteTTSError, RemoteTTSModel, TTSResponse


def _dump_tts_response(response: TTSResponse) -> dict:
    """Dump a response without generated identifiers."""
    dumped = {
        key: value
        for key, value in dict(response).items()
        if key not in {"id", "created_at"}
    }
    if response.content is not None:
        dumped["content"] = response.content.model_dump(exclude={"id"})
    return dumped


class TestRemoteTTSModel(IsolatedAsyncioTestCase):
    """Remote request/response protocol tests."""

    @staticmethod
    def _mock_client(response: httpx.Response) -> AsyncMock:
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = AsyncMock(return_value=response)
        return client

    async def test_accepts_provider_voice_id_without_profile(self) -> None:
        """Phase 1 uses free-form voice IDs until discovery is available."""
        access = AsyncMock()
        access.resolve_credential.return_value = CredentialRecord(
            id="credential-a",
            user_id="user-a",
            data={
                "type": "remote_tts_credential",
                "base_url": "https://tts.example/v1",
            },
        )
        model = cast(
            RemoteTTSModel,
            await get_tts_model(
                "user-a",
                TTSModelConfig(
                    type="remote_tts_credential",
                    credential_id="credential-a",
                    model="remote-tts",
                    parameters={"voice": "zf_xiaobei"},
                ),
                access,
            ),
        )
        card = RemoteTTSCredential.list_tts_models()[0]

        self.assertEqual(
            {
                "parameters": model.parameters.model_dump(),
                "voice_schema": card.parameter_schema["properties"]["voice"],
            },
            {
                "parameters": {
                    "voice": "zf_xiaobei",
                    "response_format": "wav",
                    "speed": 1.0,
                    "language": None,
                    "instructions": None,
                    "reference_audio_base64": None,
                    "reference_audio_media_type": "audio/wav",
                    "reference_text": None,
                },
                "voice_schema": {
                    "default": "default",
                    "description": (
                        "Voice identifier understood by the remote service. "
                        "Phase 1 does not discover remote voices, so enter "
                        "the provider's voice ID manually."
                    ),
                    "title": "Voice",
                    "type": "string",
                },
            },
        )

    async def test_maps_request_and_binary_response(self) -> None:
        """Standard fields, reference audio, headers, and media type map."""
        credential = RemoteTTSCredential(
            base_url="http://tts.example/v1/",
            api_key="secret-token",
            served_model="tada",
        )
        parameters = RemoteTTSModel.Parameters(
            voice="speaker-a",
            response_format="mp3",
            speed=1.2,
            language="zh",
            instructions="calm",
            reference_audio_base64="QUJD",
            reference_audio_media_type="audio/ogg",
            reference_text="reference",
        )
        model = RemoteTTSModel(
            credential=credential,
            parameters=parameters,
        )
        response = httpx.Response(
            200,
            content=b"MP3-DATA",
            headers={
                "Content-Type": "audio/mpeg",
                "X-Request-ID": "remote-request",
            },
        )
        client = self._mock_client(response)
        uuid_mock = MagicMock()
        uuid_mock.uuid4.side_effect = [
            SimpleNamespace(hex="request"),
            SimpleNamespace(hex="idempotency"),
        ]

        with (
            patch(
                "agentscope.tts._remote._model.httpx.AsyncClient",
                return_value=client,
            ),
            patch(
                "agentscope.tts._remote._model.uuid",
                uuid_mock,
            ),
        ):
            result = cast(TTSResponse, await model.synthesize("你好"))

        call = client.post.await_args
        self.assertEqual(
            {
                "response": _dump_tts_response(result),
                "request": {
                    "args": call.args,
                    "kwargs": call.kwargs,
                },
            },
            {
                "response": {
                    "content": {
                        "type": "data",
                        "source": {
                            "type": "base64",
                            "data": base64.b64encode(b"MP3-DATA").decode(),
                            "media_type": "audio/mpeg",
                        },
                        "name": None,
                    },
                    "type": "tts",
                    "usage": None,
                    "metadata": {
                        "request_id": "remote-request",
                        "idempotency_key": "tts_idempotency",
                    },
                    "is_last": True,
                },
                "request": {
                    "args": ("http://tts.example/v1/audio/speech",),
                    "kwargs": {
                        "json": {
                            "model": "tada",
                            "input": "你好",
                            "voice": "speaker-a",
                            "response_format": "mp3",
                            "speed": 1.2,
                            "language": "zh",
                            "instructions": "calm",
                            "ref_audio": "data:audio/ogg;base64,QUJD",
                            "ref_text": "reference",
                        },
                        "headers": {
                            "X-Request-ID": "req_request",
                            "Idempotency-Key": "tts_idempotency",
                            "Authorization": "Bearer secret-token",
                        },
                    },
                },
            },
        )

    async def test_remote_error_is_actionable_and_redacted(self) -> None:
        """Status, request id, and message survive without leaking API key."""
        credential = RemoteTTSCredential(
            base_url="https://tts.example/v1",
            api_key="secret-token",
        )
        model = RemoteTTSModel(credential=credential)
        response = httpx.Response(
            400,
            json={
                "error": {
                    "message": "bad ref_audio; token=secret-token",
                },
            },
            headers={"X-Request-ID": "remote-error"},
        )
        client = self._mock_client(response)

        with (
            patch(
                "agentscope.tts._remote._model.httpx.AsyncClient",
                return_value=client,
            ),
            self.assertRaises(RemoteTTSError) as context,
        ):
            await model.synthesize("hello")

        error = context.exception
        self.assertEqual(
            {
                "endpoint": error.endpoint,
                "status_code": error.status_code,
                "request_id": error.request_id,
                "remote_message": error.remote_message,
                "message": str(error),
            },
            {
                "endpoint": "https://tts.example/v1/audio/speech",
                "status_code": 400,
                "request_id": "remote-error",
                "remote_message": "bad ref_audio; token=***",
                "message": (
                    "Remote TTS request failed: "
                    "endpoint='https://tts.example/v1/audio/speech', "
                    "status=400, request_id='remote-error', "
                    "message='bad ref_audio; token=***'."
                ),
            },
        )


class TestRemoteTTSCredential(TestCase):
    """Credential validation and public-view redaction."""

    def test_rejects_non_http_endpoint(self) -> None:
        """Remote endpoints cannot use local file or other URL schemes."""
        with self.assertRaises(ValidationError):
            RemoteTTSCredential(base_url="file:///etc/passwd")

    def test_public_view_omits_api_key(self) -> None:
        """Credential APIs must never return the remote bearer token."""
        credential = CredentialFactory.from_dict(
            {
                "type": "remote_tts_credential",
                "base_url": "https://tts.example/v1",
                "api_key": "secret-token",
            },
        )
        view = redact_credential_view(
            CredentialView.model_validate(
                {
                    "id": "credential-a",
                    "user_id": "user-a",
                    "data": {
                        "type": "remote_tts_credential",
                        "base_url": "https://tts.example/v1",
                        "api_key": "secret-token",
                    },
                    "editable": True,
                },
            ),
        )
        self.assertEqual(
            {
                "credential_type": type(credential),
                "view": view.model_dump(
                    exclude={"created_at", "updated_at"},
                ),
            },
            {
                "credential_type": RemoteTTSCredential,
                "view": {
                    "id": "credential-a",
                    "user_id": "user-a",
                    "data": {
                        "type": "remote_tts_credential",
                        "base_url": "https://tts.example/v1",
                    },
                    "editable": True,
                },
            },
        )
