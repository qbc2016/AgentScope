# -*- coding: utf-8 -*-
"""Required protocol tests for the Remote TTS adapter."""
import asyncio
import base64
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from pydantic import ValidationError

from agentscope.app._service import (
    CredentialView,
    discover_tts_models,
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
        """Free-form voice IDs remain valid without remote discovery."""
        access = AsyncMock()
        access.resolve_credential.return_value = CredentialRecord(
            id="credential-a",
            user_id="user-a",
            data={
                "type": "remote_tts_credential",
                "base_url": "https://tts.example/v1",
            },
        )
        with patch(
            "agentscope.app._service._tts_model.discover_tts_models",
            new=AsyncMock(return_value=[]),
        ):
            model = cast(
                RemoteTTSModel,
                await get_tts_model(
                    "user-a",
                    TTSModelConfig(
                        type="remote_tts_credential",
                        credential_id="credential-a",
                        model="provider-model",
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
                "parameter_descriptions": {
                    name: prop.get("description")
                    for name, prop in card.parameter_schema[
                        "properties"
                    ].items()
                },
            },
            {
                "parameters": {
                    "voice": "zf_xiaobei",
                    "response_format": "wav",
                    "speed": 1.0,
                    "language": None,
                    "instructions": None,
                    "task_type": None,
                    "reference_audio_base64": None,
                    "reference_audio_media_type": "audio/wav",
                    "reference_text": None,
                },
                "voice_schema": {
                    "default": "default",
                    "description": (
                        "Voice identifier understood by the remote service. "
                        "Enter it manually when the endpoint does not expose "
                        "voice discovery."
                    ),
                    "title": "Voice",
                    "type": "string",
                },
                "parameter_descriptions": {
                    "voice": (
                        "Voice identifier understood by the remote service. "
                        "Enter it manually when the endpoint does not expose "
                        "voice discovery."
                    ),
                    "response_format": (
                        "Requested audio encoding. The selected remote model "
                        "must support this format."
                    ),
                    "speed": (
                        "Speech speed multiplier passed to the remote service."
                    ),
                    "language": (
                        "Optional language code or name understood by the "
                        "remote service."
                    ),
                    "instructions": (
                        "Optional synthesis instructions. Support depends on "
                        "the selected remote model."
                    ),
                    "task_type": (
                        "Optional remote task type, for example CustomVoice, "
                        "VoiceDesign, or Base."
                    ),
                    "reference_text": (
                        "Optional transcript of the reference audio, required "
                        "by some voice-cloning models."
                    ),
                },
            },
        )

    async def test_maps_request_and_binary_response(self) -> None:
        """Standard fields, reference audio, headers, and media type map."""
        credential = RemoteTTSCredential(
            base_url="http://tts.example/v1/",
            api_key="secret-token",
        )
        parameters = RemoteTTSModel.Parameters(
            voice="speaker-a",
            response_format="mp3",
            speed=1.2,
            language="zh",
            instructions="calm",
            task_type="Base",
            reference_audio_base64="QUJD",
            reference_audio_media_type="audio/ogg",
            reference_text="reference",
        )
        model = RemoteTTSModel(
            credential=credential,
            model="tada",
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
                            "task_type": "Base",
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
        model = RemoteTTSModel(
            credential=credential,
            model="provider-model",
        )
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

    async def test_discovers_capabilities_and_voices(self) -> None:
        """Capability discovery produces endpoint-specific model choices."""
        credential = RemoteTTSCredential(
            base_url="https://tts.example/v1",
            api_key="secret-token",
        )
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get.side_effect = [
            httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "kokoro",
                            "label": "Kokoro",
                            "response_formats": ["wav", "mp3"],
                            "languages": ["zh", "en"],
                            "task_types": ["Base"],
                            "voice_cloning": False,
                            "reference_audio_required": False,
                            "streaming": True,
                        },
                    ],
                },
            ),
            httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "zf_xiaobei"},
                        {"id": "zm_yunjian"},
                    ],
                },
            ),
        ]

        with patch(
            "agentscope.tts._remote._model.httpx.AsyncClient",
            return_value=client,
        ):
            cards = await RemoteTTSModel.discover_models(credential)

        discovered = cards[0]
        properties = discovered.parameter_schema["properties"]
        self.assertEqual(
            {
                "names": [card.name for card in cards],
                "output_types": discovered.output_types,
                "realtime": discovered.realtime,
                "languages": properties["language"]["enum"],
                "task_types": properties["task_type"]["enum"],
                "voices": properties["voice"]["enum"],
                "authorization": client.get.await_args_list[0].kwargs[
                    "headers"
                ],
            },
            {
                "names": ["kokoro", "remote-tts"],
                "output_types": ["audio/wav", "audio/mpeg"],
                "realtime": False,
                "languages": ["zh", "en"],
                "task_types": ["Base"],
                "voices": ["zf_xiaobei", "zm_yunjian"],
                "authorization": {
                    "Authorization": "Bearer secret-token",
                },
            },
        )

    async def test_discovery_failure_keeps_manual_model_option(self) -> None:
        """An unavailable discovery API must not block manual model IDs."""
        credential = RemoteTTSCredential(
            base_url="https://offline.example/v1",
        )
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get.side_effect = [
            httpx.Response(404),
            httpx.Response(404),
        ]

        with patch(
            "agentscope.tts._remote._model.httpx.AsyncClient",
            return_value=client,
        ):
            cards = await RemoteTTSModel.discover_models(credential)

        self.assertEqual(
            {
                "models": [card.name for card in cards],
                "urls": [call.args[0] for call in client.get.await_args_list],
            },
            {
                "models": ["remote-tts"],
                "urls": [
                    "https://offline.example/v1/audio/models",
                    "https://offline.example/v1/models",
                ],
            },
        )

    async def test_malformed_capability_falls_back_to_standard_models(
        self,
    ) -> None:
        """Malformed capability JSON must not skip the /models fallback."""
        credential = RemoteTTSCredential(
            base_url="https://tts.example/v1",
        )
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get.side_effect = [
            httpx.Response(200, content=b"<html>not json</html>"),
            httpx.Response(
                200,
                json={"data": [{"id": "standard-model"}]},
            ),
        ]

        with patch(
            "agentscope.tts._remote._model.httpx.AsyncClient",
            return_value=client,
        ):
            cards = await RemoteTTSModel.discover_models(credential)

        self.assertEqual(
            {
                "models": [card.name for card in cards],
                "urls": [call.args[0] for call in client.get.await_args_list],
            },
            {
                "models": ["standard-model", "remote-tts"],
                "urls": [
                    "https://tts.example/v1/audio/models",
                    "https://tts.example/v1/models",
                ],
            },
        )

    async def test_discovery_deadline_bounds_voice_fanout(self) -> None:
        """Slow voice endpoints retain models within bounded concurrency."""
        credential = RemoteTTSCredential(
            base_url="https://slow.example/v1",
        )
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        active_requests = 0
        max_active_requests = 0
        voice_requests = 0

        async def get(url: str, **kwargs: object) -> httpx.Response:
            del kwargs
            nonlocal active_requests
            nonlocal max_active_requests
            nonlocal voice_requests
            if url.endswith("/audio/models"):
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": f"model-{index}"} for index in range(40)
                        ],
                    },
                )
            voice_requests += 1
            active_requests += 1
            max_active_requests = max(
                max_active_requests,
                active_requests,
            )
            try:
                await asyncio.sleep(1)
            finally:
                active_requests -= 1
            return httpx.Response(200, json={"data": []})

        client.get = AsyncMock(side_effect=get)
        with (
            patch(
                "agentscope.tts._remote._model.httpx.AsyncClient",
                return_value=client,
            ),
            patch(
                "agentscope.tts._remote._model._DISCOVERY_DEADLINE_SECONDS",
                0.02,
            ),
            patch(
                "agentscope.tts._remote._model."
                "_MAX_CONCURRENT_VOICE_REQUESTS",
                3,
            ),
        ):
            cards = await RemoteTTSModel.discover_models(credential)

        self.assertEqual(len(cards), 41)
        self.assertLessEqual(max_active_requests, 3)
        self.assertLessEqual(voice_requests, 3)

    async def test_discovery_cache_is_scoped_by_credential_revision(
        self,
    ) -> None:
        """Repeated reads share a cache until the credential is edited."""
        updated_at = datetime(2026, 7, 30, 12, 0, 0)
        record = CredentialRecord(
            id="credential-cache-test",
            user_id="owner-cache-test",
            updated_at=updated_at,
            data={
                "type": "remote_tts_credential",
                "base_url": "https://tts.example/v1",
            },
        )
        discovered = RemoteTTSCredential.list_tts_models()
        discover_mock = AsyncMock(return_value=discovered)

        with patch(
            "agentscope.tts._remote._model.RemoteTTSModel.discover_models",
            discover_mock,
        ):
            first = await discover_tts_models(record)
            second = await discover_tts_models(record)
            revised = record.model_copy(
                update={
                    "updated_at": updated_at + timedelta(seconds=1),
                },
            )
            third = await discover_tts_models(revised)

        self.assertIs(first, second)
        self.assertEqual(third, discovered)
        self.assertEqual(discover_mock.await_count, 2)


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
