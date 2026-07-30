# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for the Voicebox TTS model."""
import base64
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
import httpx

from agentscope.app._service._tts_model import (
    get_tts_model,
    validate_tts_model_config,
)
from agentscope.app.storage import CredentialRecord, TTSModelConfig
from agentscope.credential import VoiceboxCredential
from agentscope.tts import VoiceboxTTSModel, TTSResponse


_AUDIO = b"FAKE_WAV_DATA"
_AUDIO_B64 = base64.b64encode(_AUDIO).decode()


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


class TestVoiceboxTTSModel(IsolatedAsyncioTestCase):
    """Unit tests for VoiceboxTTSModel."""

    def _make_model(
        self,
        **kwargs: Any,
    ) -> "VoiceboxTTSModel":
        """Create a VoiceboxTTSModel with test credential."""
        return VoiceboxTTSModel(
            credential=VoiceboxCredential(
                endpoint="http://127.0.0.1:17493",
            ),
            **kwargs,
        )

    async def test_none_text_returns_empty(self) -> None:
        """None text returns TTSResponse with None content."""
        model = self._make_model()
        result = cast(TTSResponse, await model.synthesize(None))
        self.assertEqual(
            _dump_tts_response(result),
            {
                "content": None,
                "type": "tts",
                "usage": None,
                "metadata": None,
                "is_last": True,
            },
        )

    async def test_successful_synthesis_uses_bound_profile_defaults(
        self,
    ) -> None:
        """The bound profile supplies engine/language and receives audio."""
        model = self._make_model(
            client_id="agentscope-user",
            require_client_binding=True,
        )

        binding_response = MagicMock()
        binding_response.raise_for_status = MagicMock()
        binding_response.json.return_value = {
            "items": [
                {
                    "client_id": "agentscope-user",
                    "profile_id": "profile-1",
                    "default_engine": "kokoro",
                },
            ],
        }
        profiles_response = MagicMock()
        profiles_response.raise_for_status = MagicMock()
        profiles_response.json.return_value = [
            {
                "id": "profile-1",
                "name": "Chinese voice",
                "language": "zh",
                "default_engine": None,
                "preset_engine": "kokoro",
            },
        ]
        audio_response = MagicMock()
        audio_response.raise_for_status = MagicMock()
        audio_response.content = _AUDIO
        audio_response.headers = {"content-type": "audio/wav"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(
            side_effect=[binding_response, profiles_response],
        )
        mock_client.post = AsyncMock(return_value=audio_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = cast(TTSResponse, await model.synthesize("你好"))

        post_call = mock_client.post.await_args
        self.assertEqual(
            {
                "response": _dump_tts_response(result),
                "request": {
                    "args": post_call.args,
                    "kwargs": post_call.kwargs,
                },
            },
            {
                "response": {
                    "content": {
                        "type": "data",
                        "source": {
                            "type": "base64",
                            "data": _AUDIO_B64,
                            "media_type": "audio/wav",
                        },
                        "name": None,
                    },
                    "type": "tts",
                    "usage": None,
                    "metadata": None,
                    "is_last": True,
                },
                "request": {
                    "args": ("http://127.0.0.1:17493/generate/stream",),
                    "kwargs": {
                        "json": {
                            "profile_id": "profile-1",
                            "text": "你好",
                            "language": "zh",
                            "engine": "kokoro",
                        },
                        "headers": {
                            "Content-Type": "application/json",
                            "X-Voicebox-Client-Id": "agentscope-user",
                        },
                    },
                },
            },
        )

    async def test_binding_required_prevents_global_fallback(self) -> None:
        """A missing client binding must not use Voicebox's global voice."""
        credential_record = MagicMock()
        credential_record.data = {
            "type": "voicebox_credential",
            "endpoint": "http://127.0.0.1:17493",
        }
        access = AsyncMock()
        access.resolve_credential = AsyncMock(
            return_value=credential_record,
        )
        model = cast(
            VoiceboxTTSModel,
            await get_tts_model(
                "user-a",
                TTSModelConfig(
                    type="voicebox_credential",
                    credential_id="credential-a",
                    model="voicebox",
                    parameters={},
                ),
                access,
            ),
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {
                    "client_id": "another-client",
                    "profile_id": "another-profile",
                },
            ],
        }
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            self.assertRaises(RuntimeError) as raised,
        ):
            await model.synthesize("Hello")

        mock_client.post.assert_not_awaited()
        get_call = mock_client.get.await_args
        self.assertEqual(
            {
                "error": str(raised.exception),
                "request": {
                    "args": get_call.args,
                    "kwargs": get_call.kwargs,
                },
            },
            {
                "error": (
                    "No Voicebox profile is bound to AgentScope client "
                    f"{model._client_id!r}. Keep Voicebox running, open "
                    "Voicebox Settings -> MCP, bind this client to a "
                    "profile, then try again."
                ),
                "request": {
                    "args": ("http://127.0.0.1:17493/mcp/bindings",),
                    "kwargs": {
                        "headers": {
                            "Content-Type": "application/json",
                            "X-Voicebox-Client-Id": model._client_id,
                        },
                    },
                },
            },
        )

    async def test_connection_error_is_actionable(self) -> None:
        """An unreachable service identifies the host and startup fix."""
        model = self._make_model(
            client_id="agentscope-user",
            require_client_binding=True,
        )
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError(
                "All connection attempts failed",
            ),
        )

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            self.assertRaises(RuntimeError) as raised,
        ):
            await model.synthesize("Hello")

        self.assertEqual(
            {
                "type": type(raised.exception),
                "message": str(raised.exception),
            },
            {
                "type": RuntimeError,
                "message": (
                    "Could not connect to Voicebox at "
                    "'http://127.0.0.1:17493' while checking its client "
                    "binding. Start Voicebox 0.5.0 or newer and verify the "
                    "endpoint is reachable from the AgentScope backend. "
                    "127.0.0.1 only works when both services run on the same "
                    "host; it does not refer to the computer running the "
                    "browser."
                ),
            },
        )

    async def test_invalid_app_parameters_are_rejected_before_chat(
        self,
    ) -> None:
        """Legacy hidden parameters cannot defer failure until synthesis."""
        access = AsyncMock()
        access.resolve_credential.return_value = CredentialRecord(
            id="credential-a",
            user_id="user-a",
            data={
                "type": "voicebox_credential",
                "endpoint": "http://127.0.0.1:17493",
            },
        )

        cases = [
            (
                {"personality": True},
                {
                    "status_code": 400,
                    "detail": (
                        "Voicebox personality mode is not supported by "
                        "AgentScope's binary audio integration. Disable "
                        "personality mode."
                    ),
                    "headers": None,
                },
            ),
            (
                {"profile": "voice-a"},
                {
                    "status_code": 400,
                    "detail": (
                        "An explicit Voicebox profile is not allowed in "
                        "AgentScope App. Bind this AgentScope client to a "
                        "profile in Voicebox Settings -> MCP instead."
                    ),
                    "headers": None,
                },
            ),
        ]
        for parameters, expected in cases:
            with (
                self.subTest(parameters=parameters),
                self.assertRaises(HTTPException) as raised,
            ):
                await validate_tts_model_config(
                    "user-a",
                    TTSModelConfig(
                        type="voicebox_credential",
                        credential_id="credential-a",
                        model="voicebox",
                        parameters=parameters,
                    ),
                    access,
                    AsyncMock(),
                )
            self.assertEqual(
                {
                    "status_code": raised.exception.status_code,
                    "detail": raised.exception.detail,
                    "headers": raised.exception.headers,
                },
                expected,
            )
