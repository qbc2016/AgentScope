# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for the Voicebox TTS model.

Covers:
  * Successful synthesis via MCP endpoint.
  * Handling of None/empty text input.
  * MCP endpoint error handling.
"""
import base64
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch, MagicMock

from agentscope.app._service._tts_model import get_tts_model
from agentscope.app.storage import TTSModelConfig
from agentscope.credential import VoiceboxCredential
from agentscope.tts import VoiceboxTTSModel, TTSResponse


_AUDIO_B64 = base64.b64encode(b"FAKE_WAV_DATA").decode()


class TestVoiceboxTTSModel(IsolatedAsyncioTestCase):
    """Unit tests for VoiceboxTTSModel."""

    def _make_model(
        self,
        **kwargs: object,
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
        result = await model.synthesize(None)
        self.assertIsInstance(result, TTSResponse)
        self.assertIsNone(result.content)

    async def test_successful_synthesis(self) -> None:
        """Successful MCP call returns audio data."""
        model = self._make_model(client_id="agentscope-user")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "content": [
                    {
                        "type": "resource",
                        "resource": {
                            "blob": _AUDIO_B64,
                            "uri": "audio://output.wav",
                        },
                    },
                ],
            },
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(
            return_value=mock_client,
        )
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            return_value=mock_response,
        )

        with patch(
            "httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await model.synthesize("Hello world")

        self.assertIsInstance(result, TTSResponse)
        self.assertIsNotNone(result.content)
        self.assertEqual(
            result.content.model_dump(exclude={"id"}),
            {
                "type": "data",
                "source": {
                    "type": "base64",
                    "data": _AUDIO_B64,
                    "media_type": "audio/wav",
                },
                "name": None,
            },
        )
        self.assertEqual(
            mock_client.post.await_args.kwargs["headers"][
                "X-Voicebox-Client-Id"
            ],
            "agentscope-user",
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
        model = await get_tts_model(
            "user-a",
            TTSModelConfig(
                type="voicebox_credential",
                credential_id="credential-a",
                model="voicebox",
                parameters={},
            ),
            access,
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
            self.assertRaisesRegex(
                RuntimeError,
                "No Voicebox profile is bound",
            ),
        ):
            await model.synthesize("Hello")

        mock_client.post.assert_not_awaited()
        headers = mock_client.get.await_args.kwargs["headers"]
        self.assertEqual(
            headers["X-Voicebox-Client-Id"],
            model._client_id,
        )

    async def test_mcp_error_returns_none(self) -> None:
        """MCP error returns TTSResponse with None content."""
        model = self._make_model()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "error": {"code": -1, "message": "fail"},
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(
            return_value=mock_client,
        )
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            return_value=mock_response,
        )

        with patch(
            "httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await model.synthesize("Hello")

        self.assertIsInstance(result, TTSResponse)
        self.assertIsNone(result.content)
