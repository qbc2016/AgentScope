# -*- coding: utf-8 -*-
"""Voicebox TTS model implementation via MCP HTTP endpoint."""
import json
from typing import Any, AsyncGenerator, Literal

from pydantic import BaseModel, Field

from .._tts_base import TTSModelBase
from .._tts_response import TTSResponse
from ..._logging import logger
from ...credential import VoiceboxCredential
from ...message import DataBlock, Base64Source


_MEDIA_TYPE = "audio/wav"


class VoiceboxTTSModel(TTSModelBase):
    """Voicebox TTS model via MCP Streamable HTTP.

    Connects to a local Voicebox instance and calls
    ``voicebox.speak`` to synthesize speech. Voicebox
    manages multiple TTS engines and voice profiles
    locally.

    For more details see the `Voicebox documentation
    <https://github.com/jamiepine/voicebox>`_.
    """

    class Parameters(BaseModel):
        """Frontend-exposed parameters for Voicebox TTS."""

        profile: str | None = Field(
            default=None,
            title="Voice Profile",
            description=(
                "The voice profile name or ID to use. "
                "Falls back to the Voicebox default if not set."
            ),
        )

        engine: Literal[
            "qwen",
            "qwen_custom_voice",
            "luxtts",
            "chatterbox",
            "chatterbox_turbo",
            "tada",
            "kokoro",
        ] | None = Field(
            default=None,
            title="TTS Engine",
            description=("The TTS engine to use inside Voicebox."),
        )

        language: str | None = Field(
            default=None,
            title="Language",
            description="The language code (e.g. 'en', 'zh').",
        )

        personality: bool = Field(
            default=False,
            title="Personality Mode",
            description=(
                "When true, Voicebox rewrites text via the "
                "profile's personality LLM before TTS."
            ),
        )

    type: Literal["voicebox_tts"] = "voicebox_tts"
    """The type of the TTS model."""

    realtime: bool = False

    def __init__(
        self,
        credential: VoiceboxCredential,
        model: str = "voicebox",
        parameters: "VoiceboxTTSModel.Parameters | None" = None,
        stream: bool = False,
        client_id: str | None = None,
        require_client_binding: bool = False,
    ) -> None:
        """Initialize the Voicebox TTS model.

        Args:
            credential (`VoiceboxCredential`):
                The Voicebox credential (endpoint URL).
            model (`str`, defaults to ``"voicebox"``):
                The model name (placeholder, Voicebox manages
                engines internally).
            parameters (`Parameters | None`, defaults to `None`):
                The TTS parameters.
            stream (`bool`, defaults to `False`):
                Whether to stream output. Voicebox returns
                full audio in one response.
            client_id (`str | None`, defaults to `None`):
                Value sent in the ``X-Voicebox-Client-Id`` header.
            require_client_binding (`bool`, defaults to `False`):
                Require ``client_id`` to have a Voicebox profile binding.
                This prevents Voicebox from falling back to its global
                default profile.
        """
        if require_client_binding and not client_id:
            raise ValueError(
                "client_id is required when require_client_binding is true.",
            )
        super().__init__(
            credential=credential,
            model=model,
            parameters=parameters,
            stream=stream,
        )
        self._endpoint = credential.endpoint.rstrip("/")
        self._client_id = client_id
        self._require_client_binding = require_client_binding
        self._client_binding_checked = False

    def _request_headers(self) -> dict[str, str]:
        """Build headers shared by Voicebox HTTP requests."""
        headers = {"Content-Type": "application/json"}
        if self._client_id:
            headers["X-Voicebox-Client-Id"] = self._client_id
        return headers

    async def _ensure_client_binding(self) -> None:
        """Ensure this client has its own Voicebox profile binding.

        Voicebox otherwise falls back to a process-wide default profile,
        which is unsafe for a shared AgentScope deployment.
        """
        if self._client_binding_checked:
            return

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for VoiceboxTTSModel. "
                "Install with: pip install httpx",
            ) from e

        assert self._client_id is not None
        url = f"{self._endpoint}/mcp/bindings"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    url,
                    headers=self._request_headers(),
                )
                response.raise_for_status()
                items = response.json().get("items", [])
        except Exception as e:
            raise RuntimeError(
                "Could not verify the Voicebox client binding. "
                "Make sure Voicebox supports GET /mcp/bindings.",
            ) from e

        binding = next(
            (
                item
                for item in items
                if isinstance(item, dict)
                and item.get("client_id") == self._client_id
            ),
            None,
        )
        if not binding or not binding.get("profile_id"):
            raise RuntimeError(
                "No Voicebox profile is bound to AgentScope client "
                f"{self._client_id!r}. Open Voicebox Settings -> MCP, "
                "bind this client to a profile, then try again.",
            )
        self._client_binding_checked = True

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a Voicebox MCP tool via HTTP POST.

        Args:
            tool_name (`str`):
                The MCP tool name (e.g. 'voicebox.speak').
            arguments (`dict[str, Any]`):
                The tool arguments.

        Returns:
            `dict[str, Any]`: The MCP response content.
        """
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for VoiceboxTTSModel. "
                "Install with: pip install httpx",
            ) from e

        url = f"{self._endpoint}/mcp"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        async with httpx.AsyncClient(
            timeout=120.0,
        ) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self._request_headers(),
            )
            response.raise_for_status()
            result = response.json()

        if "error" in result:
            raise RuntimeError(
                f"Voicebox MCP error: {result['error']}",
            )

        return result.get("result", {})

    async def synthesize(
        self,
        text: str | None = None,
        **kwargs: Any,
    ) -> TTSResponse | AsyncGenerator[TTSResponse, None]:
        """Synthesize speech using Voicebox.

        Args:
            text (`str | None`, defaults to `None`):
                The text to synthesize.
            **kwargs (`Any`):
                Additional keyword arguments.

        Returns:
            `TTSResponse | AsyncGenerator[TTSResponse, None]`:
                The synthesized audio response.
        """
        if text is None:
            return TTSResponse(content=None)

        if self._require_client_binding:
            if self.parameters.profile is not None:
                raise RuntimeError(
                    "An explicit Voicebox profile is not allowed when "
                    "per-client isolation is enabled. Configure the profile "
                    "binding in Voicebox Settings -> MCP instead.",
                )
            await self._ensure_client_binding()

        arguments: dict[str, Any] = {"text": text}

        if self.parameters.profile is not None:
            arguments["profile"] = self.parameters.profile

        if self.parameters.engine is not None:
            arguments["engine"] = self.parameters.engine

        if self.parameters.language is not None:
            arguments["language"] = self.parameters.language

        if self.parameters.personality:
            arguments["personality"] = True

        try:
            result = await self._call_mcp_tool(
                "voicebox.speak",
                arguments,
            )
        except Exception as e:
            logger.error("Voicebox TTS failed: %s", e)
            return TTSResponse(content=None)

        content_list = result.get("content", [])
        audio_data = None
        for item in content_list:
            if isinstance(item, dict):
                if item.get("type") == "resource":
                    resource = item.get("resource", {})
                    blob = resource.get("blob", "")
                    if blob:
                        audio_data = blob
                        break
                elif item.get("type") == "text":
                    text_content = item.get("text", "")
                    if text_content:
                        try:
                            data = json.loads(text_content)
                            if "audio" in data:
                                audio_data = data["audio"]
                        except (json.JSONDecodeError, TypeError):
                            pass

        if audio_data is None:
            logger.warning(
                "Voicebox returned no audio data.",
            )
            return TTSResponse(content=None)

        return TTSResponse(
            content=DataBlock(
                source=Base64Source(
                    data=audio_data,
                    media_type=_MEDIA_TYPE,
                ),
            ),
            is_last=True,
        )
