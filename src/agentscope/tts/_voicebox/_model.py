# -*- coding: utf-8 -*-
"""Voicebox TTS model implementation via its REST audio endpoint."""
import base64
import json
from typing import Any, AsyncGenerator, cast, Literal

from pydantic import BaseModel, Field

from .._tts_base import TTSModelBase
from .._tts_response import TTSResponse
from ...credential import VoiceboxCredential
from ...message import DataBlock, Base64Source


_DEFAULT_MEDIA_TYPE = "audio/wav"

_VoiceboxEngine = Literal[
    "qwen",
    "qwen_custom_voice",
    "luxtts",
    "chatterbox",
    "chatterbox_turbo",
    "tada",
    "kokoro",
]

_VoiceboxLanguage = Literal[
    "zh",
    "en",
    "ja",
    "ko",
    "de",
    "fr",
    "ru",
    "pt",
    "es",
    "it",
    "he",
    "ar",
    "da",
    "el",
    "fi",
    "hi",
    "ms",
    "nl",
    "no",
    "pl",
    "sv",
    "sw",
    "tr",
]


class VoiceboxTTSModel(TTSModelBase):
    """Voicebox TTS model using Voicebox's REST audio endpoint.

    AgentScope resolves the Voicebox profile assigned to its per-user client
    binding, then calls ``POST /generate/stream``. This endpoint returns WAV
    bytes to AgentScope. The similarly named ``voicebox.speak`` MCP tool is
    intentionally not used: it only returns a generation id and also plays
    audio on the machine running Voicebox.

    For more details see the `Voicebox documentation
    <https://github.com/jamiepine/voicebox>`_.
    """

    class Parameters(BaseModel):
        """Frontend-exposed parameters for Voicebox TTS."""

        profile: str | None = Field(
            default=None,
            title="Voice Profile",
            description=(
                "The Voicebox profile name or ID. AgentScope App mode uses "
                "the per-client binding instead."
            ),
        )

        engine: _VoiceboxEngine | None = Field(
            default=None,
            title="TTS Engine",
            description=(
                "Optional engine override. Leave unset to use the engine "
                "configured on the Voicebox client binding or profile."
            ),
        )

        language: _VoiceboxLanguage | None = Field(
            default=None,
            title="Language",
            description=(
                "Optional language override. Leave unset to use the language "
                "configured on the bound Voicebox profile."
            ),
        )

        personality: bool = Field(
            default=False,
            title="Personality Mode",
            description=(
                "Voicebox personality rewriting is not available through "
                "the binary audio endpoint used by AgentScope."
            ),
        )

    parameters: Parameters
    """Validated Voicebox synthesis parameters."""

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
                Placeholder model name; Voicebox manages engines.
            parameters (`Parameters | None`, defaults to `None`):
                The TTS parameters.
            stream (`bool`, defaults to `False`):
                Kept for the common TTS interface. AgentScope receives one
                complete audio response.
            client_id (`str | None`, defaults to `None`):
                Value sent in the ``X-Voicebox-Client-Id`` header.
            require_client_binding (`bool`, defaults to `False`):
                Require ``client_id`` to have a Voicebox profile binding.
                This prevents fallback to another user's global profile.
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
        self.parameters = cast(
            VoiceboxTTSModel.Parameters,
            self.parameters,
        )
        self._endpoint = credential.endpoint.rstrip("/")
        if self._endpoint.endswith("/mcp"):
            self._endpoint = self._endpoint[: -len("/mcp")]
        self._client_id = client_id
        self._require_client_binding = require_client_binding
        if self.parameters.personality:
            raise ValueError(
                "Voicebox personality mode is not supported by the binary "
                "audio endpoint used by AgentScope.",
            )
        if require_client_binding and self.parameters.profile is not None:
            raise ValueError(
                "An explicit Voicebox profile is not allowed when per-client "
                "isolation is enabled. Configure the profile binding in "
                "Voicebox Settings -> MCP instead.",
            )
        self._client_binding: dict[str, Any] | None = None
        self._resolved_profiles: dict[str, dict[str, Any]] = {}

    def _request_headers(self) -> dict[str, str]:
        """Build headers shared by Voicebox HTTP requests."""
        headers = {"Content-Type": "application/json"}
        if self._client_id:
            headers["X-Voicebox-Client-Id"] = self._client_id
        return headers

    def _connection_error(self, operation: str) -> RuntimeError:
        """Build an actionable error for an unreachable Voicebox service."""
        return RuntimeError(
            f"Could not connect to Voicebox at {self._endpoint!r} while "
            f"{operation}. Start Voicebox 0.5.0 or newer and verify the "
            "endpoint is reachable from the AgentScope backend. 127.0.0.1 "
            "only works when both services run on the same host; it does "
            "not refer to the computer running the browser.",
        )

    @staticmethod
    def _response_error(response: Any) -> str:
        """Extract a concise error message from a Voicebox response."""
        try:
            body = response.json()
        except (json.JSONDecodeError, TypeError, ValueError):
            return str(getattr(response, "text", ""))[:1000] or str(
                getattr(response, "reason_phrase", ""),
            )
        if isinstance(body, dict):
            detail = body.get("detail")
            if detail is not None:
                return str(detail)[:1000]
            error = body.get("error")
            if error is not None:
                return str(error)[:1000]
        return str(getattr(response, "text", ""))[:1000] or str(
            getattr(response, "reason_phrase", ""),
        )

    async def get_client_binding(self) -> dict[str, Any] | None:
        """Return this client's Voicebox binding, if one exists.

        Calling the binding endpoint with ``X-Voicebox-Client-Id`` also lets
        Voicebox discover a new AgentScope client so it can be configured in
        Voicebox Settings -> MCP.
        """
        if self._client_binding is not None:
            return self._client_binding
        if self._client_id is None:
            return None

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for VoiceboxTTSModel. "
                "Install with: pip install httpx",
            ) from e

        url = f"{self._endpoint}/mcp/bindings"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    url,
                    headers=self._request_headers(),
                )
                response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise self._connection_error("checking its client binding") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise RuntimeError(
                    "Voicebox is reachable but does not provide "
                    "GET /mcp/bindings. AgentScope requires Voicebox 0.5.0 "
                    "or newer.",
                ) from e
            raise RuntimeError(
                "Voicebox client-binding check failed with HTTP "
                f"{e.response.status_code}: "
                f"{self._response_error(e.response)}",
            ) from e

        try:
            payload = response.json()
            items = payload["items"]
            if not isinstance(items, list):
                raise TypeError
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(
                "Voicebox returned an invalid response from "
                "GET /mcp/bindings.",
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
        if binding is not None:
            self._client_binding = binding
        return binding

    async def _ensure_client_binding(self) -> dict[str, Any]:
        """Require this client to have its own Voicebox profile binding."""
        binding = await self.get_client_binding()
        if not binding or not binding.get("profile_id"):
            raise RuntimeError(
                "No Voicebox profile is bound to AgentScope client "
                f"{self._client_id!r}. Keep Voicebox running, open "
                "Voicebox Settings -> MCP, bind this client to a profile, "
                "then try again.",
            )
        return binding

    async def _load_profile(
        self,
        profile: str,
    ) -> dict[str, Any]:
        """Resolve a Voicebox profile id or name."""
        cache_key = profile.casefold()
        if cache_key in self._resolved_profiles:
            return self._resolved_profiles[cache_key]

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for VoiceboxTTSModel. "
                "Install with: pip install httpx",
            ) from e

        url = f"{self._endpoint}/profiles"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    url,
                    headers=self._request_headers(),
                )
                response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise self._connection_error("loading its voice profiles") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                "Voicebox profile lookup failed with HTTP "
                f"{e.response.status_code}: "
                f"{self._response_error(e.response)}",
            ) from e

        try:
            profiles = response.json()
            if not isinstance(profiles, list):
                raise TypeError
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                "Voicebox returned an invalid response from GET /profiles.",
            ) from e

        normalized = profile.casefold()
        resolved = next(
            (
                item
                for item in profiles
                if isinstance(item, dict)
                and (
                    item.get("id") == profile
                    or (
                        isinstance(item.get("name"), str)
                        and item["name"].casefold() == normalized
                    )
                )
            ),
            None,
        )
        if resolved is None:
            raise RuntimeError(
                f"Voicebox profile {profile!r} was not found at "
                f"{self._endpoint!r}.",
            )
        self._resolved_profiles[cache_key] = resolved
        for identity_key in ("id", "name"):
            identity = resolved.get(identity_key)
            if isinstance(identity, str):
                self._resolved_profiles[identity.casefold()] = resolved
        return resolved

    async def _resolve_profile_and_binding(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Resolve the profile used for binary audio generation."""
        if self._require_client_binding:
            binding = await self._ensure_client_binding()
            return await self._load_profile(binding["profile_id"]), binding

        if self.parameters.profile is not None:
            return await self._load_profile(self.parameters.profile), None

        client_binding = await self.get_client_binding()
        if client_binding and client_binding.get("profile_id"):
            return (
                await self._load_profile(client_binding["profile_id"]),
                client_binding,
            )

        raise RuntimeError(
            "Voicebox audio generation requires either an explicit profile "
            "or a client binding. Configure a profile in Voicebox Settings "
            "-> MCP.",
        )

    async def _generate_audio(
        self,
        text: str,
        profile: dict[str, Any],
        binding: dict[str, Any] | None,
    ) -> tuple[bytes, str]:
        """Generate and return audio bytes from Voicebox."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for VoiceboxTTSModel. "
                "Install with: pip install httpx",
            ) from e

        engine = self.parameters.engine
        if engine is None and binding is not None:
            engine = binding.get("default_engine")
        if engine is None:
            engine = profile.get("default_engine") or profile.get(
                "preset_engine",
            )
        language = self.parameters.language or profile.get("language")
        if not isinstance(language, str) or not language:
            raise RuntimeError(
                f"Voicebox profile {profile.get('id')!r} has no language. "
                "Set a language override in AgentScope or configure the "
                "profile language in Voicebox.",
            )

        payload = {
            "profile_id": profile["id"],
            "text": text,
            "language": language,
            # Voicebox defaults an omitted engine to qwen before consulting
            # the profile. Explicit null preserves profile-side resolution.
            "engine": engine,
        }
        url = f"{self._endpoint}/generate/stream"
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._request_headers(),
                )
                response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise self._connection_error("generating speech") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                "Voicebox speech generation failed with HTTP "
                f"{e.response.status_code}: "
                f"{self._response_error(e.response)}",
            ) from e

        media_type = response.headers.get(
            "content-type",
            _DEFAULT_MEDIA_TYPE,
        ).split(";", maxsplit=1)[0]
        if not media_type.startswith("audio/"):
            raise RuntimeError(
                "Voicebox returned a non-audio response from "
                f"POST /generate/stream: {media_type!r}.",
            )
        if not response.content:
            raise RuntimeError("Voicebox returned an empty audio response.")
        return response.content, media_type

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
        profile, binding = await self._resolve_profile_and_binding()
        audio, media_type = await self._generate_audio(
            text,
            profile,
            binding,
        )
        return TTSResponse(
            content=DataBlock(
                source=Base64Source(
                    data=base64.b64encode(audio).decode("ascii"),
                    media_type=media_type,
                ),
            ),
            is_last=True,
        )
