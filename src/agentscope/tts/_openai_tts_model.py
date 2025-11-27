# -*- coding: utf-8 -*-
"""OpenAI TTS model implementation."""
import base64
from typing import TYPE_CHECKING

from ._tts_base import TTSModelBase
from ._tts_response import TTSResponse
from ..message import Msg, AudioBlock, Base64Source
from ..types import JSONSerializableObject

if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = "openai.OpenAI"


class OpenAITTSModel(TTSModelBase):
    """OpenAI TTS model implementation."""

    # This model does not support streaming input (requires complete text)
    supports_streaming_input: bool = False

    def __init__(
        self,
        model_name: str = "gpt-4o-mini-tts",
        api_key: str | None = None,
        voice: str = "alloy",
        client_kwargs: dict = None,
        generate_kwargs: dict[str, JSONSerializableObject] | None = None,
    ) -> None:
        """Initialize the OpenAI TTS model.

        Args:
            model_name (`str`):
                The TTS model name. Defaults to "gpt-4o-mini-tts".
            api_key (`str`, optional):
                The OpenAI API key. If not provided, will use
                environment variable OPENAI_API_KEY.
            voice (`str`):
                The voice to use. Options: "alloy", "echo", "fable", "onyx",
                "nova", "shimmer". Defaults to "alloy".
            client_kwargs (`dict`, default `None`):
                The extra keyword arguments to initialize the OpenAI client.
            generate_kwargs (`dict[str, JSONSerializableObject] | None`, \
             optional):
               The extra keyword arguments used in OpenAI API generation,
               e.g. `temperature`, `seed`.
        """
        super().__init__(model_name=model_name, stream=False)

        self.api_key = api_key
        self.voice = voice
        self._client: OpenAI | None = None
        self._connected = False
        # Text buffer for each message to accumulate text before synthesis
        # Key is msg.id, value is the accumulated text
        self._text_buffer: dict[str, str] = {}
        self.client_kwargs = client_kwargs or {}
        self.generate_kwargs = generate_kwargs or {}

    async def initialize(self) -> None:
        """Initialize the OpenAI TTS model and create client."""
        if self._connected:
            return

        import openai

        self._client = openai.OpenAI(
            api_key=self.api_key,
            **self.client_kwargs,
        )

        self._connected = True

    async def _call_api(self, msg: Msg, last: bool = False) -> TTSResponse:
        """Append text to be synthesized and return TTS response.

        Args:
            msg (`Msg`):
                The message to be synthesized.
            last (`bool`):
                Whether this is the last chunk. Defaults to False.

        Returns:
            `TTSResponse`:
                The TTSResponse containing audio blocks.
        """
        if not self._connected or self._client is None:
            raise RuntimeError(
                "TTS model is not initialized. Call initialize() first.",
            )

        msg_id = msg.id
        # Initialize text buffer for this message if not exists
        if msg_id not in self._text_buffer:
            self._text_buffer[msg_id] = ""

        # Extract text content
        for block in msg.get_content_blocks():
            if block["type"] == "text":
                text = block["text"]
                self._text_buffer[msg_id] = text

        # Only call API for synthesis when last=True
        if last and self._text_buffer.get(msg_id):
            try:
                # Call OpenAI TTS API
                response = self._client.audio.speech.create(
                    model=self.model_name,
                    voice=self.voice,
                    input=self._text_buffer[msg_id],
                    **self.generate_kwargs,
                )

                # Get audio data
                audio_data = response.content

                # Convert audio data to base64
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")

                # Clear text buffer for this message
                del self._text_buffer[msg_id]

                audio_block = AudioBlock(
                    type="audio",
                    source=Base64Source(
                        type="base64",
                        data=audio_base64,
                        media_type="audio/mp3",
                    ),
                )
                return TTSResponse(content=[audio_block])
            except Exception:
                import traceback

                traceback.print_exc()
                # Clear text buffer for this message on error
                if msg_id in self._text_buffer:
                    del self._text_buffer[msg_id]

                audio_block = AudioBlock(
                    type="audio",
                    source=Base64Source(
                        type="base64",
                        data="",
                        media_type="audio/mp3",
                    ),
                )
                return TTSResponse(content=[audio_block])
        else:
            # Not the last chunk, return empty AudioBlock
            audio_block = AudioBlock(
                type="audio",
                source=Base64Source(
                    type="base64",
                    data="",
                    media_type="audio/mp3",
                ),
            )
            return TTSResponse(content=[audio_block])

    async def close(self) -> None:
        """Close the OpenAI TTS model and clean up resources."""
        if not self._connected:
            return

        self._client = None
        self._connected = False
        self._text_buffer.clear()
