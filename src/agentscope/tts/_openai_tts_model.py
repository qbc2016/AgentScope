# -*- coding: utf-8 -*-
"""OpenAI TTS model implementation."""
import base64
from typing import TYPE_CHECKING

from ._tts_base import TTSModelBase
from ..message import Msg, AudioBlock, Base64Source


if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = "openai.OpenAI"


class OpenAITTSModel(TTSModelBase):
    """OpenAI TTS model implementation."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini-tts",
        api_key: str | None = None,
        voice: str = "alloy",
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
        """
        super().__init__(model_name=model_name, stream=False)

        self.api_key = api_key
        self.voice = voice
        self._client: OpenAI | None = None
        self._connected = False
        self._text_buffer = ""

    async def initialize(self) -> None:
        """Initialize the OpenAI TTS model and create client."""
        if self._connected:
            return

        import openai

        self._client = openai.OpenAI(api_key=self.api_key)

        self._connected = True
        self._text_buffer = ""
        print("[OpenAI TTS] TTS service initialized")

    async def send_msg(self, msg: Msg, last: bool = False) -> AudioBlock:
        """Append text to be synthesized and return audio block.

        Args:
            msg (`Msg`):
                The message to be synthesized.
            last (`bool`):
                Whether this is the last chunk. Defaults to False.

        Returns:
            `AudioBlock`:
                The AudioBlock (may have empty data if not last or no audio
                available).
        """
        if not self._connected or self._client is None:
            raise RuntimeError(
                "TTS model is not initialized. Call initialize() first.",
            )

        # Extract text content
        for block in msg.get_content_blocks():
            if block["type"] == "text":
                text = block["text"]
                self._text_buffer += text

        # Only call API for synthesis when last=True
        if last and self._text_buffer:
            try:
                # Call OpenAI TTS API
                response = self._client.audio.speech.create(
                    model=self.model_name,
                    voice=self.voice,
                    input=self._text_buffer,
                )

                # Get audio data
                audio_data = response.content

                # Convert audio data to base64
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")

                # Clear text buffer
                self._text_buffer = ""

                # Auto close when last=True
                await self.close()

                return AudioBlock(
                    type="audio",
                    source=Base64Source(
                        type="base64",
                        data=audio_base64,
                        media_type="audio/mp3",
                    ),
                )
            except Exception as e:
                print(f"[OpenAI TTS Error] {e}")
                import traceback

                traceback.print_exc()
                self._text_buffer = ""

                # Auto close when last=True (even on error)
                await self.close()

                return AudioBlock(
                    type="audio",
                    source=Base64Source(
                        type="base64",
                        data="",
                        media_type="audio/mp3",
                    ),
                )
        else:
            # Not the last chunk, return empty AudioBlock
            return AudioBlock(
                type="audio",
                source=Base64Source(
                    type="base64",
                    data="",
                    media_type="audio/mp3",
                ),
            )

    async def close(self) -> None:
        """Close the OpenAI TTS model and clean up resources."""
        if not self._connected:
            return

        self._client = None
        self._connected = False
        self._text_buffer = ""

    def is_initialized(self) -> bool:
        """Check if the TTS model is initialized.

        Returns:
            `bool`:
                True if initialized, False otherwise.
        """
        return self._connected
