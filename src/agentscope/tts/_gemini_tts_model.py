# -*- coding: utf-8 -*-
"""Gemini TTS model implementation."""
import base64
from typing import TYPE_CHECKING

from ._tts_base import TTSModelBase
from ..message import Msg, AudioBlock, Base64Source

if TYPE_CHECKING:
    from google.genai import Client
else:
    Client = "google.genai.Client"


class GeminiTTSModel(TTSModelBase):
    """Gemini TTS model implementation."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-preview-tts",
        api_key: str | None = None,
        voice: str = "Kore",
    ) -> None:
        """Initialize the Gemini TTS model.

        Args:
            model_name (`str`):
                The TTS model name. Defaults to "gemini-2.5-flash-preview-tts".
            api_key (`str`, optional):
                The Gemini API key. If not provided, will use
                environment variable GOOGLE_API_KEY or GEMINI_API_KEY.
            voice (`str`):
                The voice name to use. Defaults to "Kore".
        """
        super().__init__(model_name=model_name, stream=False)

        self.api_key = api_key
        self.voice = voice
        self._client: Client | None = None
        self._connected = False
        # Text buffer for each message to accumulate text before synthesis
        # Key is msg.id, value is the accumulated text
        self._text_buffer: dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize the Gemini TTS model and create client."""
        if self._connected:
            return
        from google import genai

        # Create client (API key is set via environment variable)
        self._client = genai.Client(
            api_key=self.api_key,
        )

        self._connected = True
        print("[Gemini TTS] TTS service initialized")

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
        from google.genai import types

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
                self._text_buffer[msg_id] += text

        # Only call API for synthesis when last=True
        if last and self._text_buffer.get(msg_id):
            try:
                # Call Gemini TTS API
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=self._text_buffer[msg_id],
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(  # noqa
                                    voice_name=self.voice,
                                ),
                            ),
                        ),
                    ),
                )

                # Extract audio data
                if (
                    response.candidates
                    and response.candidates[0].content
                    and response.candidates[0].content.parts
                    and response.candidates[0].content.parts[0].inline_data
                ):
                    audio_data = (
                        response.candidates[0]
                        .content.parts[0]
                        .inline_data.data
                    )
                    # Convert PCM data to base64
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

                    # Clear text buffer for this message
                    del self._text_buffer[msg_id]

                    return AudioBlock(
                        type="audio",
                        source=Base64Source(
                            type="base64",
                            data=audio_base64,
                            media_type="audio/pcm;rate=24000",
                        ),
                    )
                else:
                    # No audio data returned
                    # Clear text buffer for this message
                    if msg_id in self._text_buffer:
                        del self._text_buffer[msg_id]

                    return AudioBlock(
                        type="audio",
                        source=Base64Source(
                            type="base64",
                            data="",
                            media_type="audio/pcm;rate=24000",
                        ),
                    )
            except Exception as e:
                print(f"[Gemini TTS Error] {e}")
                import traceback

                traceback.print_exc()
                # Clear text buffer for this message on error
                if msg_id in self._text_buffer:
                    del self._text_buffer[msg_id]

                return AudioBlock(
                    type="audio",
                    source=Base64Source(
                        type="base64",
                        data="",
                        media_type="audio/pcm;rate=24000",
                    ),
                )
        else:
            # Not the last chunk, return empty AudioBlock
            return AudioBlock(
                type="audio",
                source=Base64Source(
                    type="base64",
                    data="",
                    media_type="audio/pcm;rate=24000",
                ),
            )

    async def close(self) -> None:
        """Close the Gemini TTS model and clean up resources."""
        if not self._connected:
            return

        self._client = None
        self._connected = False
        self._text_buffer.clear()

    def is_initialized(self) -> bool:
        """Check if the TTS model is initialized.

        Returns:
            `bool`:
                True if initialized, False otherwise.
        """
        return self._connected
