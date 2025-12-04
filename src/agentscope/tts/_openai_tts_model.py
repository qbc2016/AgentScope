# -*- coding: utf-8 -*-
"""OpenAI TTS model implementation."""
import base64
from typing import TYPE_CHECKING, Any, Literal, AsyncGenerator

from ._tts_base import TTSModelBase
from ._tts_response import TTSResponse
from ..message import Msg, AudioBlock, Base64Source
from ..types import JSONSerializableObject

if TYPE_CHECKING:
    from openai import HttpxBinaryResponseContent
else:
    HttpxBinaryResponseContent = "openai.HttpxBinaryResponseContent"


class OpenAITTSModel(TTSModelBase):
    """OpenAI TTS model implementation.
    For more details, please see the `official document
    <https://platform.openai.com/docs/api-reference/audio>`_.
    """

    # This model does not support streaming input (requires complete text)
    supports_streaming_input: bool = False

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini-tts",
        voice: Literal["alloy", "ash", "ballad", "coral"] | str = "alloy",
        stream: bool = True,
        client_kwargs: dict | None = None,
        generate_kwargs: dict[str, JSONSerializableObject] | None = None,
    ) -> None:
        """Initialize the OpenAI TTS model.

        .. note::
            More details about the parameters, such as `model_name` and
            `voice` can be found in the `official document
            <https://platform.openai.com/docs/api-reference/audio/createSpeech>`_.

        Args:
            api_key (`str`):
                The OpenAI API key.
            model_name (`str`,  defaults to "gpt-4o-mini-tts"):
                The TTS model name. Supported models are "gpt-4o-mini-tts",
                "tts-1", etc.
            voice (`Literal["alloy", "ash", "ballad", "coral"] | str `,
             defaults to "alloy"):
                The voice to use. Supported voices are "alloy", "ash",
                "ballad", "coral", etc.
            client_kwargs (`dict | None`, default `None`):
                The extra keyword arguments to initialize the OpenAI client.
            generate_kwargs (`dict[str, JSONSerializableObject] | None`, \
             optional):
               The extra keyword arguments used in OpenAI API generation,
               e.g. `temperature`, `seed`.
        """
        super().__init__(model_name=model_name, stream=stream)

        self.api_key = api_key
        self.voice = voice
        self.stream = stream

        import openai

        self._client = openai.AsyncOpenAI(
            api_key=self.api_key,
            **client_kwargs or {},
        )

        # Text buffer for each message to accumulate text before synthesis
        # Key is msg.id, value is the accumulated text
        self.generate_kwargs = generate_kwargs or {}

    async def synthesize(
        self,
        msg: Msg | None = None,
        **kwargs: Any,
    ) -> TTSResponse | AsyncGenerator[TTSResponse, None]:
        """Append text to be synthesized and return TTS response.

        Args:
            msg (`Msg | None`, optional):
                The message to be synthesized.
            **kwargs (`Any`):
                Additional keyword arguments to pass to the TTS API call.

        Returns:
            `TTSResponse | AsyncGenerator[TTSResponse, None]`:
                The TTSResponse object in non-streaming mode, or an async
                generator yielding TTSResponse objects in streaming mode.
        """
        if msg is None:
            return TTSResponse(content=[])

        text = msg.get_text_content()

        if text:
            response = await self._client.audio.speech.create(
                model=self.model_name,
                voice=self.voice,
                input=text,
                **self.generate_kwargs,
                **kwargs,
            )

            if self.stream:
                return await self._parse_into_async_generator(response)

            audio_base64 = base64.b64encode(response.content).decode(
                "utf-8",
            )
            return TTSResponse(
                content=[
                    AudioBlock(
                        type="audio",
                        source=Base64Source(
                            type="base64",
                            data=audio_base64,
                            media_type="audio/mp3",
                        ),
                    ),
                ],
            )

        return TTSResponse(content=[])

    @staticmethod
    async def _parse_into_async_generator(
        response: HttpxBinaryResponseContent,
    ) -> AsyncGenerator[TTSResponse, None]:
        """Parse the streaming response into an async generator of TTSResponse.

        Args:
            response (`HttpxBinaryResponseContent`):
                The streaming response from OpenAI TTS API.

        Yields:
            `TTSResponse`:
                The TTSResponse object containing audio blocks.
        """
        # TODO: @qbc Implement streaming response parsing
