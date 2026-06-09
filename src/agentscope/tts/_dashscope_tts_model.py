# -*- coding: utf-8 -*-
"""DashScope TTS model implementation using MultiModalConversation API."""
import base64
import io
import wave
from typing import (
    Any,
    Literal,
    AsyncGenerator,
    Generator,
    TYPE_CHECKING,
)

from ._tts_base import TTSModelBase
from ._tts_response import TTSResponse
from .._utils._audio import _build_streaming_wav_header
from ..credential import DashScopeCredential
from ..message import Msg, DataBlock, Base64Source
from ..types import JSONSerializableObject

if TYPE_CHECKING:
    from dashscope.api_entities.dashscope_response import (
        MultiModalConversationResponse,
    )


# DashScope TTS returns raw PCM (24kHz, mono, 16-bit). We wrap it as WAV
# on the way out so the frontend can play it: streaming deltas get a
# streaming WAV header on the first chunk; non-streaming returns a
# self-contained fixed-size WAV.
_TTS_SAMPLE_RATE = 24000
_TTS_CHANNELS = 1
_TTS_BITS_PER_SAMPLE = 16
_DEFAULT_MEDIA_TYPE = "audio/wav"


class DashScopeTTSModel(TTSModelBase):
    """DashScope TTS model implementation using the MultiModalConversation
    API. For more details please see the `official document
    <https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=2879134>`_.
    """

    supports_streaming_input: bool = False
    """Whether the model supports streaming input. DashScope's standard TTS
    is non-realtime; for streaming-input TTS use the realtime variant."""

    def __init__(
        self,
        credential: DashScopeCredential,
        model: str = "qwen3-tts-flash",
        voice: Literal["Cherry", "Serena", "Ethan", "Chelsie"]
        | str = "Cherry",
        language_type: str = "Auto",
        stream: bool = True,
        generate_kwargs: dict[str, JSONSerializableObject] | None = None,
    ) -> None:
        """Initialize the DashScope TTS model.

        .. note:: More details about the parameters, such as ``model``,
         ``voice``, and ``language_type`` can be found in the
         `official document
         <https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=2879134>`_.

        Args:
            credential (`DashScopeCredential`):
                The DashScope credential used to authenticate the API call.
            model (`str`, defaults to ``"qwen3-tts-flash"``):
                The TTS model name. Supported models include
                ``qwen3-tts-flash``, ``qwen-tts``, etc.
            voice (`Literal["Cherry", "Serena", "Ethan", "Chelsie"] | str`, \
            defaults to ``"Cherry"``):
                The voice to use. Supported voices include ``"Cherry"``,
                ``"Serena"``, ``"Ethan"``, ``"Chelsie"``, etc.
            language_type (`str`, defaults to ``"Auto"``):
                The language type. Should match the text language for
                correct pronunciation and natural intonation.
            stream (`bool`, defaults to `True`):
                Whether to use streaming output. When `True`,
                :meth:`synthesize` returns an async generator yielding
                ``TTSResponse`` chunks; when `False`, it returns a single
                aggregated ``TTSResponse``.
            generate_kwargs (`dict[str, JSONSerializableObject] | None`, \
            optional):
                Additional keyword arguments passed through to the DashScope
                TTS API (e.g. ``temperature``, ``seed``).
        """
        super().__init__(credential=credential, model=model, stream=stream)

        self.voice = voice
        self.language_type = language_type
        self.generate_kwargs = generate_kwargs or {}

    async def synthesize(
        self,
        msg: Msg | None = None,
        **kwargs: Any,
    ) -> TTSResponse | AsyncGenerator[TTSResponse, None]:
        """Call the DashScope TTS API to synthesize speech from text.

        Args:
            msg (`Msg | None`, optional):
                The message whose text will be synthesized.
            **kwargs (`Any`):
                Additional keyword arguments to pass to the TTS API call.

        Returns:
            `TTSResponse | AsyncGenerator[TTSResponse, None]`:
                A single ``TTSResponse`` when ``stream=False``, or an async
                generator yielding ``TTSResponse`` chunks when ``stream=True``.
        """
        if msg is None:
            return TTSResponse(content=None)

        text = msg.get_text_content()

        import dashscope

        response = dashscope.MultiModalConversation.call(
            model=self.model,
            api_key=self.credential.api_key.get_secret_value(),
            text=text,
            voice=self.voice,
            language_type=self.language_type,
            stream=True,
            **self.generate_kwargs,
            **kwargs,
        )

        if self.stream:
            return self._parse_into_async_generator(response)

        audio_bytes = bytearray()
        for chunk in response:
            if chunk.output is not None:
                audio = chunk.output.audio
                if audio and audio.data:
                    audio_bytes += base64.b64decode(audio.data)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(_TTS_CHANNELS)
            wav.setsampwidth(_TTS_BITS_PER_SAMPLE // 8)
            wav.setframerate(_TTS_SAMPLE_RATE)
            wav.writeframes(bytes(audio_bytes))

        return TTSResponse(
            content=DataBlock(
                source=Base64Source(
                    data=base64.b64encode(buf.getvalue()).decode("ascii"),
                    media_type=_DEFAULT_MEDIA_TYPE,
                ),
            ),
        )

    @staticmethod
    async def _parse_into_async_generator(
        response: Generator["MultiModalConversationResponse", None, None],
    ) -> AsyncGenerator[TTSResponse, None]:
        """Parse the streaming TTS response into an async generator.

        Each yielded ``TTSResponse`` carries an **incremental** WAV chunk:
        the first chunk is prefixed with a streaming WAV/RIFF header so the
        frontend can start playback immediately (without waiting for
        end-of-stream); subsequent chunks are raw PCM bytes appended to that
        open stream. The final response has ``is_last=True``.

        Args:
            response (`Generator[MultiModalConversationResponse, None, None]`):
                The streaming response from the DashScope TTS API.

        Yields:
            `TTSResponse`:
                A ``TTSResponse`` for each incremental audio chunk; the final
                response has ``is_last=True``.
        """
        pending: TTSResponse | None = None
        header_sent = False
        for chunk in response:
            if chunk.output is None:
                continue
            audio = chunk.output.audio
            if not audio or not audio.data:
                continue
            delta_bytes = base64.b64decode(audio.data)
            if not delta_bytes:
                continue
            if not header_sent:
                payload = (
                    _build_streaming_wav_header(
                        sample_rate=_TTS_SAMPLE_RATE,
                        channels=_TTS_CHANNELS,
                        bits_per_sample=_TTS_BITS_PER_SAMPLE,
                    )
                    + delta_bytes
                )
                header_sent = True
            else:
                payload = delta_bytes
            if pending is not None:
                yield pending
            pending = TTSResponse(
                content=DataBlock(
                    source=Base64Source(
                        data=base64.b64encode(payload).decode("ascii"),
                        media_type=_DEFAULT_MEDIA_TYPE,
                    ),
                ),
                is_last=False,
            )

        if pending is not None:
            pending.is_last = True
            yield pending
        else:
            yield TTSResponse(content=None, is_last=True)
