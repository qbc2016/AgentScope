# -*- coding: utf-8 -*-
"""Kokoro TTS model implementation.

Kokoro is a lightweight, high-quality local TTS engine that supports
multiple languages with pre-built voice packs. No reference audio
is needed; it uses pre-set voice names.

Dependencies:
    pip install kokoro>=0.9.4 soundfile
    System: espeak-ng
"""
import asyncio
import base64
import io
import os
import threading
from typing import Any, AsyncGenerator, Literal

from pydantic import BaseModel, Field

from .._tts_base import TTSModelBase
from .._tts_model_card import TTSModelCard
from .._tts_response import TTSResponse
from ..._logging import logger
from ...credential import LocalTTSCredential
from ...message import DataBlock, Base64Source


_SAMPLE_RATE = 24000
_MEDIA_TYPE = "audio/wav"
_VOICE_LANGUAGE_CODES = frozenset(
    {"a", "b", "e", "f", "h", "i", "j", "p", "z"},
)
_LANGUAGE_DEPENDENCY_HINTS = {
    "j": ("Japanese", "ja"),
    "z": ("Mandarin Chinese", "zh"),
}


class KokoroTTSModel(TTSModelBase):
    """Kokoro local TTS model.

    Uses the Kokoro Python library to synthesize speech locally.
    Requires ``kokoro>=0.9.4``, ``soundfile``, and system-level
    ``espeak-ng``.
    """

    class Parameters(BaseModel):
        """Frontend-exposed parameters for Kokoro TTS."""

        voice: str = Field(
            default="af_heart",
            title="Voice",
            description=(
                "The voice name/pack to use. See Kokoro docs "
                "for available voices."
            ),
        )

        lang_code: Literal[
            "a",
            "b",
            "e",
            "f",
            "h",
            "i",
            "j",
            "p",
            "z",
        ] = Field(
            default="a",
            title="Language Code",
            description=(
                "Language code: 'a' for American English, "
                "'b' for British English, 'j' for Japanese, "
                "'z' for Mandarin Chinese, plus 'e', 'f', "
                "'h', 'i', and 'p' for Spanish, French, "
                "Hindi, Italian, and Brazilian Portuguese."
            ),
        )

        speed: float = Field(
            default=1.0,
            title="Speed",
            description="Speech speed multiplier.",
            ge=0.5,
            le=2.0,
        )

    type: Literal["kokoro_tts"] = "kokoro_tts"
    """The type of the TTS model."""

    _MODELS_DIR = os.path.join(
        os.path.dirname(__file__),
        "_kokoro_models",
    )

    realtime: bool = False

    _pipelines: dict[str, Any] = {}
    """Class-level pipeline cache keyed by ``"{lang_code}:{device}"``,
    so new model instances reuse already-loaded weights."""

    _lock = threading.Lock()
    """Serializes pipeline loading and inference across threads."""

    @classmethod
    def clear_cache(cls) -> None:
        """Release all cached pipelines so memory can be reclaimed.

        Safe to call from any thread; acquires the class lock.
        Subsequent :meth:`synthesize` calls will reload weights
        on demand.
        """
        with cls._lock:
            cls._pipelines.clear()

    @classmethod
    def list_models(
        cls,
        custom_yaml_dir: str | None = None,
    ) -> list[TTSModelCard]:
        """List Kokoro model cards from its YAML directory."""
        return super().list_models(
            custom_yaml_dir=custom_yaml_dir or cls._MODELS_DIR,
        )

    def __init__(
        self,
        credential: LocalTTSCredential,
        model: str = "kokoro",
        parameters: "KokoroTTSModel.Parameters | None" = None,
        stream: bool = False,
    ) -> None:
        """Initialize the Kokoro TTS model.

        Args:
            credential: The credential (LocalTTSCredential).
            model (`str`, defaults to ``"kokoro"``):
                The model name.
            parameters (`Parameters | None`, defaults to `None`):
                The TTS parameters.
            stream (`bool`, defaults to `False`):
                Whether to stream output chunks.
        """
        super().__init__(
            credential=credential,
            model=model,
            parameters=parameters,
            stream=stream,
        )

    def _get_pipeline(self) -> Any:
        """Lazily initialize and cache the Kokoro pipeline.

        .. note:: Must be called while holding ``_lock``.
        """
        lang_code = self._effective_lang_code()
        device = self.credential.device
        key = f"{lang_code}:{device}"
        if key not in self._pipelines:
            try:
                from kokoro import KPipeline
            except ImportError as e:
                raise ImportError(
                    "kokoro is required for KokoroTTSModel. "
                    "Install with: pip install 'kokoro>=0.9.4' "
                    "soundfile",
                ) from e
            try:
                self._pipelines[key] = KPipeline(
                    lang_code=lang_code,
                    device=device,
                )
            except ImportError as e:
                dependency_hint = _LANGUAGE_DEPENDENCY_HINTS.get(lang_code)
                if dependency_hint is None:
                    raise
                language, extra = dependency_hint
                missing_module = getattr(e, "name", None)
                missing_detail = (
                    f" (missing module: {missing_module!r})"
                    if missing_module
                    else ""
                )
                raise ImportError(
                    f"Kokoro {language} support could not load its "
                    f"optional dependencies{missing_detail}. "
                    f"Install with: pip install 'misaki[{extra}]>=0.9.4'",
                ) from e
        return self._pipelines[key]

    def _effective_lang_code(self) -> str:
        """Derive the language from a standard Kokoro voice name.

        Kokoro voice packs are prefixed with their language code.
        Using the prefix avoids invalid pairs such as ``zf_xiaobei``
        with the default American-English pipeline.
        """
        voice_prefix = self.parameters.voice.split(",", 1)[0].strip()[:1]
        if voice_prefix in _VOICE_LANGUAGE_CODES:
            return voice_prefix
        return self.parameters.lang_code

    def _synthesize_sync(self, text: str) -> str | None:
        """Run the blocking Kokoro synthesis in a worker thread.

        Returns the base64-encoded WAV audio, or ``None`` on failure.
        """
        try:
            import numpy as np
            import soundfile as sf
        except ImportError as e:
            raise ImportError(
                "numpy and soundfile are required for "
                "KokoroTTSModel. Install with: "
                "pip install numpy soundfile",
            ) from e

        with self._lock:
            pipeline = self._get_pipeline()
            try:
                all_audio = []
                generator = pipeline(
                    text,
                    voice=self.parameters.voice,
                    speed=self.parameters.speed,
                )
                for _, _, audio_chunk in generator:
                    if audio_chunk is not None:
                        all_audio.append(audio_chunk)

                if not all_audio:
                    logger.warning(
                        "Kokoro returned no audio data.",
                    )
                    return None

                audio = np.concatenate(all_audio)
            except Exception as e:
                logger.error(
                    "Kokoro TTS synthesis failed: %s",
                    e,
                )
                return None

        buf = io.BytesIO()
        sf.write(
            buf,
            audio,
            _SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    async def synthesize(
        self,
        text: str | None = None,
        **kwargs: Any,
    ) -> TTSResponse | AsyncGenerator[TTSResponse, None]:
        """Synthesize speech using Kokoro.

        The blocking inference runs in a worker thread via
        :func:`asyncio.to_thread` to keep the event loop responsive.

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

        audio_b64 = await asyncio.to_thread(self._synthesize_sync, text)
        if audio_b64 is None:
            return TTSResponse(content=None)

        return TTSResponse(
            content=DataBlock(
                source=Base64Source(
                    data=audio_b64,
                    media_type=_MEDIA_TYPE,
                ),
            ),
            is_last=True,
        )
