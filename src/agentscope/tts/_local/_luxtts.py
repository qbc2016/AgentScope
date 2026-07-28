# -*- coding: utf-8 -*-
"""LuxTTS model implementation.

LuxTTS is a fast, high-quality local TTS engine with voice cloning.
It requires a reference audio for speaker encoding.

Dependencies:
    pip install git+https://github.com/ysharma3501/LuxTTS.git
"""
import asyncio
import base64
import hashlib
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
from ._utils import cleanup_tempfile, decode_to_tempfile


_SAMPLE_RATE = 48000
_MEDIA_TYPE = "audio/wav"


class LuxTTSModel(TTSModelBase):
    """LuxTTS local TTS model.

    Uses the LuxTTS library for fast speech synthesis with voice
    cloning. Requires a reference audio file for speaker encoding.

    Requires ``luxtts`` package.
    """

    class Parameters(BaseModel):
        """Frontend-exposed parameters for LuxTTS."""

        reference_audio_path: str | None = Field(
            default=None,
            title="Reference Audio",
            description=(
                "Path to a reference audio file for speaker "
                "encoding (required for voice cloning)."
            ),
        )

        reference_audio_base64: str | None = Field(
            default=None,
            title="Reference Audio (Base64)",
            description=(
                "Base64-encoded reference audio for voice "
                "cloning. Used when audio is stored in the "
                "voice profile metadata."
            ),
        )

        num_steps: int = Field(
            default=4,
            title="Number of Steps",
            description=(
                "Number of diffusion steps. More steps = "
                "higher quality but slower."
            ),
            ge=1,
            le=50,
        )

    type: Literal["luxtts_tts"] = "luxtts_tts"
    """The type of the TTS model."""

    _MODELS_DIR = os.path.join(
        os.path.dirname(__file__),
        "_luxtts_models",
    )

    realtime: bool = False

    _models: dict[str, Any] = {}
    """Class-level model cache keyed by ``"{model_id}:{device}"``, so
    new model instances reuse already-loaded weights."""

    _prompt_cache: dict[str, Any] = {}
    """Class-level speaker embedding cache keyed by
    ``"{model_key}:{audio_hash}"``.  Avoids re-encoding the
    same reference audio on every request in multi-tenant
    scenarios where Voice Profiles are reused."""

    _lock = threading.Lock()
    """Serializes model loading and inference across threads."""

    @classmethod
    def clear_cache(cls) -> None:
        """Release all cached models so memory can be reclaimed.

        Safe to call from any thread; acquires the class lock.
        Subsequent :meth:`synthesize` calls will reload weights
        on demand.
        """
        with cls._lock:
            cls._models.clear()
            cls._prompt_cache.clear()

    @classmethod
    def list_models(
        cls,
        custom_yaml_dir: str | None = None,
    ) -> list[TTSModelCard]:
        """List LuxTTS model cards from its YAML directory."""
        return super().list_models(
            custom_yaml_dir=custom_yaml_dir or cls._MODELS_DIR,
        )

    def __init__(
        self,
        credential: LocalTTSCredential,
        model: str = "luxtts",
        parameters: "LuxTTSModel.Parameters | None" = None,
        stream: bool = False,
    ) -> None:
        """Initialize the LuxTTS model.

        Args:
            credential: The credential (LocalTTSCredential).
            model (`str`, defaults to ``"luxtts"``):
                The model name / model_id for LuxTTS.
            parameters (`Parameters | None`, defaults to `None`):
                The TTS parameters.
            stream (`bool`, defaults to `False`):
                Whether to stream output.
        """
        super().__init__(
            credential=credential,
            model=model,
            parameters=parameters,
            stream=stream,
        )

    def _get_model(self) -> Any:
        """Lazily initialize and cache the LuxTTS model.

        .. note:: Must be called while holding ``_lock``.
        """
        device = self.credential.device
        key = f"{self.model}:{device}"
        if key not in self._models:
            try:
                from zipvoice.luxvoice import LuxTTS
            except ImportError as e:
                raise ImportError(
                    "LuxTTS is required. Install with: "
                    "pip install "
                    "git+https://github.com/ysharma3501/"
                    "LuxTTS.git",
                ) from e
            self._models[key] = LuxTTS(
                model_path="YatharthS/LuxTTS",
                device=device,
            )
        return self._models[key]

    _PROMPT_CACHE_MAX = 64
    """Max number of cached speaker embeddings."""

    def _encode_prompt_cached(
        self,
        model: Any,
        ref_path: str,
        audio_b64: str | None,
    ) -> Any:
        """Encode reference audio, caching by content hash.

        Caching is only used when the actual encoding source is
        the base64 data (``audio_b64`` is not None), i.e. the
        Voice Profile flow.  When ``reference_audio_path`` was
        provided directly the file content may change between
        calls, so no caching is applied.

        Args:
            model: The LuxTTS model instance.
            ref_path (`str`): Path to the reference audio file.
            audio_b64 (`str | None`):
                The base64 string that was decoded to produce
                ``ref_path``, or ``None`` if ``ref_path`` was
                supplied directly by the caller.

        .. note:: Must be called while holding ``_lock``.
        """
        if audio_b64 is not None:
            digest = hashlib.sha256(
                audio_b64.encode("ascii"),
            ).hexdigest()
            device = self.credential.device
            key = f"{self.model}:{device}:{digest}"
            cached = self._prompt_cache.get(key)
            if cached is not None:
                return cached
            encoded = model.encode_prompt(ref_path)
            if len(self._prompt_cache) >= self._PROMPT_CACHE_MAX:
                oldest = next(iter(self._prompt_cache))
                del self._prompt_cache[oldest]
            self._prompt_cache[key] = encoded
            return encoded
        return model.encode_prompt(ref_path)

    def _synthesize_sync(self, text: str) -> str | None:
        """Run the blocking LuxTTS synthesis in a worker thread.

        Returns the base64-encoded WAV audio, or ``None`` on failure.
        """
        ref_path = self.parameters.reference_audio_path
        tmp_path: str | None = None
        b64_source: str | None = None
        if ref_path is None:
            b64_source = self.parameters.reference_audio_base64
            if b64_source:
                tmp_path = decode_to_tempfile(b64_source)
                ref_path = tmp_path
            else:
                logger.warning(
                    "LuxTTS requires reference_audio_path"
                    " for speaker encoding.",
                )
                return None

        try:
            try:
                import soundfile as sf
            except ImportError as e:
                raise ImportError(
                    "soundfile is required for LuxTTSModel.",
                ) from e

            with self._lock:
                model = self._get_model()
                try:
                    encoded = self._encode_prompt_cached(
                        model,
                        ref_path,
                        b64_source,
                    )
                    audio_np = model.generate_speech(
                        text,
                        encoded,
                        num_steps=self.parameters.num_steps,
                    )
                except Exception as e:
                    logger.error(
                        "LuxTTS synthesis failed: %s",
                        e,
                    )
                    return None
        finally:
            cleanup_tempfile(tmp_path)

        buf = io.BytesIO()
        sf.write(
            buf,
            audio_np,
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
        """Synthesize speech using LuxTTS.

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
