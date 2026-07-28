# -*- coding: utf-8 -*-
"""TADA TTS model implementation.

TADA (Text-Aligned Diffusion Audio) is a local TTS engine from
Hume AI that supports voice cloning with reference audio and text.

Dependencies:
    pip install hume-tada
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
from ._utils import cleanup_tempfile, decode_to_tempfile


_SAMPLE_RATE = 24000
_MEDIA_TYPE = "audio/wav"


class TadaTTSModel(TTSModelBase):
    """TADA local TTS model.

    Uses the Hume TADA library for speech synthesis with voice
    cloning. Requires both a reference audio file and the
    transcript of that reference for optimal results.

    Requires ``hume-tada`` package.
    """

    class Parameters(BaseModel):
        """Frontend-exposed parameters for TADA TTS."""

        reference_audio_path: str | None = Field(
            default=None,
            title="Reference Audio",
            description=(
                "Path to a reference audio file for voice "
                "cloning (required)."
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

        reference_text: str | None = Field(
            default=None,
            title="Reference Text",
            description=(
                "Transcript of the reference audio. "
                "Required for best quality."
            ),
        )

    type: Literal["tada_tts"] = "tada_tts"
    """The type of the TTS model."""

    _MODELS_DIR = os.path.join(
        os.path.dirname(__file__),
        "_tada_models",
    )

    realtime: bool = False

    _models: dict[str, tuple[Any, Any]] = {}
    """Class-level (encoder, model) cache keyed by device, so new
    model instances reuse already-loaded weights."""

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

    @classmethod
    def list_models(
        cls,
        custom_yaml_dir: str | None = None,
    ) -> list[TTSModelCard]:
        """List TADA model cards from its YAML directory."""
        return super().list_models(
            custom_yaml_dir=custom_yaml_dir or cls._MODELS_DIR,
        )

    def __init__(
        self,
        credential: LocalTTSCredential,
        model: str = "tada",
        parameters: "TadaTTSModel.Parameters | None" = None,
        stream: bool = False,
    ) -> None:
        """Initialize the TADA TTS model.

        Args:
            credential: The credential (LocalTTSCredential).
            model (`str`, defaults to ``"tada"``):
                The model name.
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

    def _load_models(self) -> tuple[Any, Any]:
        """Lazily load and cache the TADA encoder and model.

        .. note:: Must be called while holding ``_lock``.
        """
        device = self.credential.device
        if device not in self._models:
            try:
                from hume_tada import (
                    Encoder,
                    TadaForCausalLM,
                )
            except ImportError as e:
                raise ImportError(
                    "hume-tada is required for TadaTTSModel. "
                    "Install with: pip install hume-tada",
                ) from e
            encoder = Encoder.from_pretrained(device=device)
            tada_model = TadaForCausalLM.from_pretrained(
                device=device,
            )
            self._models[device] = (encoder, tada_model)
        return self._models[device]

    def _synthesize_sync(self, text: str) -> str | None:
        """Run the blocking TADA synthesis in a worker thread.

        Returns the base64-encoded WAV audio, or ``None`` on failure.
        """
        ref_path = self.parameters.reference_audio_path
        tmp_path: str | None = None
        if ref_path is None:
            if self.parameters.reference_audio_base64:
                tmp_path = decode_to_tempfile(
                    self.parameters.reference_audio_base64,
                )
                ref_path = tmp_path
            else:
                logger.warning(
                    "TADA requires reference_audio_path.",
                )
                return None

        try:
            try:
                import soundfile as sf
                import torch
            except ImportError as e:
                raise ImportError(
                    "soundfile and torch are required for TadaTTSModel.",
                ) from e

            with self._lock:
                encoder, tada_model = self._load_models()
                try:
                    ref_text_list = []
                    if self.parameters.reference_text:
                        ref_text_list = [
                            self.parameters.reference_text,
                        ]

                    prompt = encoder(
                        ref_path,
                        text=ref_text_list or None,
                    )
                    audio_tensor = tada_model.generate(
                        prompt,
                        text=text,
                    )

                    if isinstance(audio_tensor, torch.Tensor):
                        audio_np = audio_tensor.squeeze().cpu().numpy()
                    else:
                        audio_np = audio_tensor
                except Exception as e:
                    logger.error("TADA TTS synthesis failed: %s", e)
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
        """Synthesize speech using TADA.

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
