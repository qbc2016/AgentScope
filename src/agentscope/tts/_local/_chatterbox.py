# -*- coding: utf-8 -*-
"""Chatterbox TTS model implementation.

Chatterbox is a local TTS engine supporting voice cloning with
three variants: English, Multilingual, and Turbo.

Dependencies:
    pip install chatterbox-tts
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


_MEDIA_TYPE = "audio/wav"


class ChatterboxTTSModel(TTSModelBase):
    """Chatterbox local TTS model.

    Supports three variants:
    - English: Voice cloning with a reference audio prompt
    - Multilingual: Multi-language support with language_id
    - Turbo: Fast synthesis without reference audio

    Requires ``chatterbox-tts`` package.
    """

    class Parameters(BaseModel):
        """Frontend-exposed parameters for Chatterbox TTS."""

        variant: Literal[
            "english",
            "multilingual",
            "turbo",
        ] = Field(
            default="english",
            title="Variant",
            description=(
                "The Chatterbox variant: 'english' for "
                "voice cloning, 'multilingual' for "
                "multi-language, 'turbo' for fast synthesis."
            ),
        )

        reference_audio_path: str | None = Field(
            default=None,
            title="Reference Audio",
            description=(
                "Path to a reference audio file for voice "
                "cloning (English variant). Not needed for "
                "Turbo variant."
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

        language_id: str | None = Field(
            default=None,
            title="Language ID",
            description=(
                "Language identifier for the multilingual "
                "variant (e.g. 'en', 'es', 'fr', 'de', "
                "'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs',"
                " 'ar', 'zh-cn', 'ja', 'hu', 'ko')."
            ),
        )

        exaggeration: float = Field(
            default=0.5,
            title="Exaggeration",
            description=(
                "Controls expressiveness. Higher values "
                "produce more animated speech."
            ),
            ge=0.0,
            le=1.0,
        )

        cfg_weight: float = Field(
            default=0.5,
            title="CFG Weight",
            description=(
                "Classifier-free guidance weight. Higher "
                "values increase fidelity to the reference."
            ),
            ge=0.0,
            le=1.0,
        )

    type: Literal["chatterbox_tts"] = "chatterbox_tts"
    """The type of the TTS model."""

    _MODELS_DIR = os.path.join(
        os.path.dirname(__file__),
        "_chatterbox_models",
    )

    realtime: bool = False

    _model_cache: dict[str, Any] = {}
    """Class-level model cache keyed by ``"{variant}:{device}"``, so
    new model instances reuse already-loaded weights."""

    _default_conds: dict[str, Any] = {}
    """Snapshot of each cached model's built-in voice conditionals
    (``conds``) taken at load time, keyed like :attr:`_model_cache`."""

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
            cls._model_cache.clear()
            cls._default_conds.clear()

    @classmethod
    def list_models(
        cls,
        custom_yaml_dir: str | None = None,
    ) -> list[TTSModelCard]:
        """List Chatterbox model cards from its YAML directory."""
        return super().list_models(
            custom_yaml_dir=custom_yaml_dir or cls._MODELS_DIR,
        )

    def __init__(
        self,
        credential: LocalTTSCredential,
        model: str = "chatterbox",
        parameters: "ChatterboxTTSModel.Parameters | None" = None,
        stream: bool = False,
    ) -> None:
        """Initialize the Chatterbox TTS model.

        Args:
            credential: The credential (LocalTTSCredential).
            model (`str`, defaults to ``"chatterbox"``):
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

    @staticmethod
    def _patch_perth_watermarker() -> None:
        """Patch perth if PerthImplicitWatermarker is unavailable.

        setuptools >= 70 removed pkg_resources, which breaks the
        perth import chain. Fall back to DummyWatermarker so TTS
        still works (audio quality is unaffected).
        """
        try:
            import perth
        except ImportError:
            return

        if perth.PerthImplicitWatermarker is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker

    def _get_model(self, variant: str) -> Any:
        """Lazily load and cache the Chatterbox model.

        .. note:: Must be called while holding ``_lock``.
        """
        device = self.credential.device
        key = f"{variant}:{device}"
        if key not in self._model_cache:
            self._patch_perth_watermarker()
            try:
                if variant == "english":
                    from chatterbox.tts import (
                        ChatterboxTTS,
                    )

                    m = ChatterboxTTS.from_pretrained(
                        device=device,
                    )
                elif variant == "multilingual":
                    from chatterbox.tts import (
                        ChatterboxMultilingualTTS,
                    )

                    m = ChatterboxMultilingualTTS.from_pretrained(
                        t3_model="v3",
                        device=device,
                    )
                elif variant == "turbo":
                    from chatterbox.tts import (
                        ChatterboxTurboTTS,
                    )

                    m = ChatterboxTurboTTS.from_pretrained(
                        device=device,
                    )
                else:
                    raise ValueError(
                        f"Unknown variant: {variant}",
                    )
                self._model_cache[key] = m
                self._default_conds[key] = getattr(m, "conds", None)
            except ImportError as e:
                raise ImportError(
                    "chatterbox-tts is required. "
                    "Install with: pip install chatterbox-tts",
                ) from e
        return self._model_cache[key]

    def _restore_default_voice(self, model: Any, variant: str) -> None:
        """Restore the built-in voice conditionals on a cached model.

        ``generate`` without ``audio_prompt_path`` reuses the
        ``conds`` left over from the previous call, which may hold
        another session's cloned voice. Restoring the snapshot taken
        at load time prevents that voice from leaking across
        requests.
        """
        device = self.credential.device
        key = f"{variant}:{device}"
        model.conds = self._default_conds.get(key)

    def _synthesize_sync(self, text: str) -> str | None:
        """Run the blocking Chatterbox synthesis in a worker thread.

        Returns the base64-encoded WAV audio, or ``None`` on failure.
        """
        variant = self.parameters.variant
        ref_path = self.parameters.reference_audio_path
        tmp_path: str | None = None

        if ref_path is None and self.parameters.reference_audio_base64:
            tmp_path = decode_to_tempfile(
                self.parameters.reference_audio_base64,
            )
            ref_path = tmp_path

        if variant == "english" and ref_path is None:
            logger.warning(
                "Chatterbox English variant requires "
                "reference audio for voice cloning.",
            )
            return None

        try:
            try:
                import soundfile as sf
                import torch
            except ImportError as e:
                raise ImportError(
                    "soundfile and torch are required for "
                    "ChatterboxTTSModel.",
                ) from e

            with self._lock:
                model = self._get_model(variant)
                try:
                    if ref_path is None:
                        self._restore_default_voice(
                            model,
                            variant,
                        )

                    if variant == "english":
                        audio_tensor = model.generate(
                            text,
                            audio_prompt_path=ref_path,
                            exaggeration=(self.parameters.exaggeration),
                            cfg_weight=(self.parameters.cfg_weight),
                        )
                    elif variant == "multilingual":
                        gen_kwargs: dict[str, Any] = {
                            "language_id": (
                                self.parameters.language_id or "en"
                            ),
                        }
                        if ref_path is not None:
                            gen_kwargs["audio_prompt_path"] = ref_path
                        audio_tensor = model.generate(
                            text,
                            **gen_kwargs,
                        )
                    else:
                        if ref_path is not None:
                            audio_tensor = model.generate(
                                text,
                                audio_prompt_path=ref_path,
                            )
                        else:
                            audio_tensor = model.generate(
                                text,
                            )

                    if isinstance(
                        audio_tensor,
                        torch.Tensor,
                    ):
                        audio_np = audio_tensor.squeeze().cpu().numpy()
                    else:
                        audio_np = audio_tensor

                    sample_rate = getattr(
                        model,
                        "sr",
                        24000,
                    )
                except Exception as e:
                    logger.error(
                        "Chatterbox TTS failed: %s",
                        e,
                    )
                    return None
                finally:
                    self._restore_default_voice(
                        model,
                        variant,
                    )
        finally:
            cleanup_tempfile(tmp_path)

        buf = io.BytesIO()
        sf.write(
            buf,
            audio_np,
            sample_rate,
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
        """Synthesize speech using Chatterbox.

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
