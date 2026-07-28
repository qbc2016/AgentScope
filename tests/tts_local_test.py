# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for local TTS engines.

Covers:
  * KokoroTTSModel: None input, ImportError handling.
  * ChatterboxTTSModel: None input, variant validation,
    voice restore after synthesis, clear_cache.
  * LuxTTSModel: None input, missing reference audio,
    prompt cache, clear_cache.
  * TadaTTSModel: None input, missing reference audio.
  * _resolve_tts_class: model-name based class resolution.
  * _utils: decode_to_tempfile / cleanup_tempfile.
  * _enrich_from_profile: voice profile enrichment.
"""
import base64
import os
import sys
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from agentscope.app._service._tts_model import (
    _enrich_from_profile,
    _resolve_tts_class,
)
from agentscope.credential import LocalTTSCredential
from agentscope.tts import (
    KokoroTTSModel,
    ChatterboxTTSModel,
    LuxTTSModel,
    TadaTTSModel,
    TTSResponse,
)
from agentscope.tts._local._utils import (
    cleanup_tempfile,
    decode_to_tempfile,
)


class TestKokoroTTSModel(IsolatedAsyncioTestCase):
    """Unit tests for KokoroTTSModel."""

    def _make_model(self) -> object:
        """Create a KokoroTTSModel with test credential."""
        cred = LocalTTSCredential(device="cpu")
        return KokoroTTSModel(credential=cred)

    async def test_none_text_returns_empty(self) -> None:
        """None text returns TTSResponse with None content."""
        model = self._make_model()
        result = await model.synthesize(None)
        self.assertIsInstance(result, TTSResponse)
        self.assertIsNone(result.content)

    async def test_parameters_defaults(self) -> None:
        """Default parameters are correct."""
        model = self._make_model()
        self.assertEqual(
            model.parameters.model_dump(),
            {
                "voice": "af_heart",
                "lang_code": "a",
                "speed": 1.0,
            },
        )


class TestChatterboxTTSModel(IsolatedAsyncioTestCase):
    """Unit tests for ChatterboxTTSModel."""

    def _make_model(self, **kwargs: object) -> object:
        """Create a ChatterboxTTSModel with test credential."""
        cred = LocalTTSCredential(device="cpu")
        return ChatterboxTTSModel(credential=cred, **kwargs)

    async def test_none_text_returns_empty(self) -> None:
        """None text returns TTSResponse with None content."""
        model = self._make_model()
        result = await model.synthesize(None)
        self.assertIsInstance(result, TTSResponse)
        self.assertIsNone(result.content)

    async def test_english_without_reference_returns_empty(self) -> None:
        """English variant without ref audio returns None."""
        model = self._make_model()
        model.parameters.variant = "english"
        model.parameters.reference_audio_path = None
        model._model_cache["english:cpu"] = MagicMock()
        self.addCleanup(model._model_cache.pop, "english:cpu", None)
        result = await model.synthesize("Hello")
        self.assertIsInstance(result, TTSResponse)
        self.assertIsNone(result.content)

    async def test_restore_default_voice(self) -> None:
        """Leftover cloned-voice conds are reset to the load-time
        snapshot, so a previous request's voice cannot leak."""
        model = self._make_model()
        sentinel = object()
        model._default_conds["turbo:cpu"] = sentinel
        self.addCleanup(model._default_conds.pop, "turbo:cpu", None)
        cached = MagicMock()
        cached.conds = "leaked-cloned-voice"
        model._restore_default_voice(cached, "turbo")
        self.assertIs(cached.conds, sentinel)

    async def test_parameters_defaults(self) -> None:
        """Default parameters are correct."""
        model = self._make_model()
        self.assertEqual(
            model.parameters.model_dump(),
            {
                "variant": "english",
                "reference_audio_path": None,
                "reference_audio_base64": None,
                "language_id": None,
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
            },
        )


class TestLuxTTSModel(IsolatedAsyncioTestCase):
    """Unit tests for LuxTTSModel."""

    def _make_model(self) -> object:
        """Create a LuxTTSModel with test credential."""
        cred = LocalTTSCredential(device="cpu")
        return LuxTTSModel(credential=cred)

    async def test_none_text_returns_empty(self) -> None:
        """None text returns TTSResponse with None content."""
        model = self._make_model()
        result = await model.synthesize(None)
        self.assertIsInstance(result, TTSResponse)
        self.assertIsNone(result.content)

    async def test_missing_reference_returns_empty(self) -> None:
        """Missing reference audio returns None content."""
        model = self._make_model()
        result = await model.synthesize("Hello")
        self.assertIsInstance(result, TTSResponse)
        self.assertIsNone(result.content)


class TestResolveTTSClass(TestCase):
    """Unit tests for ``_resolve_tts_class`` with local TTS classes."""

    _CLASSES: list = [
        KokoroTTSModel,
        ChatterboxTTSModel,
        LuxTTSModel,
        TadaTTSModel,
    ]

    def test_known_models_resolve(self) -> None:
        """Each local model name resolves to its own class."""
        expected = {
            "kokoro": KokoroTTSModel,
            "chatterbox": ChatterboxTTSModel,
            "luxtts": LuxTTSModel,
            "tada": TadaTTSModel,
        }
        resolved = {
            name: _resolve_tts_class(self._CLASSES, name) for name in expected
        }
        self.assertEqual(resolved, expected)

    def test_unknown_model_raises(self) -> None:
        """Unknown model with multiple classes raises 400."""
        with self.assertRaises(HTTPException) as ctx:
            _resolve_tts_class(self._CLASSES, "no-such-model")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_single_class_falls_back(self) -> None:
        """A single-class credential keeps the fallback behavior."""
        cls = _resolve_tts_class([KokoroTTSModel], "no-such-model")
        self.assertIs(cls, KokoroTTSModel)


class TestTadaTTSModel(IsolatedAsyncioTestCase):
    """Unit tests for TadaTTSModel."""

    def _make_model(self) -> object:
        """Create a TadaTTSModel with test credential."""
        cred = LocalTTSCredential(device="cpu")
        return TadaTTSModel(credential=cred)

    async def test_none_text_returns_empty(self) -> None:
        """None text returns TTSResponse with None content."""
        model = self._make_model()
        result = await model.synthesize(None)
        self.assertIsInstance(result, TTSResponse)
        self.assertIsNone(result.content)

    async def test_missing_reference_returns_empty(self) -> None:
        """Missing reference audio returns None content."""
        model = self._make_model()
        result = await model.synthesize("Hello")
        self.assertIsInstance(result, TTSResponse)
        self.assertIsNone(result.content)


class TestDecodeToTempfile(TestCase):
    """Unit tests for decode_to_tempfile / cleanup_tempfile."""

    def test_roundtrip(self) -> None:
        """Encode -> decode -> read back matches original."""
        payload = b"RIFF\x00\x00\x00\x00WAVEfmt "
        b64 = base64.b64encode(payload).decode("ascii")
        path = decode_to_tempfile(b64)
        try:
            self.assertTrue(os.path.isfile(path))
            with open(path, "rb") as f:
                self.assertEqual(f.read(), payload)
        finally:
            cleanup_tempfile(path)
        self.assertFalse(os.path.exists(path))

    def test_cleanup_none_is_noop(self) -> None:
        """cleanup_tempfile(None) does not raise."""
        cleanup_tempfile(None)

    def test_cleanup_missing_file_is_noop(self) -> None:
        """cleanup_tempfile on missing path is silent."""
        cleanup_tempfile("/tmp/__no_such_file__.wav")


class TestClearCache(TestCase):
    """Verify clear_cache empties class-level caches."""

    def test_kokoro_clear_cache(self) -> None:
        """KokoroTTSModel.clear_cache empties _pipelines."""
        KokoroTTSModel._pipelines["test:cpu"] = "dummy"
        self.addCleanup(
            KokoroTTSModel._pipelines.pop,
            "test:cpu",
            None,
        )
        KokoroTTSModel.clear_cache()
        self.assertEqual(KokoroTTSModel._pipelines, {})

    def test_chatterbox_clear_cache(self) -> None:
        """ChatterboxTTSModel.clear_cache empties caches."""
        ChatterboxTTSModel._model_cache["t:cpu"] = "m"
        ChatterboxTTSModel._default_conds["t:cpu"] = "c"
        self.addCleanup(
            ChatterboxTTSModel._model_cache.pop,
            "t:cpu",
            None,
        )
        self.addCleanup(
            ChatterboxTTSModel._default_conds.pop,
            "t:cpu",
            None,
        )
        ChatterboxTTSModel.clear_cache()
        self.assertEqual(
            ChatterboxTTSModel._model_cache,
            {},
        )
        self.assertEqual(
            ChatterboxTTSModel._default_conds,
            {},
        )

    def test_luxtts_clear_cache(self) -> None:
        """LuxTTSModel.clear_cache empties both caches."""
        LuxTTSModel._models["t:cpu"] = "m"
        LuxTTSModel._prompt_cache["t:cpu:abc"] = "e"
        self.addCleanup(
            LuxTTSModel._models.pop,
            "t:cpu",
            None,
        )
        self.addCleanup(
            LuxTTSModel._prompt_cache.pop,
            "t:cpu:abc",
            None,
        )
        LuxTTSModel.clear_cache()
        self.assertEqual(LuxTTSModel._models, {})
        self.assertEqual(LuxTTSModel._prompt_cache, {})

    def test_tada_clear_cache(self) -> None:
        """TadaTTSModel.clear_cache empties _models."""
        TadaTTSModel._models["cpu"] = "m"
        self.addCleanup(
            TadaTTSModel._models.pop,
            "cpu",
            None,
        )
        TadaTTSModel.clear_cache()
        self.assertEqual(TadaTTSModel._models, {})


class TestChatterboxVoiceRestore(IsolatedAsyncioTestCase):
    """Verify conds are restored after synthesis."""

    async def test_finally_restores_after_error(self) -> None:
        """After a failed synthesis, model.conds is reset."""
        mock_sf = MagicMock()
        mock_torch = MagicMock()
        mock_torch.Tensor = type("Tensor", (), {})

        with patch.dict(
            sys.modules,
            {"soundfile": mock_sf, "torch": mock_torch},
        ):
            cred = LocalTTSCredential(device="cpu")
            model = ChatterboxTTSModel(credential=cred)

            sentinel = object()
            cached = MagicMock()
            cached.sr = 24000
            cached.conds = "before"
            cached.generate.side_effect = RuntimeError(
                "boom",
            )

            model._model_cache["english:cpu"] = cached
            model._default_conds["english:cpu"] = sentinel
            self.addCleanup(
                model._model_cache.pop,
                "english:cpu",
                None,
            )
            self.addCleanup(
                model._default_conds.pop,
                "english:cpu",
                None,
            )

            model.parameters.reference_audio_path = "/f.wav"
            result = await model.synthesize("test")
            self.assertIsNone(result.content)
            self.assertIs(cached.conds, sentinel)


class TestEnrichFromProfile(IsolatedAsyncioTestCase):
    """Unit tests for _enrich_from_profile."""

    async def test_merges_metadata(self) -> None:
        """Profile metadata is merged into params."""
        storage = AsyncMock()
        profile = MagicMock()
        profile.data.metadata = {
            "reference_audio_base64": "AAAA",
            "reference_text": "hello",
        }
        storage.get_voice_profile = AsyncMock(
            return_value=profile,
        )
        params: dict = {}
        result = await _enrich_from_profile(
            storage,
            "user1",
            "profile1",
            params,
        )
        self.assertEqual(
            result,
            {
                "reference_audio_base64": "AAAA",
                "reference_text": "hello",
            },
        )

    async def test_explicit_params_precedence(self) -> None:
        """Explicit params are not overwritten."""
        storage = AsyncMock()
        profile = MagicMock()
        profile.data.metadata = {
            "reference_audio_base64": "BBBB",
            "reference_text": "world",
        }
        storage.get_voice_profile = AsyncMock(
            return_value=profile,
        )
        params = {"reference_audio_base64": "EXISTING"}
        result = await _enrich_from_profile(
            storage,
            "user1",
            "profile1",
            params,
        )
        self.assertEqual(
            result,
            {
                "reference_audio_base64": "EXISTING",
                "reference_text": "world",
            },
        )

    async def test_missing_profile_is_noop(self) -> None:
        """Non-existent profile leaves params unchanged."""
        storage = AsyncMock()
        storage.get_voice_profile = AsyncMock(
            return_value=None,
        )
        params = {"key": "val"}
        result = await _enrich_from_profile(
            storage,
            "user1",
            "missing",
            params,
        )
        self.assertEqual(result, {"key": "val"})
