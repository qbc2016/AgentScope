# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for Voice Profile storage model and constants.

Covers:
  * VoiceProfileData and VoiceProfileRecord model creation.
  * ENGINE_TO_CREDENTIAL_TYPE mapping completeness.
  * ENGINE_VOICE_CLONING mapping.
"""
import typing
from unittest import TestCase
from unittest.mock import patch

from pydantic import ValidationError

from agentscope.app._router._voice_profile import (
    _voice_profile_summary,
)
from agentscope.app.storage._model import (
    ENGINE_TO_CREDENTIAL_TYPE,
    VoiceProfileData,
    VoiceProfileRecord,
)
from agentscope.app.storage._model._voice_profile import (
    _ENGINE_TYPE,
    ENGINE_SOURCE,
    ENGINE_GPU_REQUIREMENT,
    ENGINE_VOICE_CLONING,
)


class TestVoiceProfileModel(TestCase):
    """Unit tests for VoiceProfileData and VoiceProfileRecord."""

    def test_create_voice_profile_data_minimal(self) -> None:
        """VoiceProfileData can be created with only name."""
        data = VoiceProfileData(name="Test Voice")
        self.assertEqual(
            data.model_dump(),
            {
                "name": "Test Voice",
                "engine": None,
                "model": None,
                "source": None,
                "voice": None,
                "metadata": None,
            },
        )

    def test_create_voice_profile_data_full(self) -> None:
        """VoiceProfileData can be created with all fields."""
        data = VoiceProfileData(
            name="Clone Voice",
            engine="dashscope_tts",
            model="qwen3-tts-flash",
            source="api",
            voice="cosyvoice-clone-abc123",
            metadata={"quality": "high"},
        )
        self.assertEqual(
            data.model_dump(),
            {
                "name": "Clone Voice",
                "engine": "dashscope_tts",
                "model": "qwen3-tts-flash",
                "source": "api",
                "voice": "cosyvoice-clone-abc123",
                "metadata": {"quality": "high"},
            },
        )

    def test_create_voice_profile_record(self) -> None:
        """VoiceProfileRecord can be created with auto id."""
        data = VoiceProfileData(name="Test")
        record = VoiceProfileRecord(
            user_id="user-1",
            data=data,
        )
        self.assertEqual(record.user_id, "user-1")
        self.assertEqual(record.data.name, "Test")
        self.assertIsNotNone(record.id)
        self.assertIsNotNone(record.created_at)

    def test_source_is_derived_from_engine(self) -> None:
        """Client input cannot persist an inconsistent source."""
        data = VoiceProfileData(
            name="Local voice",
            engine="kokoro",
            source="api",
        )
        self.assertEqual(data.source, "local")

    def test_invalid_reference_audio_base64_is_rejected(self) -> None:
        """Malformed inline reference audio is rejected."""
        with self.assertRaises(ValidationError):
            VoiceProfileData(
                name="Bad audio",
                engine="tada",
                metadata={
                    "reference_audio_base64": "not base64!",
                },
            )

    def test_oversized_reference_audio_is_rejected(self) -> None:
        """Decoded reference audio is capped by the backend."""
        with (
            patch(
                "agentscope.app.storage._model._voice_profile."
                "_MAX_REFERENCE_AUDIO_BYTES",
                3,
            ),
            patch(
                "agentscope.app.storage._model._voice_profile."
                "_MAX_REFERENCE_AUDIO_BASE64_CHARS",
                8,
            ),
            self.assertRaises(ValidationError),
        ):
            VoiceProfileData(
                name="Large audio",
                engine="tada",
                metadata={
                    "reference_audio_base64": "MTIzNA==",
                },
            )

    def test_list_summary_redacts_reference_audio(self) -> None:
        """List records keep only an audio-presence marker."""
        record = VoiceProfileRecord(
            user_id="user-1",
            data=VoiceProfileData(
                name="Clone",
                engine="tada",
                metadata={
                    "reference_audio_base64": "QUJD",
                    "reference_text": "hello",
                },
            ),
        )
        summary = _voice_profile_summary(record)
        self.assertEqual(
            summary.data.metadata,
            {
                "reference_text": "hello",
                "has_reference_audio": True,
            },
        )
        self.assertEqual(
            record.data.metadata["reference_audio_base64"],
            "QUJD",
        )


class TestEngineToCredentialMapping(TestCase):
    """Unit tests for ENGINE_TO_CREDENTIAL_TYPE mapping."""

    def test_api_engines_map_correctly(self) -> None:
        """All engines map to their respective credentials."""
        expected = {
            "cosyvoice": "dashscope_credential",
            "dashscope_tts": "dashscope_credential",
            "openai_tts": "openai_credential",
            "gemini_tts": "gemini_credential",
            "kokoro": "local_tts_credential",
            "chatterbox": "local_tts_credential",
            "luxtts": "local_tts_credential",
            "tada": "local_tts_credential",
            "voicebox": "voicebox_credential",
        }
        self.assertEqual(ENGINE_TO_CREDENTIAL_TYPE, expected)

    def test_all_engines_have_mapping(self) -> None:
        """Every _ENGINE_TYPE value has a credential mapping."""
        args = typing.get_args(_ENGINE_TYPE)
        for engine in args:
            self.assertIn(
                engine,
                ENGINE_TO_CREDENTIAL_TYPE,
                f"Missing mapping for engine '{engine}'",
            )


class TestEngineConstants(TestCase):
    """Unit tests for engine-level constant dicts."""

    def test_all_engines_have_source(self) -> None:
        """Every engine type has an ENGINE_SOURCE entry."""
        args = typing.get_args(_ENGINE_TYPE)
        for engine in args:
            self.assertIn(engine, ENGINE_SOURCE)

    def test_all_engines_have_gpu_requirement(self) -> None:
        """Every engine type has an ENGINE_GPU_REQUIREMENT."""
        args = typing.get_args(_ENGINE_TYPE)
        for engine in args:
            self.assertIn(engine, ENGINE_GPU_REQUIREMENT)

    def test_all_engines_have_voice_cloning_flag(self) -> None:
        """Every engine type has an ENGINE_VOICE_CLONING."""
        args = typing.get_args(_ENGINE_TYPE)
        for engine in args:
            self.assertIn(engine, ENGINE_VOICE_CLONING)

    def test_api_engines_no_gpu(self) -> None:
        """API engines should not require GPU."""
        expected = {
            "cosyvoice": None,
            "dashscope_tts": None,
            "openai_tts": None,
            "gemini_tts": None,
            "kokoro": None,
            "chatterbox": "CUDA recommended",
            "luxtts": "<1 GB VRAM",
            "tada": "CUDA recommended",
            "voicebox": None,
        }
        self.assertEqual(ENGINE_GPU_REQUIREMENT, expected)

    def test_voice_cloning_flags(self) -> None:
        """Verify known cloning support flags."""
        expected = {
            "cosyvoice": True,
            "dashscope_tts": True,
            "openai_tts": True,
            "gemini_tts": False,
            "kokoro": False,
            "chatterbox": True,
            "luxtts": True,
            "tada": True,
            "voicebox": False,
        }
        self.assertEqual(ENGINE_VOICE_CLONING, expected)
