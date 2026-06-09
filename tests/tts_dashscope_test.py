# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for the TTS module.

Covers:
  * ``TTSModelBase`` default no-op behaviour for ``connect`` / ``close`` /
    ``push`` (so non-realtime subclasses needn't override them).
  * ``DashScopeTTSModel`` non-streaming aggregation.
  * ``DashScopeTTSModel`` streaming: incremental deltas and ``is_last``
    placement at the final chunk only.
"""
import base64
import io
import unittest
import wave
from typing import Any, AsyncGenerator
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

from agentscope.credential import DashScopeCredential
from agentscope.message import Msg, TextBlock
from agentscope.tts import DashScopeTTSModel, TTSModelBase, TTSResponse


_MEDIA_TYPE = "audio/wav"
# DashScope TTS emits 24kHz / mono / 16-bit PCM; the WAV wrapping in the
# model layer uses the same parameters.
_TTS_SAMPLE_RATE = 24000
_TTS_CHANNELS = 1
_TTS_SAMPLE_WIDTH = 2  # bytes (= 16 bit)
_WAV_HEADER_LEN = 44


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(text: str = "Hello world") -> Msg:
    return Msg(name="user", role="user", content=[TextBlock(text=text)])


def _make_api_chunk(data_bytes: bytes | None) -> MagicMock:
    """Build a chunk shaped like what dashscope.MultiModalConversation
    yields. ``data_bytes=None`` represents a chunk with no output."""
    chunk = MagicMock()
    if data_bytes is None:
        chunk.output = None
        return chunk
    chunk.output = MagicMock()
    chunk.output.audio = MagicMock()
    chunk.output.audio.data = base64.b64encode(data_bytes).decode("ascii")
    return chunk


def _make_api_generator(chunks: list[bytes | None]) -> Any:
    """Build a sync generator like ``MultiModalConversation.call`` returns."""

    def _gen() -> Any:
        for data in chunks:
            yield _make_api_chunk(data)

    return _gen()


def _make_model(stream: bool = False) -> DashScopeTTSModel:
    return DashScopeTTSModel(
        credential=DashScopeCredential(api_key="test"),
        model="qwen3-tts-flash",
        voice="Cherry",
        stream=stream,
    )


# ---------------------------------------------------------------------------
# TTSModelBase — default no-op surface for non-realtime subclasses
# ---------------------------------------------------------------------------


class _DummyTTS(TTSModelBase):
    """Minimal subclass that implements only ``synthesize`` — exercises the
    base class's no-op ``connect`` / ``close`` / ``push`` defaults."""

    async def synthesize(
        self,
        msg: Msg | None = None,
        **kwargs: Any,
    ) -> TTSResponse | AsyncGenerator[TTSResponse, None]:
        del msg, kwargs
        return TTSResponse(content=None)


class _RealtimeDummyTTS(_DummyTTS):
    """Realtime-flavoured dummy to assert ``__aenter__`` drives the lifecycle
    hooks when ``supports_streaming_input`` is True."""

    supports_streaming_input = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.connect_calls = 0
        self.close_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


def _make_dummy(cls: type = _DummyTTS) -> TTSModelBase:
    return cls(
        credential=DashScopeCredential(api_key="test"),
        model="x",
        stream=False,
    )


class TestTTSModelBaseDefaults(IsolatedAsyncioTestCase):
    """The base class supplies safe no-op defaults so non-realtime subclasses
    don't need to implement realtime-only hooks."""

    async def test_default_connect_close_noop(self) -> None:
        """Default connect/close return without raising."""
        model = _make_dummy()
        await model.connect()
        await model.close()

    async def test_default_push_returns_empty(self) -> None:
        """Default push returns an empty TTSResponse rather than raising,
        so a misuse on a non-realtime model degrades gracefully."""
        model = _make_dummy()
        resp = await model.push(_make_msg("ignored"))
        self.assertIsInstance(resp, TTSResponse)
        self.assertIsNone(resp.content)

    async def test_aenter_skips_hooks_for_non_realtime(self) -> None:
        """``async with`` on a non-realtime model must not invoke connect/
        close (gated by ``supports_streaming_input``)."""
        model = _make_dummy(_DummyTTS)
        async with model as m:
            self.assertIs(m, model)

    async def test_aenter_invokes_hooks_for_realtime(self) -> None:
        """For ``supports_streaming_input=True`` subclasses, connect/close
        fire on enter/exit exactly once."""
        model = _make_dummy(_RealtimeDummyTTS)
        async with model:
            self.assertEqual(model.connect_calls, 1)
            self.assertEqual(model.close_calls, 0)
        self.assertEqual(model.close_calls, 1)


# ---------------------------------------------------------------------------
# DashScopeTTSModel — non-streaming
# ---------------------------------------------------------------------------


def _parse_wav_payload(wav_bytes: bytes) -> bytes:
    """Decode a full WAV file and return its raw PCM frames."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        return wav.readframes(wav.getnframes())


class TestDashScopeTTSModelNonStream(IsolatedAsyncioTestCase):
    """Non-streaming synthesis aggregates every audio chunk into one
    self-contained WAV (so ``<audio>`` elements can play it directly)."""

    @patch("dashscope.MultiModalConversation")
    async def test_aggregates_chunks(self, mock_mmc: MagicMock) -> None:
        """All API chunks are aggregated into one self-contained WAV."""
        mock_mmc.call.return_value = _make_api_generator(
            [b"AAAA", b"BBBB", b"CCCC"],
        )
        model = _make_model(stream=False)

        result = await model.synthesize(_make_msg())

        self.assertIsInstance(result, TTSResponse)
        self.assertEqual(
            result.content.source.media_type,
            _MEDIA_TYPE,
        )
        wav_bytes = base64.b64decode(result.content.source.data)
        # The output is a self-contained WAV; the wave module must be able
        # to parse the parameters we wrapped it with and recover the
        # original concatenated PCM frames.
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            self.assertEqual(wav.getframerate(), _TTS_SAMPLE_RATE)
            self.assertEqual(wav.getnchannels(), _TTS_CHANNELS)
            self.assertEqual(wav.getsampwidth(), _TTS_SAMPLE_WIDTH)
            self.assertEqual(wav.readframes(wav.getnframes()), b"AAAABBBBCCCC")
        self.assertTrue(result.is_last)

    @patch("dashscope.MultiModalConversation")
    async def test_msg_none_short_circuits(
        self,
        mock_mmc: MagicMock,
    ) -> None:
        """``synthesize(None)`` returns an empty response without touching
        the API."""
        model = _make_model(stream=False)

        result = await model.synthesize(None)

        self.assertIsInstance(result, TTSResponse)
        self.assertIsNone(result.content)
        mock_mmc.call.assert_not_called()

    @patch("dashscope.MultiModalConversation")
    async def test_skips_empty_chunks(self, mock_mmc: MagicMock) -> None:
        """Chunks without ``output`` are ignored during aggregation."""
        mock_mmc.call.return_value = _make_api_generator(
            [None, b"AAAA", None, b"BBBB"],
        )
        model = _make_model(stream=False)

        result = await model.synthesize(_make_msg())

        wav_bytes = base64.b64decode(result.content.source.data)
        self.assertEqual(_parse_wav_payload(wav_bytes), b"AAAABBBB")


# ---------------------------------------------------------------------------
# DashScopeTTSModel — streaming
# ---------------------------------------------------------------------------


class TestDashScopeTTSModelStream(IsolatedAsyncioTestCase):
    """Streaming synthesis yields one TTSResponse per audio chunk.

    The first chunk is prefixed with a streaming WAV/RIFF header so the
    frontend can start playback immediately; subsequent chunks are raw PCM
    appended to that open WAV stream. Only the final yielded chunk has
    ``is_last=True``.
    """

    @patch("dashscope.MultiModalConversation")
    async def test_incremental_deltas(self, mock_mmc: MagicMock) -> None:
        """Each API chunk yields one TTSResponse with incremental PCM."""
        mock_mmc.call.return_value = _make_api_generator(
            [b"AAAA", b"BBBB", b"CCCC"],
        )
        model = _make_model(stream=True)

        gen = await model.synthesize(_make_msg())
        chunks = [c async for c in gen]

        payloads = [base64.b64decode(c.content.source.data) for c in chunks]

        # First chunk: WAV header + first PCM delta.
        self.assertTrue(payloads[0].startswith(b"RIFF"))
        self.assertEqual(payloads[0][8:12], b"WAVE")
        self.assertEqual(payloads[0][_WAV_HEADER_LEN:], b"AAAA")
        # Subsequent chunks: raw PCM, no header.
        self.assertEqual(payloads[1], b"BBBB")
        self.assertEqual(payloads[2], b"CCCC")

        self.assertEqual(
            [c.is_last for c in chunks],
            [False, False, True],
        )
        for chunk in chunks:
            self.assertEqual(
                chunk.content.source.media_type,
                _MEDIA_TYPE,
            )

    @patch("dashscope.MultiModalConversation")
    async def test_single_chunk_marked_last(
        self,
        mock_mmc: MagicMock,
    ) -> None:
        """A lone audio chunk must still be flagged ``is_last=True`` and
        carry the streaming WAV header in front of its PCM payload."""
        mock_mmc.call.return_value = _make_api_generator([b"ONLYCHUNK"])
        model = _make_model(stream=True)

        gen = await model.synthesize(_make_msg())
        chunks = [c async for c in gen]

        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].is_last)
        payload = base64.b64decode(chunks[0].content.source.data)
        self.assertTrue(payload.startswith(b"RIFF"))
        self.assertEqual(payload[_WAV_HEADER_LEN:], b"ONLYCHUNK")

    @patch("dashscope.MultiModalConversation")
    async def test_empty_stream_yields_terminal(
        self,
        mock_mmc: MagicMock,
    ) -> None:
        """When the API yields no audio at all, the generator must still
        emit a single terminal sentinel so consumers can detect EOS."""
        mock_mmc.call.return_value = _make_api_generator([None, None])
        model = _make_model(stream=True)

        gen = await model.synthesize(_make_msg())
        chunks = [c async for c in gen]

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].content)
        self.assertTrue(chunks[0].is_last)


if __name__ == "__main__":
    unittest.main()
