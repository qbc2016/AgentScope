# -*- coding: utf-8 -*-
"""Unit tests for TTSMiddleware."""
import base64
from typing import Any, AsyncGenerator
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from agentscope.event import (
    DataBlockDeltaEvent,
    DataBlockEndEvent,
    DataBlockStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
)
from agentscope.message import Base64Source, DataBlock
from agentscope.middleware import TTSMiddleware
from agentscope.tts import TTSModelBase, TTSResponse


def _make_tts_response(
    data: str,
    media_type: str = "audio/wav",
) -> TTSResponse:
    return TTSResponse(
        content=DataBlock(
            source=Base64Source(data=data, media_type=media_type),
        ),
    )


def _make_agent_stub(
    reply_id: str = "reply-1",
    name: str = "agent",
) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.state.reply_id = reply_id
    return agent


class TestTTSMiddlewareNonRealtime(IsolatedAsyncioTestCase):
    """Tests for non-realtime (non-streaming-input) TTS path."""

    async def test_synthesize_on_text_block_end(self) -> None:
        """After TextBlockEnd, synthesize() is called with accumulated text
        and DATA_BLOCK_START + DATA_BLOCK_DELTA + DATA_BLOCK_END are emitted.
        """
        audio_b64 = base64.b64encode(b"\x00\x01\x02").decode()

        tts = MagicMock(spec=TTSModelBase)
        tts.supports_streaming_input = False
        tts.__aenter__ = AsyncMock(return_value=tts)
        tts.__aexit__ = AsyncMock(return_value=None)
        tts.synthesize = AsyncMock(
            return_value=_make_tts_response(audio_b64),
        )

        middleware = TTSMiddleware(tts_model=tts)
        agent = _make_agent_stub()

        upstream_events = [
            TextBlockDeltaEvent(
                reply_id="reply-1",
                block_id="blk-1",
                delta="Hello ",
            ),
            TextBlockDeltaEvent(
                reply_id="reply-1",
                block_id="blk-1",
                delta="world",
            ),
            TextBlockEndEvent(reply_id="reply-1", block_id="blk-1"),
        ]

        async def next_handler(**_kwargs: Any) -> AsyncGenerator:
            for evt in upstream_events:
                yield evt

        emitted = []
        async for evt in middleware.on_reasoning(agent, {}, next_handler):
            emitted.append(evt)

        # Upstream events are passed through
        self.assertIs(emitted[0], upstream_events[0])
        self.assertIs(emitted[1], upstream_events[1])
        self.assertIs(emitted[2], upstream_events[2])

        # After TextBlockEnd: START + DELTA + END
        self.assertIsInstance(emitted[3], DataBlockStartEvent)
        self.assertEqual(emitted[3].media_type, "audio/wav")

        self.assertIsInstance(emitted[4], DataBlockDeltaEvent)
        self.assertEqual(emitted[4].data, audio_b64)
        self.assertEqual(emitted[4].media_type, "audio/wav")

        self.assertIsInstance(emitted[5], DataBlockEndEvent)
        self.assertEqual(emitted[5].block_id, emitted[3].block_id)

        # synthesize was called with a Msg containing the full text
        tts.synthesize.assert_called_once()
        msg_arg = tts.synthesize.call_args[0][0]
        self.assertEqual(msg_arg.content[0].text, "Hello world")

    async def test_empty_text_skips_synthesize(self) -> None:
        """When accumulated text is whitespace-only, synthesize is not called
        and no DATA_BLOCK events are emitted."""
        tts = MagicMock(spec=TTSModelBase)
        tts.supports_streaming_input = False
        tts.__aenter__ = AsyncMock(return_value=tts)
        tts.__aexit__ = AsyncMock(return_value=None)
        tts.synthesize = AsyncMock()

        middleware = TTSMiddleware(tts_model=tts)
        agent = _make_agent_stub()

        upstream_events = [
            TextBlockDeltaEvent(
                reply_id="reply-1",
                block_id="blk-1",
                delta="   ",
            ),
            TextBlockEndEvent(reply_id="reply-1", block_id="blk-1"),
        ]

        async def next_handler(**_kwargs: Any) -> AsyncGenerator:
            for evt in upstream_events:
                yield evt

        emitted = []
        async for evt in middleware.on_reasoning(agent, {}, next_handler):
            emitted.append(evt)

        # Only the upstream events are passed through
        self.assertEqual(len(emitted), 2)
        tts.synthesize.assert_not_called()

    async def test_streaming_output_multiple_chunks(self) -> None:
        """When synthesize() returns an async generator, each chunk produces
        a DATA_BLOCK_DELTA under the same block_id."""
        chunk1 = _make_tts_response("AAAA")
        chunk2 = _make_tts_response("BBBB")

        async def synth_gen(*_args: Any, **_kwargs: Any) -> AsyncGenerator:
            yield chunk1
            yield chunk2

        tts = MagicMock(spec=TTSModelBase)
        tts.supports_streaming_input = False
        tts.__aenter__ = AsyncMock(return_value=tts)
        tts.__aexit__ = AsyncMock(return_value=None)
        tts.synthesize = AsyncMock(return_value=synth_gen())

        middleware = TTSMiddleware(tts_model=tts)
        agent = _make_agent_stub()

        upstream_events = [
            TextBlockDeltaEvent(
                reply_id="reply-1",
                block_id="blk-1",
                delta="hi",
            ),
            TextBlockEndEvent(reply_id="reply-1", block_id="blk-1"),
        ]

        async def next_handler(**_kwargs: Any) -> AsyncGenerator:
            for evt in upstream_events:
                yield evt

        emitted = []
        async for evt in middleware.on_reasoning(agent, {}, next_handler):
            emitted.append(evt)

        # Upstream pass-through + START + 2x DELTA + END = 2 + 4 = 6
        data_events = [
            e
            for e in emitted
            if isinstance(
                e,
                (DataBlockStartEvent, DataBlockDeltaEvent, DataBlockEndEvent),
            )
        ]
        self.assertEqual(len(data_events), 4)
        self.assertIsInstance(data_events[0], DataBlockStartEvent)
        self.assertIsInstance(data_events[1], DataBlockDeltaEvent)
        self.assertEqual(data_events[1].data, "AAAA")
        self.assertIsInstance(data_events[2], DataBlockDeltaEvent)
        self.assertEqual(data_events[2].data, "BBBB")
        self.assertIsInstance(data_events[3], DataBlockEndEvent)

        # All share the same block_id
        block_id = data_events[0].block_id
        for e in data_events:
            self.assertEqual(e.block_id, block_id)


class TestTTSMiddlewareRealtime(IsolatedAsyncioTestCase):
    """Tests for realtime (streaming-input) TTS path."""

    async def test_push_on_delta_and_drain_on_end(self) -> None:
        """In realtime mode, push() is called on each TextBlockDelta and
        synthesize() drains on TextBlockEnd."""
        push_audio = _make_tts_response("PUSH1")
        drain_audio = _make_tts_response("DRAIN")

        tts = MagicMock(spec=TTSModelBase)
        tts.supports_streaming_input = True
        tts.__aenter__ = AsyncMock(return_value=tts)
        tts.__aexit__ = AsyncMock(return_value=None)
        tts.push = AsyncMock(return_value=push_audio)
        tts.synthesize = AsyncMock(return_value=drain_audio)

        middleware = TTSMiddleware(tts_model=tts)
        agent = _make_agent_stub()

        upstream_events = [
            TextBlockDeltaEvent(
                reply_id="reply-1",
                block_id="blk-1",
                delta="Hello",
            ),
            TextBlockEndEvent(reply_id="reply-1", block_id="blk-1"),
        ]

        async def next_handler(**_kwargs: Any) -> AsyncGenerator:
            for evt in upstream_events:
                yield evt

        emitted = []
        async for evt in middleware.on_reasoning(agent, {}, next_handler):
            emitted.append(evt)

        # push() called with the delta text
        tts.push.assert_called_once()
        push_msg = tts.push.call_args[0][0]
        self.assertEqual(push_msg.id, "blk-1")
        self.assertEqual(push_msg.content[0].text, "Hello")

        # synthesize() called to drain
        tts.synthesize.assert_called_once()

        data_events = [
            e
            for e in emitted
            if isinstance(
                e,
                (DataBlockStartEvent, DataBlockDeltaEvent, DataBlockEndEvent),
            )
        ]
        # START + DELTA(push) + DELTA(drain) + END
        self.assertEqual(len(data_events), 4)
        self.assertIsInstance(data_events[0], DataBlockStartEvent)
        self.assertEqual(data_events[1].data, "PUSH1")
        self.assertEqual(data_events[2].data, "DRAIN")
        self.assertIsInstance(data_events[3], DataBlockEndEvent)

    async def test_push_returns_none_no_audio_emitted(self) -> None:
        """When push() returns empty content, no DATA_BLOCK events are emitted
        until synthesize() drains."""
        empty_response = TTSResponse(content=None)
        drain_audio = _make_tts_response("FINAL")

        tts = MagicMock(spec=TTSModelBase)
        tts.supports_streaming_input = True
        tts.__aenter__ = AsyncMock(return_value=tts)
        tts.__aexit__ = AsyncMock(return_value=None)
        tts.push = AsyncMock(return_value=empty_response)
        tts.synthesize = AsyncMock(return_value=drain_audio)

        middleware = TTSMiddleware(tts_model=tts)
        agent = _make_agent_stub()

        upstream_events = [
            TextBlockDeltaEvent(
                reply_id="reply-1",
                block_id="blk-1",
                delta="Hi",
            ),
            TextBlockDeltaEvent(
                reply_id="reply-1",
                block_id="blk-1",
                delta=" there",
            ),
            TextBlockEndEvent(reply_id="reply-1", block_id="blk-1"),
        ]

        async def next_handler(**_kwargs: Any) -> AsyncGenerator:
            for evt in upstream_events:
                yield evt

        emitted = []
        async for evt in middleware.on_reasoning(agent, {}, next_handler):
            emitted.append(evt)

        # push called twice but produced no audio
        self.assertEqual(tts.push.call_count, 2)

        data_events = [
            e
            for e in emitted
            if isinstance(
                e,
                (DataBlockStartEvent, DataBlockDeltaEvent, DataBlockEndEvent),
            )
        ]
        # Only drain produces: START + DELTA + END
        self.assertEqual(len(data_events), 3)
        self.assertIsInstance(data_events[0], DataBlockStartEvent)
        self.assertEqual(data_events[1].data, "FINAL")
        self.assertIsInstance(data_events[2], DataBlockEndEvent)
