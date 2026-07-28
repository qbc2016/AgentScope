# -*- coding: utf-8 -*-
"""Tests for channel gateway internals that stand alone from a live run.

Covers the event-stream folding (``_collect``), media aggregation, and
pending-confirm persistence. The full two-phase orchestration needs a
running agent and is exercised end-to-end once the Feishu adapter lands.
"""
# pylint: disable=protected-access,missing-function-docstring
from typing import Any, AsyncIterator
from unittest import IsolatedAsyncioTestCase

from agentscope.app.channel._gateway import ChannelGateway
from agentscope.app.channel._media import buffer_blocks, drain_blocks
from agentscope.app.channel._pending import (
    PendingConfirm,
    save_pending,
    take_pending,
)
from agentscope.app.channel._base import ChannelEvent
from agentscope.app.message_bus import InMemoryMessageBus
from agentscope.app.storage import (
    ChannelBinding,
    ChannelRecord,
    ReplyPresentation,
    RoutingConfig,
    SessionSettings,
)
from agentscope.event import EventType
from agentscope.message import DataBlock, TextBlock
from agentscope.message._block import URLSource
from agentscope.types import ReplyFinishedReason


def _record(**presentation: Any) -> ChannelRecord:
    return ChannelRecord(
        id="chan-1",
        channel_type="feishu",
        user_id="owner-1",
        routing=RoutingConfig(
            bindings=[ChannelBinding(match_value="*", agent_id="a")],
        ),
        session=SessionSettings(chat_model_config={"type": "x"}),
        presentation=ReplyPresentation(**presentation),
        created_at="t",
        updated_at="t",
    )


async def _aiter(events: list[dict]) -> AsyncIterator[dict]:
    for e in events:
        yield e


class CollectTest(IsolatedAsyncioTestCase):
    """The event-stream → (text, confirm?) folding."""

    def _gw(self) -> ChannelGateway:
        return ChannelGateway(storage=None, message_bus=InMemoryMessageBus())

    async def test_text_reply(self) -> None:
        events = [
            {"type": EventType.REPLY_START},
            {"type": EventType.TEXT_BLOCK_DELTA, "delta": "Hello "},
            {"type": EventType.TEXT_BLOCK_DELTA, "delta": "world"},
            {"type": EventType.REPLY_END},
        ]
        text, confirm = await self._gw()._collect(_aiter(events), _record())
        self.assertEqual(text, "Hello world")
        self.assertIsNone(confirm)

    async def test_confirm_returns_early(self) -> None:
        confirm_evt = {
            "type": EventType.REQUIRE_USER_CONFIRM,
            "id": "req-1",
            "reply_id": "r-1",
            "tool_calls": [],
        }
        events = [
            {"type": EventType.REPLY_START},
            {"type": EventType.TEXT_BLOCK_DELTA, "delta": "working"},
            confirm_evt,
            {"type": EventType.REPLY_END},  # not reached
        ]
        text, confirm = await self._gw()._collect(_aiter(events), _record())
        self.assertEqual(text, "working")
        self.assertEqual(confirm["id"], "req-1")

    async def test_error_reply_end(self) -> None:
        events = [
            {"type": EventType.REPLY_START},
            {
                "type": EventType.REPLY_END,
                "finished_reason": ReplyFinishedReason.ERROR,
                "error": {"type": "internal", "message": "boom"},
            },
        ]
        text, confirm = await self._gw()._collect(_aiter(events), _record())
        self.assertIn("error", text.lower())
        self.assertIsNone(confirm)

    async def test_error_without_reply_start(self) -> None:
        # Orphan error (deleted session) terminates even before REPLY_START.
        events = [
            {
                "type": EventType.REPLY_END,
                "finished_reason": ReplyFinishedReason.ERROR,
                "error": {"type": "internal", "message": "gone"},
            },
        ]
        text, _ = await self._gw()._collect(_aiter(events), _record())
        self.assertIn("error", text.lower())

    async def test_thinking_filtered_by_default(self) -> None:
        events = [
            {"type": EventType.REPLY_START},
            {"type": EventType.THINKING_BLOCK_START},
            {"type": EventType.THINKING_BLOCK_DELTA, "delta": "hmm"},
            {"type": EventType.THINKING_BLOCK_END},
            {"type": EventType.TEXT_BLOCK_DELTA, "delta": "answer"},
            {"type": EventType.REPLY_END},
        ]
        text, _ = await self._gw()._collect(_aiter(events), _record())
        self.assertEqual(text, "answer")

    async def test_thinking_shown_when_enabled(self) -> None:
        events = [
            {"type": EventType.REPLY_START},
            {"type": EventType.THINKING_BLOCK_START},
            {"type": EventType.THINKING_BLOCK_DELTA, "delta": "hmm"},
            {"type": EventType.TEXT_BLOCK_DELTA, "delta": "answer"},
            {"type": EventType.REPLY_END},
        ]
        text, _ = await self._gw()._collect(
            _aiter(events),
            _record(show_thinking=True),
        )
        self.assertIn("hmm", text)


class MediaBufferTest(IsolatedAsyncioTestCase):
    """Media-only messages buffer; a text message drains them."""

    def _img(self, name: str) -> DataBlock:
        return DataBlock(
            source=URLSource(
                url=f"https://example.com/{name}",
                media_type="image/png",
            ),
        )

    async def test_buffer_then_drain(self) -> None:
        bus = InMemoryMessageBus()
        await buffer_blocks(bus, "c", "chat", "u", [self._img("a.png")])
        await buffer_blocks(bus, "c", "chat", "u", [self._img("b.png")])
        drained = await drain_blocks(bus, "c", "chat", "u")
        self.assertEqual(len(drained), 2)
        # Second drain is empty.
        self.assertEqual(await drain_blocks(bus, "c", "chat", "u"), [])

    async def test_aggregate_media_only_buffers(self) -> None:
        bus = InMemoryMessageBus()
        gw = ChannelGateway(storage=None, message_bus=bus)
        event = ChannelEvent(
            channel_id="c",
            channel_user_id="u",
            chat_id="chat",
            content=[self._img("a.png")],
        )
        self.assertIsNone(await gw._aggregate_media(event))

    async def test_aggregate_text_drains_buffer(self) -> None:
        bus = InMemoryMessageBus()
        gw = ChannelGateway(storage=None, message_bus=bus)
        await buffer_blocks(bus, "c", "chat", "u", [self._img("a.png")])
        event = ChannelEvent(
            channel_id="c",
            channel_user_id="u",
            chat_id="chat",
            content=[TextBlock(text="look")],
        )
        content = await gw._aggregate_media(event)
        self.assertEqual(len(content), 2)  # buffered image + text
        self.assertIsInstance(content[0], DataBlock)
        self.assertIsInstance(content[1], TextBlock)


class PendingConfirmTest(IsolatedAsyncioTestCase):
    """Pending-confirm persistence is single-use."""

    async def test_save_take_roundtrip(self) -> None:
        bus = InMemoryMessageBus()
        pending = PendingConfirm(
            session_id="s",
            agent_id="a",
            user_id="u",
            reply_id="r",
            tool_calls=[],
            event=ChannelEvent(
                channel_id="c",
                channel_user_id="u",
                chat_id="chat",
            ),
            ref="card-1",
        )
        await save_pending(bus, "req-1", pending)
        loaded = await take_pending(bus, "req-1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.ref, "card-1")
        # Single-use: gone after take.
        self.assertIsNone(await take_pending(bus, "req-1"))
