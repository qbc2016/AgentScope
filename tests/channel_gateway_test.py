# -*- coding: utf-8 -*-
"""Tests for channel data-plane internals that stand alone from a live run.

Covers the presenter's event-stream folding (``_collect``, driven off a
seeded replay log), the gateway's media aggregation, and pending-confirm
persistence. Full two-phase orchestration needs a running agent and is
exercised end-to-end against a real bus / bot.
"""
# pylint: disable=protected-access,missing-function-docstring,unused-argument
from typing import Any
from unittest import IsolatedAsyncioTestCase

from agentscope.app.channel._base import (
    ChannelBase,
    ChannelCapability,
    ChannelEvent,
)
from agentscope.app.channel._gateway import ChannelGateway
from agentscope.app.channel._pending import PendingConfirm
from agentscope.app.channel._presenter import ChannelPresenter
from agentscope.app.message_bus import InMemoryMessageBus, MessageBusKeys
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

_SESSION_ID = "s1"


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


def _event() -> ChannelEvent:
    return ChannelEvent(channel_id="chan-1", channel_user_id="u", chat_id="c")


class _FakeChannel(ChannelBase):
    """A channel that records what streaming would have sent."""

    def __init__(self, streaming: bool = False) -> None:
        self.capabilities = ChannelCapability(streaming=streaming)
        self.updates: list[str] = []
        self.ended: str | None = None

    @property
    def channel_id(self) -> str:
        return "chan-1"

    async def start_listening(self) -> None:
        pass

    async def send_response(self, event: Any, content: Any) -> None:
        pass

    async def stream_start(self, event: ChannelEvent) -> str | None:
        return "card-1"

    async def stream_update(self, ref: str, content: Any) -> None:
        self.updates.append(
            "".join(b.text for b in content if isinstance(b, TextBlock)),
        )

    async def stream_end(self, ref: str, content: Any) -> None:
        self.ended = "".join(
            b.text for b in content if isinstance(b, TextBlock)
        )


async def _collect(
    events: list[dict],
    streaming: bool = False,
    **presentation: Any,
) -> tuple[tuple[str, dict | None], _FakeChannel]:
    """Seed a session-events replay log, then fold it via the presenter.

    The presenter subscribes first, then replays the log — seeding the
    log (including the terminal event) is enough to exercise the fold
    without a live producer.
    """
    bus = InMemoryMessageBus()
    key = MessageBusKeys.session_events(_SESSION_ID)
    for event in events:
        await bus.log_append(key, event)
    presenter = ChannelPresenter(storage=None, message_bus=bus)
    channel = _FakeChannel(streaming=streaming)
    result = await presenter._collect(
        _SESSION_ID,
        _record(**presentation),
        channel,
        _event(),
    )
    return result, channel


class CollectTest(IsolatedAsyncioTestCase):
    """The event-stream → (text, confirm?) folding."""

    async def test_text_reply(self) -> None:
        (text, confirm), _ = await _collect(
            [
                {"type": EventType.REPLY_START},
                {"type": EventType.TEXT_BLOCK_DELTA, "delta": "Hello "},
                {"type": EventType.TEXT_BLOCK_DELTA, "delta": "world"},
                {"type": EventType.REPLY_END},
            ],
        )
        self.assertEqual(text, "Hello world")
        self.assertIsNone(confirm)

    async def test_streaming_delivers_and_suppresses_text(self) -> None:
        (text, _), channel = await _collect(
            [
                {"type": EventType.REPLY_START},
                {"type": EventType.TEXT_BLOCK_DELTA, "delta": "Hi "},
                {"type": EventType.TEXT_BLOCK_DELTA, "delta": "there"},
                {"type": EventType.REPLY_END},
            ],
            streaming=True,
        )
        # Streaming delivered the reply; no text returned for a re-send.
        self.assertEqual(text, "")
        self.assertEqual(channel.ended, "Hi there")

    async def test_confirm_returns_early(self) -> None:
        (text, confirm), _ = await _collect(
            [
                {"type": EventType.REPLY_START},
                {"type": EventType.TEXT_BLOCK_DELTA, "delta": "working"},
                {
                    "type": EventType.REQUIRE_USER_CONFIRM,
                    "id": "req-1",
                    "reply_id": "r-1",
                    "tool_calls": [],
                },
                {"type": EventType.REPLY_END},  # not reached
            ],
        )
        self.assertEqual(text, "working")
        assert confirm is not None
        self.assertEqual(confirm["id"], "req-1")

    async def test_error_reply_end(self) -> None:
        (text, confirm), _ = await _collect(
            [
                {"type": EventType.REPLY_START},
                {
                    "type": EventType.REPLY_END,
                    "finished_reason": ReplyFinishedReason.ERROR,
                    "error": {"type": "internal", "message": "boom"},
                },
            ],
        )
        self.assertIn("error", text.lower())
        self.assertIsNone(confirm)

    async def test_thinking_filtered_by_default(self) -> None:
        (text, _), _ = await _collect(
            [
                {"type": EventType.REPLY_START},
                {"type": EventType.THINKING_BLOCK_START},
                {"type": EventType.THINKING_BLOCK_DELTA, "delta": "hmm"},
                {"type": EventType.THINKING_BLOCK_END},
                {"type": EventType.TEXT_BLOCK_DELTA, "delta": "answer"},
                {"type": EventType.REPLY_END},
            ],
        )
        self.assertEqual(text, "answer")

    async def test_thinking_shown_when_enabled(self) -> None:
        (text, _), _ = await _collect(
            [
                {"type": EventType.REPLY_START},
                {"type": EventType.THINKING_BLOCK_START},
                {"type": EventType.THINKING_BLOCK_DELTA, "delta": "hmm"},
                {"type": EventType.TEXT_BLOCK_DELTA, "delta": "answer"},
                {"type": EventType.REPLY_END},
            ],
            show_thinking=True,
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

    def _media_event(self, name: str) -> ChannelEvent:
        return ChannelEvent(
            channel_id="c",
            channel_user_id="u",
            chat_id="chat",
            content=[self._img(name)],
        )

    async def test_aggregate_media_only_buffers(self) -> None:
        bus = InMemoryMessageBus()
        gw = ChannelGateway(storage=None, message_bus=bus)
        self.assertIsNone(
            await gw._aggregate_media(self._media_event("a.png")),
        )

    async def test_aggregate_text_drains_buffered_media(self) -> None:
        bus = InMemoryMessageBus()
        gw = ChannelGateway(storage=None, message_bus=bus)
        await gw._aggregate_media(self._media_event("a.png"))
        await gw._aggregate_media(self._media_event("b.png"))
        content = await gw._aggregate_media(
            ChannelEvent(
                channel_id="c",
                channel_user_id="u",
                chat_id="chat",
                content=[TextBlock(text="look")],
            ),
        )
        assert content is not None
        self.assertEqual(len(content), 3)  # two buffered images + text
        self.assertIsInstance(content[0], DataBlock)
        self.assertIsInstance(content[-1], TextBlock)


class PendingConfirmTest(IsolatedAsyncioTestCase):
    """Pending-confirm persistence is single-use."""

    async def test_save_take_roundtrip(self) -> None:
        bus = InMemoryMessageBus()
        pending = PendingConfirm(
            session_id="s",
            agent_id="a",
            user_id="u",
            channel_id="c",
            chat_id="chat",
            reply_id="r",
            tool_calls=[],
            ref="card-1",
        )
        await pending.save(bus, "req-1")
        loaded = await PendingConfirm.take(bus, "req-1")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.ref, "card-1")
        # Single-use: gone after take.
        self.assertIsNone(await PendingConfirm.take(bus, "req-1"))
