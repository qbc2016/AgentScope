# -*- coding: utf-8 -*-
"""Tests for queued-message steering state and reasoning boundaries."""
from types import SimpleNamespace
from typing import Any, AsyncGenerator
from unittest import IsolatedAsyncioTestCase

from agentscope.app._bus_ops import (
    ChatSteeringUnavailableError,
    acknowledge_steering_chat_inputs,
    enqueue_chat_input,
    finish_active_chat_reply,
    list_chat_inputs,
    register_active_chat_reply,
    steer_chat_input,
)
from agentscope.app.message_bus import InMemoryMessageBus, MessageBusKeys
from agentscope.app.middleware import SteeringMiddleware
from agentscope.event import HintBlockEvent
from agentscope.message import (
    AssistantMsg,
    HintBlock,
    Msg,
    ToolCallBlock,
    UserMsg,
)


def _make_agent() -> SimpleNamespace:
    """Build the state shape used by SteeringMiddleware."""
    return SimpleNamespace(
        name="agent",
        state=SimpleNamespace(
            session_id="session",
            reply_id="reply",
            context=[UserMsg(name="user", content="initial")],
        ),
    )


async def _collect(generator: AsyncGenerator) -> list:
    """Collect one asynchronous event generator."""
    return [event async for event in generator]


def _without_timestamps(value: Any) -> Any:
    """Remove volatile timestamps while retaining the complete structure."""
    if isinstance(value, dict):
        return {
            key: _without_timestamps(item)
            for key, item in value.items()
            if key not in {"created_at", "finished_at"}
        }
    if isinstance(value, list):
        return [_without_timestamps(item) for item in value]
    return value


def _message_structures(messages: list[Msg]) -> list[dict]:
    """Serialize complete messages without volatile timestamps."""
    return [
        _without_timestamps(message.model_dump(mode="json"))
        for message in messages
    ]


def _hint_event_structure(event: HintBlockEvent) -> dict:
    """Serialize a complete hint event without generated identity fields."""
    payload = _without_timestamps(event.model_dump(mode="json"))
    payload.pop("id")
    return payload


def _expected_hint_message(item_id: str, message: Msg) -> Msg:
    """Build the current-reply context message for one steered input."""
    return AssistantMsg(
        id="reply",
        name="agent",
        content=[
            HintBlock(
                id=f"{item_id}:0",
                source=message.name,
                hint=list(message.content),
            ),
        ],
    )


def _expected_hint_event(item_id: str, message: Msg) -> dict:
    """Build the stable complete event structure for one steering hint."""
    return _hint_event_structure(
        HintBlockEvent(
            reply_id="reply",
            block_id=f"{item_id}:0",
            source=message.name,
            hint=list(message.content),
        ),
    )


async def _custom_events(bus: InMemoryMessageBus) -> list[dict]:
    """Read complete custom events without generated identity fields."""
    entries = await bus.log_read(MessageBusKeys.session_events("session"))
    events: list[dict] = []
    for _entry_id, payload in entries:
        if payload.get("type") != "CUSTOM":
            continue
        stable = _without_timestamps(payload)
        stable.pop("id")
        events.append(stable)
    return events


class TestSteeringQueueState(IsolatedAsyncioTestCase):
    """Queue reservations remain durable until injection or fallback."""

    async def test_reserve_is_idempotent_and_finish_restores(self) -> None:
        """Repeated Steer targets once and ReplyEnd restores deferred send."""
        bus = InMemoryMessageBus()
        item = await enqueue_chat_input(
            bus,
            "user",
            "session",
            "agent",
            UserMsg(name="user", content="guide"),
        )

        async with bus.acquire_lock(
            MessageBusKeys.session_lock("session"),
            ttl_secs=60,
        ):
            await register_active_chat_reply(bus, "session", "reply")
            first = await steer_chat_input(
                bus,
                "user",
                "session",
                "agent",
                item["id"],
                "reply",
            )
            second = await steer_chat_input(
                bus,
                "user",
                "session",
                "agent",
                item["id"],
                "reply",
            )
            steering_item = {
                **item,
                "state": "steering",
                "error": None,
            }
            self.assertEqual(first, [steering_item])
            self.assertEqual(second, [steering_item])

            restored = await finish_active_chat_reply(
                bus,
                "session",
                "reply",
            )
            self.assertEqual(restored, [item["id"]])

        self.assertEqual(await list_chat_inputs(bus, "session"), [item])

    async def test_acknowledge_non_head_preserves_other_items(self) -> None:
        """Injecting one selected item never changes other FIFO entries."""
        bus = InMemoryMessageBus()
        first = await enqueue_chat_input(
            bus,
            "user",
            "session",
            "agent",
            UserMsg(name="user", content="first"),
        )
        selected = await enqueue_chat_input(
            bus,
            "user",
            "session",
            "agent",
            UserMsg(name="user", content="selected"),
        )
        third = await enqueue_chat_input(
            bus,
            "user",
            "session",
            "agent",
            UserMsg(name="user", content="third"),
        )

        async with bus.acquire_lock(
            MessageBusKeys.session_lock("session"),
            ttl_secs=60,
        ):
            await register_active_chat_reply(bus, "session", "reply")
            await steer_chat_input(
                bus,
                "user",
                "session",
                "agent",
                selected["id"],
                "reply",
            )
            removed = await acknowledge_steering_chat_inputs(
                bus,
                "session",
                "reply",
                [selected["id"]],
            )
            self.assertEqual(
                removed,
                [
                    {
                        **selected,
                        "state": "steering",
                        "error": None,
                    },
                ],
            )

        self.assertEqual(
            await list_chat_inputs(bus, "session"),
            [first, third],
        )

    async def test_stale_reply_is_rejected_without_changing_item(self) -> None:
        """A client reply id must match the server-side active registry."""
        bus = InMemoryMessageBus()
        item = await enqueue_chat_input(
            bus,
            "user",
            "session",
            "agent",
            UserMsg(name="user", content="guide"),
        )

        async with bus.acquire_lock(
            MessageBusKeys.session_lock("session"),
            ttl_secs=60,
        ):
            await register_active_chat_reply(bus, "session", "current")
            with self.assertRaises(ChatSteeringUnavailableError):
                await steer_chat_input(
                    bus,
                    "user",
                    "session",
                    "agent",
                    item["id"],
                    "stale",
                )

        self.assertEqual(await list_chat_inputs(bus, "session"), [item])


class TestSteeringMiddleware(IsolatedAsyncioTestCase):
    """Steering is consumed before or after a model call as needed."""

    async def test_pre_call_steering_reaches_one_model_call(self) -> None:
        """An already reserved item is visible to the immediate model call."""
        bus = InMemoryMessageBus()
        agent = _make_agent()
        initial = agent.state.context[0]
        guide = UserMsg(name="user", content="guide")
        item = await enqueue_chat_input(
            bus,
            "user",
            "session",
            "agent",
            guide,
        )
        calls = 0
        response = AssistantMsg(
            id="reply",
            name="agent",
            content="done",
        )

        async def model_call(**_kwargs: object) -> AsyncGenerator:
            nonlocal calls
            calls += 1
            self.assertEqual(
                _message_structures(agent.state.context),
                _message_structures(
                    [initial, _expected_hint_message(item["id"], guide)],
                ),
            )
            agent.state.context.append(response)
            yield response

        async with bus.acquire_lock(
            MessageBusKeys.session_lock("session"),
            ttl_secs=60,
        ):
            await register_active_chat_reply(bus, "session", "reply")
            await steer_chat_input(
                bus,
                "user",
                "session",
                "agent",
                item["id"],
                "reply",
            )
            events = await _collect(
                SteeringMiddleware(bus).on_reasoning(
                    agent,
                    {},
                    model_call,
                ),
            )

        self.assertEqual(calls, 1)
        self.assertEqual(
            [
                _hint_event_structure(event)
                if isinstance(event, HintBlockEvent)
                else event
                for event in events
            ],
            [_expected_hint_event(item["id"], guide), response],
        )
        self.assertEqual(await list_chat_inputs(bus, "session"), [])

    async def test_hint_does_not_attach_to_previous_reply(self) -> None:
        """A same-agent tail from an older reply remains unchanged."""
        bus = InMemoryMessageBus()
        agent = _make_agent()
        previous = AssistantMsg(
            id="previous-reply",
            name="agent",
            content="previous response",
        )
        agent.state.context = [previous]
        guide = UserMsg(name="user", content="guide current reply")
        item = await enqueue_chat_input(
            bus,
            "user",
            "session",
            "agent",
            guide,
        )

        async def model_call(**_kwargs: object) -> AsyncGenerator:
            self.assertEqual(
                _message_structures(agent.state.context),
                _message_structures(
                    [
                        previous,
                        _expected_hint_message(item["id"], guide),
                    ],
                ),
            )
            yield AssistantMsg(id="reply", name="agent", content="done")

        async with bus.acquire_lock(
            MessageBusKeys.session_lock("session"),
            ttl_secs=60,
        ):
            await register_active_chat_reply(bus, "session", "reply")
            await steer_chat_input(
                bus,
                "user",
                "session",
                "agent",
                item["id"],
                "reply",
            )
            await _collect(
                SteeringMiddleware(bus).on_reasoning(
                    agent,
                    {},
                    model_call,
                ),
            )

        self.assertEqual(
            _message_structures(agent.state.context),
            _message_structures(
                [previous, _expected_hint_message(item["id"], guide)],
            ),
        )

    async def test_post_call_steering_continues_model_only_reply(self) -> None:
        """A steer arriving mid-stream causes another non-interrupting call."""
        bus = InMemoryMessageBus()
        agent = _make_agent()
        guide = UserMsg(name="user", content="new direction")
        item = await enqueue_chat_input(
            bus,
            "user",
            "session",
            "agent",
            guide,
        )
        calls = 0
        responses: list[Msg] = []

        async def model_call(**_kwargs: object) -> AsyncGenerator:
            nonlocal calls
            calls += 1
            response = AssistantMsg(
                id="reply",
                name="agent",
                content=f"response {calls}",
            )
            responses.append(response)
            agent.state.context.append(response)
            if calls == 1:
                await steer_chat_input(
                    bus,
                    "user",
                    "session",
                    "agent",
                    item["id"],
                    "reply",
                )
            yield response

        async with bus.acquire_lock(
            MessageBusKeys.session_lock("session"),
            ttl_secs=60,
        ):
            await register_active_chat_reply(bus, "session", "reply")
            events = await _collect(
                SteeringMiddleware(bus).on_reasoning(
                    agent,
                    {},
                    model_call,
                ),
            )

        self.assertEqual(calls, 2)
        self.assertEqual(
            [
                _hint_event_structure(event)
                if isinstance(event, HintBlockEvent)
                else event
                for event in events
            ],
            [_expected_hint_event(item["id"], guide), responses[1]],
        )
        self.assertEqual(
            await _custom_events(bus),
            [
                {
                    "metadata": {},
                    "type": "CUSTOM",
                    "name": "chat_input_injected",
                    "value": {
                        "queue_item_id": item["id"],
                        "message_ids": [guide.id],
                    },
                },
            ],
        )

    async def test_post_call_tool_path_does_not_repeat_reasoning(self) -> None:
        """A tool-bound response proceeds to acting with the hint retained."""
        bus = InMemoryMessageBus()
        agent = _make_agent()
        initial = agent.state.context[0]
        guide = UserMsg(name="user", content="guide after tool")
        item = await enqueue_chat_input(
            bus,
            "user",
            "session",
            "agent",
            guide,
        )
        calls = 0

        async def tool_model_call(**_kwargs: object) -> AsyncGenerator:
            nonlocal calls
            calls += 1
            agent.state.context.append(
                AssistantMsg(
                    id="reply",
                    name="agent",
                    content=[
                        ToolCallBlock(
                            id="tool",
                            name="work",
                            input="{}",
                        ),
                    ],
                ),
            )
            await steer_chat_input(
                bus,
                "user",
                "session",
                "agent",
                item["id"],
                "reply",
            )
            return
            yield  # pragma: no cover

        async with bus.acquire_lock(
            MessageBusKeys.session_lock("session"),
            ttl_secs=60,
        ):
            await register_active_chat_reply(bus, "session", "reply")
            events = await _collect(
                SteeringMiddleware(bus).on_reasoning(
                    agent,
                    {},
                    tool_model_call,
                ),
            )

        self.assertEqual(calls, 1)
        self.assertEqual(
            [_hint_event_structure(event) for event in events],
            [_expected_hint_event(item["id"], guide)],
        )
        expected_tool_message = AssistantMsg(
            id="reply",
            name="agent",
            content=[
                ToolCallBlock(id="tool", name="work", input="{}"),
                _expected_hint_message(item["id"], guide).content[0],
            ],
        )
        self.assertEqual(
            _message_structures(agent.state.context),
            _message_structures([initial, expected_tool_message]),
        )

    async def test_unsupported_content_stays_failed_in_queue(self) -> None:
        """Preparation failures remain visible and never discard the item."""
        bus = InMemoryMessageBus()
        agent = _make_agent()
        original = UserMsg(name="user", content="will be corrupted")
        item = await enqueue_chat_input(
            bus,
            "user",
            "session",
            "agent",
            original,
        )
        entries = await bus.queue_read(
            MessageBusKeys.chat_inputs("session"),
            max_count=10,
        )
        payloads = [payload for _entry_id, payload in entries]
        payloads[0]["input"] = {"invalid": "message"}
        await bus.queue_replace(
            MessageBusKeys.chat_inputs("session"),
            payloads,
        )

        async def model_call(**_kwargs: object) -> AsyncGenerator:
            response = AssistantMsg(
                id="reply",
                name="agent",
                content="done",
            )
            agent.state.context.append(response)
            yield response

        async with bus.acquire_lock(
            MessageBusKeys.session_lock("session"),
            ttl_secs=60,
        ):
            await register_active_chat_reply(bus, "session", "reply")
            await steer_chat_input(
                bus,
                "user",
                "session",
                "agent",
                item["id"],
                "reply",
            )
            events = await _collect(
                SteeringMiddleware(bus).on_reasoning(
                    agent,
                    {},
                    model_call,
                ),
            )

        queued = await list_chat_inputs(bus, "session")
        error = queued[0]["error"]
        self.assertIsInstance(error, str)
        self.assertEqual(
            queued,
            [
                {
                    **item,
                    "input": {"invalid": "message"},
                    "state": "failed",
                    "error": error,
                },
            ],
        )
        self.assertEqual(
            [event for event in events if isinstance(event, Msg)],
            [agent.state.context[-1]],
        )
        self.assertEqual(
            await _custom_events(bus),
            [
                {
                    "metadata": {},
                    "type": "CUSTOM",
                    "name": "chat_input_steer_failed",
                    "value": {
                        "queue_item_id": item["id"],
                        "message_ids": [],
                        "message": error,
                    },
                },
            ],
        )
