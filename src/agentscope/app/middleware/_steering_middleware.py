# -*- coding: utf-8 -*-
"""Inject selected pending chat inputs at safe reasoning boundaries."""
from typing import Any, AsyncGenerator, Callable

from pydantic import TypeAdapter

from .._bus_ops import (
    acknowledge_steering_chat_inputs,
    fail_steering_chat_input,
    list_steering_chat_inputs,
    publish_session_event,
)
from ..message_bus import MessageBus
from ...agent import Agent
from ...event import CustomEvent, HintBlockEvent
from ...message import (
    AssistantMsg,
    DataBlock,
    HintBlock,
    Msg,
    TextBlock,
)
from ...middleware import MiddlewareBase

_MESSAGE_INPUT_ADAPTER: TypeAdapter = TypeAdapter(Msg | list[Msg])
_STEERING_HINT_SOURCE = "chat_input_steering"


class SteeringMiddleware(MiddlewareBase):
    """Consume reply-targeted queue items around every model call.

    The after-call check catches input submitted while a model stream was in
    progress. If that call produced a terminal ``Msg``, the middleware keeps
    its already-streamed blocks in context but withholds the internal terminal
    candidate and performs another model call in the same reasoning round.

    Args:
        message_bus (`MessageBus`):
            Application message bus containing the pending-input FIFO.
    """

    def __init__(self, message_bus: MessageBus) -> None:
        """Initialize the middleware with its shared message bus."""
        self._bus = message_bus

    async def on_reasoning(  # type: ignore[override]
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator[Any, None]:
        """Inject before and after model calls without cancelling streaming.

        Args:
            agent (`Agent`):
                Agent whose context receives the steering hints.
            input_kwargs (`dict`):
                Reasoning arguments forwarded to downstream middleware.
            next_handler (`Callable[..., AsyncGenerator]`):
                Downstream reasoning chain.

        Yields:
            `Any`:
                Injection events and every downstream reasoning event.
        """
        _injected, events = await self._consume(agent)
        for event in events:
            yield event

        while True:
            terminal_messages: list[Msg] = []
            async for event in next_handler(**input_kwargs):
                if isinstance(event, Msg):
                    terminal_messages.append(event)
                else:
                    yield event

            injected_after, events = await self._consume(agent)
            for event in events:
                yield event

            if injected_after and terminal_messages:
                # The completed response is already in context and its stream
                # remains visible. Re-run downstream reasoning so the model can
                # respond to the newly appended hint before this reply exits.
                continue

            for message in terminal_messages:
                yield message
            return

    async def _consume(
        self,
        agent: Agent,
    ) -> tuple[bool, list[HintBlockEvent]]:
        """Inject every pending item targeting the agent's current reply.

        Args:
            agent (`Agent`):
                Agent whose current reply and context select the items.

        Returns:
            `tuple[bool, list[HintBlockEvent]]`:
                Whether at least one item was injected and its emitted events.
        """
        session_id = agent.state.session_id
        reply_id = agent.state.reply_id
        items = await list_steering_chat_inputs(
            self._bus,
            session_id,
            reply_id,
        )
        if not items:
            return False, []

        prepared: list[tuple[dict, list[Msg], list[HintBlock]]] = []
        events: list[HintBlockEvent] = []
        for item in items:
            try:
                parsed_input = _MESSAGE_INPUT_ADAPTER.validate_python(
                    item["input"],
                )
                messages = (
                    parsed_input
                    if isinstance(parsed_input, list)
                    else [parsed_input]
                )
                hints = self._to_hints(item["id"], messages)
                prepared.append((item, messages, hints))
            except Exception as exc:
                error = f"The queued message could not be injected: {exc}"
                await fail_steering_chat_input(
                    self._bus,
                    session_id,
                    reply_id,
                    item["id"],
                    error,
                )
                event = CustomEvent(
                    name="chat_input_steer_failed",
                    value={
                        "queue_item_id": item["id"],
                        "message_ids": self._message_ids(item["input"]),
                        "message": error,
                    },
                )
                await publish_session_event(
                    self._bus,
                    session_id,
                    event.model_dump(mode="json"),
                )

        if not prepared:
            return False, events

        hints = [
            hint
            for _item, _messages, item_hints in prepared
            for hint in item_hints
        ]
        rollback = self._append_hints(agent, hints)
        try:
            await acknowledge_steering_chat_inputs(
                self._bus,
                session_id,
                reply_id,
                [item["id"] for item, _messages, _hints in prepared],
            )
        except Exception:
            rollback()
            raise

        for item, messages, item_hints in prepared:
            for hint in item_hints:
                events.append(
                    HintBlockEvent(
                        reply_id=reply_id,
                        block_id=hint.id,
                        source=hint.source,
                        hint=hint.hint,
                    ),
                )
            event = CustomEvent(
                name="chat_input_injected",
                value={
                    "queue_item_id": item["id"],
                    "message_ids": [message.id for message in messages],
                },
            )
            await publish_session_event(
                self._bus,
                session_id,
                event.model_dump(mode="json"),
            )
        return True, events

    @staticmethod
    def _to_hints(item_id: str, messages: list[Msg]) -> list[HintBlock]:
        """Convert one queued turn into correlated multimodal hints.

        Args:
            item_id (`str`):
                Queue item used to derive stable block ids.
            messages (`list[Msg]`):
                Parsed messages carried by the queued turn.

        Returns:
            `list[HintBlock]`:
                One hint per message, preserving text and data blocks.

        Raises:
            `ValueError`:
                A message is empty or contains a block unsupported by hints.
        """
        hints: list[HintBlock] = []
        for index, message in enumerate(messages):
            content = [
                block
                for block in message.content
                if isinstance(block, (TextBlock, DataBlock))
            ]
            if not content or len(content) != len(message.content):
                raise ValueError(
                    f"Message '{message.id}' contains unsupported content.",
                )
            hints.append(
                HintBlock(
                    id=f"{item_id}:{index}",
                    hint=content,
                    source=_STEERING_HINT_SOURCE,
                ),
            )
        return hints

    @staticmethod
    def _append_hints(
        agent: Agent,
        hints: list[HintBlock],
    ) -> Callable[[], None]:
        """Append hints synchronously and return an exact rollback callback."""
        if agent.state.context:
            last_message = agent.state.context[-1]
            if (
                last_message.role == "assistant"
                and last_message.name == agent.name
                and last_message.id == agent.state.reply_id
            ):
                original_length = len(last_message.content)
                last_message.content.extend(hints)

                def rollback_extension() -> None:
                    del last_message.content[original_length:]

                return rollback_extension

        injected_message = AssistantMsg(
            id=agent.state.reply_id,
            name=agent.name,
            content=list(hints),
        )
        agent.state.context.append(injected_message)

        def rollback_message() -> None:
            if (
                agent.state.context
                and agent.state.context[-1] is injected_message
            ):
                agent.state.context.pop()

        return rollback_message

    @staticmethod
    def _message_ids(raw_input: object) -> list[str]:
        """Best-effort message ids for a visible preparation failure."""
        raw_messages = (
            raw_input if isinstance(raw_input, list) else [raw_input]
        )
        return [
            str(message.get("id"))
            for message in raw_messages
            if isinstance(message, dict) and message.get("id")
        ]
