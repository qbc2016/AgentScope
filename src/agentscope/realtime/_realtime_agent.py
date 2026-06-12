# -*- coding: utf-8 -*-
"""Thin agent that wraps a :class:`RealtimeModelBase` and exposes its
incoming ``ModelEvents`` as a public ``AgentEvent`` stream.

Unlike :class:`agentscope.agent.Agent`, the realtime agent is bidirectional
and has no request/reply boundary: the caller pushes user audio via
:meth:`send`, while :meth:`event_stream` continuously yields the agent's
events translated from the model's WebSocket frames.
"""
import asyncio
import uuid
from asyncio import Queue
from typing import AsyncGenerator, Any, cast

from ._base import RealtimeModelBase
from ._events import ModelEvents
from .._logging import logger
from ..event import (
    AgentEvent,
    DataBlockDeltaEvent,
    DataBlockEndEvent,
    DataBlockStartEvent,
    ModelCallEndEvent,
    ModelCallStartEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    TextBlockStartEvent,
    UserInputAudioEndEvent,
    UserInputAudioStartEvent,
    UserInputTranscriptionEvent,
)
from ..message import DataBlock, TextBlock, ToolResultBlock


class RealtimeAgent:
    """A thin translator between a :class:`RealtimeModelBase` and the public
    :class:`AgentEvent` stream.

    Example:

        .. code-block:: python

            async with RealtimeAgent(
                name="Friday",
                model=DashScopeRealtimeModel(...),
                instructions="You are a helpful assistant.",
            ) as agent:
                # Drain events in one task...
                async def consume():
                    async for evt in agent.event_stream():
                        print(evt)

                consumer = asyncio.create_task(consume())

                # ...and push input audio in another.
                await agent.send(audio_data_block)

                await consumer
    """

    def __init__(
        self,
        name: str,
        model: RealtimeModelBase,
        instructions: str,
        session_id: str | None = None,
    ) -> None:
        """Initialize the realtime agent.

        Args:
            name (`str`):
                Display name of the agent (used in ``ReplyStartEvent.name``).
            model (`RealtimeModelBase`):
                The realtime model client.
            instructions (`str`):
                System instructions forwarded to the model on connect.
            session_id (`str | None`, optional):
                Session id stamped on every emitted event. A new uuid is
                generated if omitted.
        """
        self.name = name
        self.model = model
        self.instructions = instructions
        self.session_id = session_id or uuid.uuid4().hex

        self._model_queue: Queue = Queue()
        self._connected = False

        # Per-response bookkeeping: response_id (used as reply_id) → block IDs
        self._audio_blocks: dict[str, str] = {}
        self._text_blocks: dict[str, str] = {}
        self._audio_media_types: dict[str, str] = {}
        # response_ids for which we have already emitted ReplyStartEvent
        self._started_responses: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "RealtimeAgent":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc: Exception | None,
        tb: Any,
    ) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        """Open the underlying realtime model session."""
        if self._connected:
            return
        await self.model.connect(
            outgoing_queue=self._model_queue,
            instructions=self.instructions,
        )
        self._connected = True

    async def disconnect(self) -> None:
        """Close the underlying realtime model session."""
        if not self._connected:
            return
        await self.model.disconnect()
        self._connected = False

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    async def send(
        self,
        data: DataBlock | TextBlock | ToolResultBlock,
    ) -> None:
        """Forward a content block to the underlying realtime model.

        Args:
            data (`DataBlock | TextBlock | ToolResultBlock`):
                Same semantics as :meth:`RealtimeModelBase.send`.
        """
        await self.model.send(data)

    async def event_stream(self) -> AsyncGenerator[AgentEvent, None]:
        """Yield translated :class:`AgentEvent` instances until the session
        ends (model disconnects or caller cancels).
        """
        while True:
            try:
                model_evt = await self._model_queue.get()
            except asyncio.CancelledError:
                return

            for agent_evt in self._translate(model_evt):
                yield agent_evt

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    # pylint: disable=too-many-return-statements
    def _translate(
        self,
        evt: ModelEvents.EventBase,
    ) -> list[AgentEvent]:
        """Map one ``ModelEvents.*`` instance into 0..N ``AgentEvent``s."""
        sid = self.session_id

        # ---- Session ----
        if isinstance(evt, ModelEvents.ModelSessionCreatedEvent):
            logger.debug(
                "RealtimeAgent: model session created (id=%s)",
                evt.session_id,
            )
            return []
        if isinstance(evt, ModelEvents.ModelSessionEndedEvent):
            logger.debug(
                "RealtimeAgent: model session ended (reason=%s)",
                evt.reason,
            )
            return []

        # ---- User input / VAD ----
        if isinstance(evt, ModelEvents.ModelInputStartedEvent):
            return [
                UserInputAudioStartEvent(
                    session_id=sid,
                    item_id=evt.item_id,
                    audio_start_ms=evt.audio_start_ms,
                ),
            ]
        if isinstance(evt, ModelEvents.ModelInputDoneEvent):
            return [
                UserInputAudioEndEvent(
                    session_id=sid,
                    item_id=evt.item_id,
                    audio_end_ms=evt.audio_end_ms,
                ),
            ]
        if isinstance(evt, ModelEvents.ModelInputTranscriptionDoneEvent):
            return [
                UserInputTranscriptionEvent(
                    session_id=sid,
                    item_id=evt.item_id,
                    transcript=evt.transcript,
                ),
            ]
        if isinstance(evt, ModelEvents.ModelInputTranscriptionDeltaEvent):
            # No public AgentEvent equivalent for partial input transcripts;
            # the final transcription is emitted via
            # UserInputTranscriptionEvent.
            return []

        # ---- Response lifecycle ----
        if isinstance(evt, ModelEvents.ModelResponseCreatedEvent):
            reply_id = evt.response_id
            return self._ensure_reply_started(reply_id)
        if isinstance(evt, ModelEvents.ModelResponseDoneEvent):
            reply_id = evt.response_id
            out: list[AgentEvent] = []
            out.extend(self._ensure_reply_started(reply_id))
            out.extend(self._close_text_block(reply_id))
            out.extend(self._close_audio_block(reply_id))
            out.append(
                ModelCallEndEvent(
                    reply_id=reply_id,
                    input_tokens=evt.input_tokens,
                    output_tokens=evt.output_tokens,
                ),
            )
            out.append(
                ReplyEndEvent(session_id=sid, reply_id=reply_id),
            )
            self._started_responses.discard(reply_id)
            return out

        # ---- Audio output ----
        if isinstance(evt, ModelEvents.ModelResponseAudioDeltaEvent):
            reply_id = evt.response_id
            media_type = f"{evt.format.type};rate={evt.format.rate}"
            out = cast(list[AgentEvent], [])
            out.extend(self._ensure_reply_started(reply_id))
            block_id = self._audio_blocks.get(reply_id)
            if block_id is None:
                block_id = uuid.uuid4().hex
                self._audio_blocks[reply_id] = block_id
                self._audio_media_types[reply_id] = media_type
                out.append(
                    DataBlockStartEvent(
                        reply_id=reply_id,
                        block_id=block_id,
                        media_type=media_type,
                    ),
                )
            out.append(
                DataBlockDeltaEvent(
                    reply_id=reply_id,
                    block_id=block_id,
                    data=evt.delta,
                    media_type=self._audio_media_types[reply_id],
                ),
            )
            return out
        if isinstance(evt, ModelEvents.ModelResponseAudioDoneEvent):
            return self._close_audio_block(evt.response_id)

        # ---- Transcript output ----
        if isinstance(
            evt,
            ModelEvents.ModelResponseAudioTranscriptDeltaEvent,
        ):
            reply_id = evt.response_id
            out = cast(list[AgentEvent], [])
            out.extend(self._ensure_reply_started(reply_id))
            block_id = self._text_blocks.get(reply_id)
            if block_id is None:
                block_id = uuid.uuid4().hex
                self._text_blocks[reply_id] = block_id
                out.append(
                    TextBlockStartEvent(
                        reply_id=reply_id,
                        block_id=block_id,
                    ),
                )
            out.append(
                TextBlockDeltaEvent(
                    reply_id=reply_id,
                    block_id=block_id,
                    delta=evt.delta,
                ),
            )
            return out
        if isinstance(
            evt,
            ModelEvents.ModelResponseAudioTranscriptDoneEvent,
        ):
            return self._close_text_block(evt.response_id)

        # ---- Tool calls (model doesn't emit these for DashScope yet, but
        #      keep the translation in place so other vendors light up) ----
        if isinstance(evt, ModelEvents.ModelResponseToolCallDeltaEvent):
            # We don't yet expose realtime tool calls on the public stream;
            # a future RealtimeAgent revision will route these through the
            # standard ToolCall*/ToolResult* events.
            return []
        if isinstance(evt, ModelEvents.ModelResponseToolCallDoneEvent):
            return []

        # ---- Error ----
        if isinstance(evt, ModelEvents.ModelErrorEvent):
            logger.error(
                "RealtimeAgent: model error %s/%s: %s",
                evt.error_type,
                evt.code,
                evt.message,
            )
            return []

        logger.debug(
            "RealtimeAgent: unhandled model event %s",
            type(evt).__name__,
        )
        return []

    # ------------------------------------------------------------------
    # Block close helpers
    # ------------------------------------------------------------------

    def _ensure_reply_started(self, reply_id: str) -> list[AgentEvent]:
        """Emit ``ReplyStartEvent`` + ``ModelCallStartEvent`` for *reply_id*
        if we haven't done so yet.  This covers providers (e.g. ElevenLabs)
        that never send an explicit ``response.created`` frame."""
        if reply_id in self._started_responses:
            return []
        self._started_responses.add(reply_id)
        return [
            ReplyStartEvent(
                session_id=self.session_id,
                reply_id=reply_id,
                name=self.name,
            ),
            ModelCallStartEvent(
                reply_id=reply_id,
                model_name=self.model.model_name,
            ),
        ]

    def _close_audio_block(self, reply_id: str) -> list[AgentEvent]:
        block_id = self._audio_blocks.pop(reply_id, None)
        self._audio_media_types.pop(reply_id, None)
        if block_id is None:
            return []
        return [DataBlockEndEvent(reply_id=reply_id, block_id=block_id)]

    def _close_text_block(self, reply_id: str) -> list[AgentEvent]:
        block_id = self._text_blocks.pop(reply_id, None)
        if block_id is None:
            return []
        return [TextBlockEndEvent(reply_id=reply_id, block_id=block_id)]
