# -*- coding: utf-8 -*-
"""Realtime WebSocket router — bidirectional audio bridge between the
browser and :class:`RealtimeAgent`.

The single ``WS /realtime/{session_id}`` endpoint manages the full
lifecycle: resolve the realtime model from the session's chat config,
create a :class:`RealtimeAgent`, and run two concurrent tasks:

- **Upstream** — read JSON frames from the browser (``{"type": "audio",
  "data": "<base64>"}``), wrap them in a ``DataBlock``, and forward to
  ``agent.send()``.
- **Downstream** — iterate ``agent.event_stream()``, serialise each
  ``AgentEvent``, and send it back as a JSON text frame.  Events are
  also published to the message bus so the SSE stream stays in sync.
  User transcriptions and assistant replies are persisted to the
  session's message store so they survive page refreshes.
"""
import asyncio
import json

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from .._service._realtime_model import get_realtime_model
from ..message_bus import MessageBus
from ..storage import StorageBase
from ..._logging import logger
from ...event import (
    ModelCallEndEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    TextBlockDeltaEvent,
    UserInputTranscriptionEvent,
)
from ...message import (
    AssistantMsg,
    Base64Source,
    DataBlock,
    TextBlock,
    Usage,
    UserMsg,
)
from ...realtime import RealtimeAgent

realtime_router = APIRouter(
    prefix="/realtime",
    tags=["realtime"],
)

_active_sessions: dict[str, WebSocket] = {}


@realtime_router.websocket("/{session_id}")
async def realtime_ws(
    websocket: WebSocket,
    session_id: str,
    agent_id: str = Query(...),
    user_id: str = Query(...),
) -> None:
    """Bidirectional audio bridge for a realtime session.

    Browsers cannot set custom headers on WebSocket connections, so
    ``user_id`` is passed as a query parameter (same security posture
    as the temporary ``X-User-ID`` header used elsewhere).
    """
    storage: StorageBase = websocket.app.state.storage
    message_bus: MessageBus = websocket.app.state.message_bus

    # ---- Validate ownership ----
    session_record = await storage.get_session(
        user_id,
        agent_id,
        session_id,
    )
    if session_record is None:
        await websocket.close(code=4004, reason="Session not found.")
        return

    agent_record = await storage.get_agent(user_id, agent_id)
    if agent_record is None:
        await websocket.close(code=4004, reason="Agent not found.")
        return

    # ---- Prevent duplicate connections ----
    if session_id in _active_sessions:
        await websocket.close(
            code=4009,
            reason="Realtime session already active.",
        )
        return

    # ---- Resolve realtime model ----
    model_cfg = session_record.config.realtime_model_config
    if not model_cfg:
        await websocket.close(
            code=4000,
            reason="No realtime model configured on this session.",
        )
        return

    try:
        model = await get_realtime_model(user_id, model_cfg, storage)
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))
        return

    # ---- Accept connection ----
    await websocket.accept()
    _active_sessions[session_id] = websocket

    # ---- Create agent ----
    agent = RealtimeAgent(
        name=agent_record.data.name,
        model=model,
        instructions=agent_record.data.system_prompt,
        session_id=session_id,
    )

    try:
        await agent.connect()

        upstream_task = asyncio.create_task(
            _upstream(websocket, agent, model.input_sample_rate),
            name=f"rt-upstream:{session_id}",
        )
        downstream_task = asyncio.create_task(
            _downstream(
                websocket,
                agent,
                message_bus,
                storage,
                user_id,
                session_id,
                agent_name=agent_record.data.name,
            ),
            name=f"rt-downstream:{session_id}",
        )

        done, pending = await asyncio.wait(
            [upstream_task, downstream_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            if task.exception() and not isinstance(
                task.exception(),
                (WebSocketDisconnect, asyncio.CancelledError),
            ):
                logger.error(
                    "Realtime task error for session %s: %s",
                    session_id,
                    task.exception(),
                )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception(
            "Realtime session error for %s",
            session_id,
        )
    finally:
        await agent.disconnect()
        _active_sessions.pop(session_id, None)


async def _upstream(
    websocket: WebSocket,
    agent: RealtimeAgent,
    input_sample_rate: int,
) -> None:
    """Read audio frames from the browser and forward to the agent."""
    while True:
        raw = await websocket.receive_text()
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if frame.get("type") == "audio":
            data = frame.get("data", "")
            if data:
                await agent.send(
                    DataBlock(
                        source=Base64Source(
                            data=data,
                            media_type=f"audio/pcm;rate={input_sample_rate}",
                        ),
                    ),
                )


async def _downstream(
    websocket: WebSocket,
    agent: RealtimeAgent,
    message_bus: MessageBus,
    storage: StorageBase,
    user_id: str,
    session_id: str,
    *,
    agent_name: str,
) -> None:
    """Forward agent events to the browser, publish to the bus, and
    persist user/assistant messages to the session's message store."""

    reply_id: str | None = None
    reply_name: str = agent_name
    reply_text_parts: list[str] = []
    reply_usage: Usage | None = None
    reply_created_at: str | None = None

    async for event in agent.event_stream():
        payload = event.model_dump(mode="json")
        try:
            await websocket.send_json(payload)
        except (WebSocketDisconnect, RuntimeError):
            return
        await message_bus.session_publish_event(session_id, payload)

        # ---- Persist messages ----
        if isinstance(event, UserInputTranscriptionEvent):
            if event.transcript:
                msg = UserMsg(name="user", content=event.transcript)
                try:
                    await storage.upsert_message(user_id, session_id, msg)
                except Exception:
                    logger.warning(
                        "Failed to persist user transcript for session %s",
                        session_id,
                        exc_info=True,
                    )

        elif isinstance(event, ReplyStartEvent):
            reply_id = event.reply_id
            reply_name = event.name
            reply_text_parts = []
            reply_usage = None
            reply_created_at = event.created_at.isoformat()

        elif isinstance(event, TextBlockDeltaEvent):
            reply_text_parts.append(event.delta)

        elif isinstance(event, ModelCallEndEvent):
            reply_usage = Usage(
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
            )

        elif isinstance(event, ReplyEndEvent):
            if reply_id is not None:
                text = "".join(reply_text_parts)
                content: list[TextBlock] = (
                    [TextBlock(text=text)] if text else []
                )
                msg = AssistantMsg(
                    name=reply_name,
                    content=content,
                    id=reply_id,
                    created_at=reply_created_at,
                    finished_at=event.created_at.isoformat(),
                    usage=reply_usage,
                )
                try:
                    await storage.upsert_message(
                        user_id,
                        session_id,
                        msg,
                    )
                except Exception:
                    logger.warning(
                        "Failed to persist assistant reply for session %s",
                        session_id,
                        exc_info=True,
                    )
                reply_id = None
                reply_text_parts = []
                reply_usage = None
