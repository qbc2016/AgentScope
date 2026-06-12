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

The :class:`RealtimeAgent` handles tool execution internally when a
:class:`~agentscope.tool.Toolkit` is provided.
"""
import asyncio
import json

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from .._service._realtime_model import get_realtime_model
from ..message_bus import MessageBus
from ..storage import StorageBase
from ..workspace_manager._base import WorkspaceManagerBase
from ..._logging import logger
from ...event import (
    ModelCallEndEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    TextBlockDeltaEvent,
    UserConfirmResultEvent,
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
from ...tool import Toolkit

realtime_router = APIRouter(
    prefix="/realtime",
    tags=["realtime"],
)

_active_sessions: dict[str, WebSocket] = {}


async def _build_realtime_toolkit(
    workspace_manager: WorkspaceManagerBase,
    user_id: str,
    agent_id: str,
    session_id: str,
    workspace_id: str,
) -> Toolkit | None:
    """Build a lightweight :class:`Toolkit` for realtime tool execution.

    Returns ``None`` if the workspace cannot be resolved (e.g. no
    workspace configured).
    """
    try:
        workspace = await workspace_manager.get_workspace(
            user_id,
            agent_id,
            session_id,
            workspace_id,
        )
        tools = await workspace.list_tools()
        if not tools:
            return None
        return Toolkit(tools=tools)
    except Exception:
        logger.debug(
            "Could not build realtime toolkit for session %s",
            session_id,
            exc_info=True,
        )
        return None


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
    workspace_manager: WorkspaceManagerBase = (
        websocket.app.state.workspace_manager
    )

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

    # ---- Build toolkit for tool execution ----
    toolkit: Toolkit | None = None
    if model.support_tools:
        workspace_id = session_record.config.workspace_id or agent_id
        toolkit = await _build_realtime_toolkit(
            workspace_manager,
            user_id,
            agent_id,
            session_id,
            workspace_id,
        )
        if toolkit is None:
            logger.info(
                "Realtime session %s: model supports tools but no "
                "workspace tools found (workspace_id=%s).",
                session_id,
                workspace_id,
            )
    else:
        logger.debug(
            "Realtime session %s: model %s does not support tools.",
            session_id,
            model.model_name,
        )

    # ---- Accept connection ----
    await websocket.accept()
    _active_sessions[session_id] = websocket

    # ---- Create agent (reuse persisted state for context continuity) ----
    agent_state = session_record.state
    agent_state.session_id = session_id

    agent = RealtimeAgent(
        name=agent_record.data.name,
        model=model,
        instructions=agent_record.data.system_prompt,
        session_id=session_id,
        toolkit=toolkit,
        state=agent_state,
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
        # Persist the updated state so context survives reconnection
        # and is available when switching to text chat mode.
        try:
            await storage.update_session_state(
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                state=agent.state,
            )
        except Exception:
            logger.warning(
                "Failed to persist realtime agent state for session %s",
                session_id,
                exc_info=True,
            )
        _active_sessions.pop(session_id, None)


async def _upstream(
    websocket: WebSocket,
    agent: RealtimeAgent,
    input_sample_rate: int,
) -> None:
    """Read frames from the browser and forward to the agent.

    Supported frame types:

    - ``audio`` — base64 PCM audio forwarded via ``agent.send()``.
    - ``user_confirm`` — tool-call confirmation forwarded to
      ``agent.handle_user_confirm()`` so the pending permission
      future is resolved.
    """
    while True:
        raw = await websocket.receive_text()
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            continue

        frame_type = frame.get("type")

        if frame_type == "audio":
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

        elif frame_type == "user_confirm":
            try:
                event = UserConfirmResultEvent(**frame.get("data", {}))
                agent.handle_user_confirm(event)
            except Exception:
                logger.warning(
                    "Failed to process user_confirm frame",
                    exc_info=True,
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
    persist user/assistant messages to the session's message store.

    Tool execution is handled internally by the :class:`RealtimeAgent`
    when a toolkit is configured.
    """
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

        # ---- Persist messages to storage ----
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
            reply_created_at = event.created_at

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
                    finished_at=event.created_at,
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
