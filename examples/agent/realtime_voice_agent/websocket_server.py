# -*- coding: utf-8 -*-
"""WebSocket server using WebSocketVoiceAgent.

This example demonstrates the correct architecture:
- WebSocketVoiceAgent: High-level agent with Memory and Toolkit support
- DashScopeWebSocketModel: WebSocket model for DashScope
- Unified event handling

Architecture:
    Browser → FastAPI WebSocket → WebSocketVoiceAgent →
    DashScopeWebSocketModel → DashScope

Usage:
    uvicorn websocket_server:app --host 0.0.0.0 --port 8000

Requirements:
    pip install fastapi uvicorn websockets
"""

import asyncio
import base64
import logging
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Agent and Model imports
from agentscope.agent.realtime_voice_agent import WebSocketVoiceAgent
from agentscope.agent.realtime_voice_agent.model import DashScopeWebSocketModel
from agentscope.agent.realtime_voice_agent._utils import MsgStream
from agentscope.memory import InMemoryMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VoiceAgent WebSocket Server")


class WebSocketVoiceSession:
    """Voice session using WebSocketVoiceAgent.

    Architecture:
    - WebSocketVoiceAgent: Handles Memory, Toolkit, and high-level logic
    - DashScopeWebSocketModel: WebSocket communication with DashScope
    - MsgStream: Message queue for Agent-Server communication
    """

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.agent: Optional[WebSocketVoiceAgent] = None
        self.model: Optional[DashScopeWebSocketModel] = None
        self.msg_stream: Optional[MsgStream] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

    @property
    def is_running(self) -> bool:
        """Check if the session is running."""
        return self._running

    async def initialize(self) -> None:
        """Initialize the voice agent and model."""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")

        # 1. Create MsgStream for communication
        self.msg_stream = MsgStream()

        # 2. Create WebSocket Model
        self.model = DashScopeWebSocketModel(
            api_key=api_key,
            model_name="qwen3-omni-flash-realtime",
            voice="Cherry",
            instructions="You are a helpful voice assistant."
            " Keep responses concise.",
            vad_enabled=True,
        )

        # 3. Create Agent with Memory
        self.agent = WebSocketVoiceAgent(
            name="assistant",
            model=self.model,
            sys_prompt="You are a helpful voice assistant.",
            msg_stream=self.msg_stream,
            memory=InMemoryMemory(),  # Enable Memory
            # toolkit=toolkit,  # Can add Toolkit here
        )

        # 4. Initialize agent (connects WebSocket)
        await self.agent.initialize()
        self._running = True
        logger.info("Session %s: Agent initialized", self.session_id)

    async def start(self) -> None:
        """Start the session tasks."""
        if not self.agent or not self.msg_stream:
            raise RuntimeError("Session not initialized")

        # Task 1: Agent reply loop
        self._tasks.append(
            asyncio.create_task(self._agent_loop()),
        )

        # Task 2: Forward MsgStream to WebSocket
        self._tasks.append(
            asyncio.create_task(self._forward_to_websocket()),
        )

        # Task 3: Receive from WebSocket and send to Agent
        self._tasks.append(
            asyncio.create_task(self._receive_from_websocket()),
        )

    async def _agent_loop(self) -> None:
        """Agent reply loop - continuously process and respond."""
        if not self.agent:
            return

        logger.info("Session %s: Agent loop started", self.session_id)
        try:
            while self._running:
                try:
                    # Agent processes input and generates response
                    response = await asyncio.wait_for(
                        self.agent.reply(),
                        timeout=60.0,  # Timeout for each reply cycle
                    )
                    if response:
                        text_content = response.get_text_content()
                        preview = (
                            text_content[:50] if text_content else "(audio)"
                        )
                        logger.info(
                            "Session %s: Agent replied: %s",
                            self.session_id,
                            preview,
                        )
                except asyncio.TimeoutError:
                    # No input for a while, continue waiting
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        "Session %s: Agent error: %s",
                        self.session_id,
                        e,
                    )
                    await asyncio.sleep(0.5)

        finally:
            logger.info("Session %s: Agent loop ended", self.session_id)

    async def _forward_to_websocket(self) -> None:
        """Forward MsgStream messages to WebSocket client."""
        if not self.msg_stream:
            return

        logger.info("Session %s: Forward loop started", self.session_id)
        msg_count = 0

        try:
            async for msg in self.msg_stream.subscribe(
                f"forward_{self.session_id}",
            ):
                if not self._running:
                    break

                msg_count += 1

                # Skip user messages (don't echo back)
                if msg.role == "user":
                    continue

                # Get metadata
                is_partial = (
                    msg.metadata.get("is_partial", True)
                    if msg.metadata
                    else True
                )

                # Process content blocks
                for block in msg.get_content_blocks():
                    block_type = block.get("type")

                    if block_type == "text":
                        # Note: The agent already sends accumulated text,
                        # so we don't need to accumulate here
                        text = block.get("text", "")

                        await self.websocket.send_json(
                            {
                                "type": "text",
                                "name": msg.name,
                                "data": text,
                                "is_partial": is_partial,
                            },
                        )

                        if not is_partial:
                            # Response complete
                            await self.websocket.send_json(
                                {
                                    "type": "event",
                                    "event": "response_end",
                                    "name": msg.name,
                                },
                            )
                            logger.info(
                                "Session %s: Response complete",
                                self.session_id,
                            )

                    elif block_type == "audio":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            audio_data = source.get("data", "")
                            media_type = source.get("media_type", "")

                            sample_rate = 24000
                            if "rate=" in media_type:
                                try:
                                    rate_str = media_type.split("rate=")[
                                        1
                                    ].split(";")[0]
                                    sample_rate = int(rate_str)
                                except (ValueError, IndexError):
                                    pass

                            await self.websocket.send_json(
                                {
                                    "type": "audio",
                                    "name": msg.name,
                                    "data": audio_data,
                                    "sample_rate": sample_rate,
                                },
                            )

        except Exception as e:
            logger.error("Session %s: Forward error: %s", self.session_id, e)
        finally:
            logger.info(
                "Session %s: Forward loop ended (%d messages)",
                self.session_id,
                msg_count,
            )

    async def _receive_from_websocket(self) -> None:
        """Receive audio from WebSocket and send to Agent."""
        if not self.model:
            return

        audio_chunk_count = 0

        try:
            while self._running:
                try:
                    data = await asyncio.wait_for(
                        self.websocket.receive_json(),
                        timeout=0.1,
                    )

                    if data.get("type") == "audio":
                        # Decode audio and send to model
                        audio_bytes = base64.b64decode(data["data"])
                        sample_rate = data.get("sample_rate", 16000)

                        audio_chunk_count += 1
                        if audio_chunk_count == 1:
                            logger.info(
                                "Session %s: First audio chunk received "
                                "(%d bytes)",
                                self.session_id,
                                len(audio_bytes),
                            )

                        # Send to model (Agent will process via model)
                        self.model.send_audio(audio_bytes, sample_rate)

                    elif data.get("type") == "control":
                        action = data.get("action")
                        logger.info(
                            "Session %s: Control: %s",
                            self.session_id,
                            action,
                        )
                        if action == "stop":
                            self._running = False
                            break
                        if action == "interrupt":
                            await self.model.cancel_response()

                except asyncio.TimeoutError:
                    continue

        except WebSocketDisconnect:
            logger.info("Session %s: WebSocket disconnected", self.session_id)
            self._running = False
        except Exception as e:
            logger.error("Session %s: Error receiving: %s", self.session_id, e)

    async def stop(self) -> None:
        """Stop the session and cleanup."""
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()

        # Close the agent (closes model internally)
        if self.agent:
            await self.agent.close()

        # Close MsgStream
        if self.msg_stream:
            await self.msg_stream.close()


# Store active sessions
sessions: dict[str, WebSocketVoiceSession] = {}


@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for voice conversations."""
    await websocket.accept()
    logger.info("Session %s: WebSocket connected", session_id)

    session = WebSocketVoiceSession(session_id, websocket)

    try:
        await websocket.send_json(
            {
                "type": "event",
                "event": "connected",
                "session_id": session_id,
            },
        )

        logger.info("Session %s: Initializing agent...", session_id)
        await session.initialize()
        sessions[session_id] = session
        logger.info("Session %s: Agent initialized successfully", session_id)

        await websocket.send_json(
            {
                "type": "event",
                "event": "ready",
                "message": "Voice agent initialized",
            },
        )

        await session.start()
        logger.info("Session %s: Session started", session_id)

        # Wait for session to end
        while session.is_running:
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info("Session %s: Client disconnected", session_id)
    except Exception as e:
        logger.error("Session %s: Error: %s", session_id, e, exc_info=True)
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                },
            )
        except Exception:
            pass
    finally:
        logger.info("Session %s: Cleaning up...", session_id)
        await session.stop()
        if session_id in sessions:
            del sessions[session_id]
        logger.info("Session %s: Session ended", session_id)


@app.get("/", response_class=HTMLResponse)
async def get_frontend() -> str:
    """Serve the frontend HTML page."""
    html_path = os.path.join(
        os.path.dirname(__file__),
        "websocket_client.html",
    )
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <body>
            <h1>VoiceAgent WebSocket Server</h1>
            <p>WebSocket endpoint: ws://localhost:8000/ws/voice/{
            session_id}</p>
            <p>Please use websocket_client.html for the full frontend.</p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "session_ids": list(sessions.keys()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
