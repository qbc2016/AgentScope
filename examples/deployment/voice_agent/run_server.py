# -*- coding: utf-8 -*-
"""A test server"""
import asyncio
import os
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from agentscope import logger
from agentscope.agent import RealtimeAgentBase
from agentscope.realtime import (
    DashScopeRealtimeModel,
    ClientEvents,
    ServerEvents,
    ClientEventType,
)

app = FastAPI()


@app.get("/")
async def get() -> FileResponse:
    """Serve the HTML test page."""
    html_path = Path(__file__).parent / "chatbot.html"
    return FileResponse(html_path)


async def frontend_receive(
    websocket: WebSocket,
    frontend_queue: asyncio.Queue,
) -> None:
    """Forward the message received from the agent to the frontend."""
    try:
        while True:
            msg: ServerEvents.EventBase = await frontend_queue.get()

            # Send the message as JSON
            await websocket.send_json(msg.model_dump())

    except Exception as e:
        print(f"[ERROR] frontend_receive error: {e}")
        traceback.print_exc()


@app.websocket("/ws/{user_id}/{session_id}")
async def single_agent_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
) -> None:
    """WebSocket endpoint for a single realtime agent."""
    try:
        await websocket.accept()

        logger.info(
            "Connected to WebSocket: user_id=%s, session_id=%s",
            user_id,
            session_id,
        )

        # Create the queue to forward messages to the frontend
        frontend_queue = asyncio.Queue()
        asyncio.create_task(
            frontend_receive(websocket, frontend_queue),
        )

        # Create the realtime agent
        agent = None

        while True:
            # Handle the incoming messages from the frontend
            # i.e. ClientEvents
            data = await websocket.receive_json()

            client_event = ClientEvents.from_json(data)

            if isinstance(
                client_event,
                ClientEvents.ClientSessionCreateEvent,
            ):
                # Create the agent by the given session arguments
                instructions = client_event.config.get("instructions", "You're a helpful assistant.")
                user_name = client_event.config.get("user_name", "User")

                sys_prompt = (
                    f"{instructions}\n"
                    f"You're talking to the user named {user_name}."
                )

                # Create the agent
                agent = RealtimeAgentBase(
                    name="Friday",
                    sys_prompt=sys_prompt,
                    model=DashScopeRealtimeModel(
                        model_name="qwen3-omni-flash-realtime",
                        api_key=os.getenv("DASHSCOPE_API_KEY"),
                    ),
                )

                await agent.start(frontend_queue)

                # Send session_created event to frontend
                await websocket.send_json(
                    ServerEvents.SessionCreatedEvent(
                        session_id=session_id,
                    ).model_dump(),
                )
                print(
                    f"Session created successfully: {session_id}",
                )

            elif client_event.type == ClientEventType.CLIENT_SESSION_END:
                # End the session with the agent
                if agent:
                    await agent.stop()
                    agent = None

            else:
                await agent.handle_input(client_event)

    except Exception as e:
        print(f"[ERROR] WebSocket endpoint error: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    uvicorn.run(
        "run_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
