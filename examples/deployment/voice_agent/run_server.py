# -*- coding: utf-8 -*-
"""A test server"""
import asyncio
import os
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from agentscope.agent import RealtimeAgentBase
from agentscope.realtime import (
    DashScopeRealtimeModel,
    ClientEvents,
    ServerEvents,
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


# pylint: disable=too-many-statements, too-many-branches, unused-argument
@app.websocket("/ws/{user_id}/{session_id}")
async def single_agent_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
) -> None:
    """WebSocket endpoint for a single realtime agent."""
    try:
        await websocket.accept()

        # Create the queue to forward messages to the frontend
        frontend_queue = asyncio.Queue()
        asyncio.create_task(
            frontend_receive(websocket, frontend_queue),
        )

        # Create the realtime agent
        agent = RealtimeAgentBase(
            name="Friday",
            sys_prompt="You are a helpful assistant.",
            model=DashScopeRealtimeModel(
                model_name="qwen3-omni-flash-realtime",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
            ),
        )

        try:
            await agent.start(frontend_queue)

        except Exception as e:
            print(f"[ERROR] Failed to start agent: {e}")
            traceback.print_exc()
            raise

        try:
            while True:
                # Handle the incoming messages from the frontend
                # i.e. ClientEvents
                data = await websocket.receive_json()

                try:
                    client_event = ClientEvents.from_json(data)

                    if (
                        client_event.type
                        == ClientEvents.ClientSessionCreateEvent
                    ):
                        print("收到 ClientSessionCreateEvent")

                    elif (
                        client_event.type == ClientEvents.ClientSessionEndEvent
                    ):
                        pass
                    else:
                        await agent.handle_input(client_event)

                except Exception as e:
                    print(
                        f"[ERROR] Failed to parse client event: {e}",
                    )

        except WebSocketDisconnect:
            pass

        finally:
            # Clear the resources
            try:
                await agent.stop()
            except Exception as e:
                print(f"[ERROR] Error stopping agent: {e}")

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
