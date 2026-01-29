# -*- coding: utf-8 -*-
"""A test server"""
import asyncio
import os
import queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from agentscope.agent import RealtimeAgentBase
from agentscope.realtime import DashScopeRealtimeModel

# 简单的 HTML 测试页面
html = """
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Demo</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin:
        50px auto; }
        #messages { border: 1px solid #ccc; height: 300px; overflow-y:
        scroll; padding: 10px; margin: 20px 0; }
        input { width: 70%; padding: 10px; }
        button { padding: 10px 20px; }
        .message { margin: 5px 0; padding: 5px; background: #f0f0f0;
        border-radius: 3px; }
    </style>
</head>
<body>
    <h1>WebSocket 聊天室</h1>
    <div>
        <input type="text" id="userId" placeholder="输入你的名字" value="User1" />
    </div>
    <div id="messages"></div>
    <div>
        <input type="text" id="messageText" placeholder="输入消息..." />
        <button onclick="sendMessage()">发送</button>
        <button onclick="disconnect()">断开连接</button>
    </div>

    <script>
        let ws = null;
        let userId = document.getElementById("userId").value;

        function connect() {
            userId = document.getElementById("userId").value;
            ws = new WebSocket(`ws://localhost:8000/ws/${userId}/session1`);

            ws.onopen = function(event) {
                addMessage("系统", "已连接到服务器");
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                addMessage("系统", data);
            };

            ws.onclose = function(event) {
                addMessage("系统", "已断开连接");
            };

            ws.onerror = function(error) {
                addMessage("系统", "连接错误");
            };
        }

        function sendMessage() {
            const input = document.getElementById("messageText");
            if (ws && input.value) {
                ws.send(JSON.stringify({type: "delta", data: [{type:
                "text", text: input.value}]}));
                input.value = '';
            }
        }

        function disconnect() {
            if (ws) {
                ws.close();
            }
        }

        function addMessage(sender, message) {
            const messagesDiv = document.getElementById("messages");
            const messageDiv = document.createElement("div");
            messageDiv.className = "message";
            messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // 回车发送消息
        document.getElementById("messageText").addEventListener(
        "keypress", function(event) {
            if (event.key === "Enter") {
                sendMessage();
            }
        });

        // 页面加载时自动连接
        window.onload = connect;
    </script>
</body>
</html>
"""

app = FastAPI()


@app.get("/")
async def get() -> HTMLResponse:
    """Serve the HTML test page."""
    return HTMLResponse(html)


@app.websocket("/ws/{user_id}/{session_id}")
async def single_agent_endpoint(
    websocket: WebSocket,
    _user_id: str,
    _session_id: str,
) -> None:
    """WebSocket endpoint for a single realtime agent."""
    await websocket.accept()

    # 发往前端的消息队列
    frontend_queue = queue.Queue()

    async def frontend_receive() -> None:
        while True:
            msg = await frontend_queue.get()
            # TODO: 处理sample rate统一成一个
            await websocket.send_json(msg)

    asyncio.create_task(frontend_receive())

    model = DashScopeRealtimeModel(
        model_name="qwen3-omni-flash-realtime",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    agent = RealtimeAgentBase(
        name="Friday",
        sys_prompt="You are a helpful assistant.",
        model=model,
    )

    # or
    # chatroom = ChatRoom(agents=[agent, ...])

    await agent.start(frontend_queue)
    # or
    # await chatroom.start(frontend_queue)

    try:
        while True:
            data = await websocket.receive_json()

            await agent.handle_input(data)
            # or
            # await chatroom.handle_frontend_event(data)

    except WebSocketDisconnect:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "run_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
    )
