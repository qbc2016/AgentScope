# -*- coding: utf-8 -*-
"""WebSocket-based Voice Agent for real-time voice models.

This agent is designed to work with WebSocketVoiceModelBase and its subclasses
(DashScopeWebSocketModel, GeminiWebSocketModel, OpenAIWebSocketModel).
"""

import asyncio
import json

from ....message import Msg, TextBlock, ToolUseBlock, ToolResultBlock
from ....memory import MemoryBase, InMemoryMemory
from ....tool import Toolkit
from ....module import StateModule

from .._utils import MsgStream, create_msg
from ..model import WebSocketVoiceModelBase, LiveEventType

from ...._logging import logger


class WebSocketVoiceAgent(StateModule):
    """Voice agent for WebSocket-based real-time models.

    This agent works with Formatter-based models:
    - DashScopeWebSocketModel
    - GeminiWebSocketModel
    - OpenAIWebSocketModel

    Features:
    - Memory support for conversation history
    - Toolkit support for tool calling
    - MsgStream for message communication
    - No local audio playback (designed for server-side use)

    Example:
        ```python
        from agentscope.agent.realtime_voice_agent import WebSocketVoiceAgent
        from agentscope.agent.realtime_voice_agent.model import
        DashScopeWebSocketModel
        from agentscope.memory import InMemoryMemory

        model = DashScopeWebSocketModel(
            api_key="your-api-key",
            model_name="qwen3-omni-flash-realtime",
        )

        agent = WebSocketVoiceAgent(
            name="assistant",
            model=model,
            memory=InMemoryMemory(),
        )

        await agent.initialize()
        response = await agent.reply()
        ```
    """

    def __init__(
        self,
        name: str,
        model: WebSocketVoiceModelBase,
        sys_prompt: str = "You are a helpful assistant.",
        toolkit: Toolkit | None = None,
        msg_stream: MsgStream | None = None,
        memory: MemoryBase | None = None,
    ) -> None:
        """Initialize the WebSocket Voice Agent.

        Args:
            name: Agent name.
            model: WebSocketVoiceModelBase instance (or subclass).
            sys_prompt: System prompt for the agent.
            toolkit: Optional toolkit for tool calling.
            msg_stream: Optional MsgStream for external communication.
            memory: Optional memory for conversation history.
        """
        super().__init__()
        self.name = name
        self.model = model
        self.sys_prompt = sys_prompt
        self.toolkit = toolkit or Toolkit()
        self.msg_stream = msg_stream or MsgStream()
        self.memory = memory or InMemoryMemory()

        self._initialized = False
        self._stop_event = asyncio.Event()
        self._listen_task: asyncio.Task | None = None

        # Response state
        self._response_text = ""
        self._streaming_msg: Msg | None = None

    async def initialize(self) -> None:
        """Initialize the agent and model.

        Sets up callbacks and starts the model connection.
        """
        if self._initialized:
            return

        # Initialize the model (connects WebSocket)
        await self.model.initialize()

        # Start listening for model events
        self._listen_task = asyncio.create_task(self._listen_loop())

        self._initialized = True
        logger.info("WebSocketVoiceAgent %s initialized", self.name)

    async def _handle_input_transcription(self, text: str) -> None:
        """Handle user input transcription and save to memory.

        Args:
            text: Transcribed user speech.
        """
        if not text:
            return

        user_msg = Msg(
            name="user",
            content=[TextBlock(type="text", text=text)],
            role="user",
        )
        await self.memory.add(user_msg)

    async def _handle_tool_call(
        self,
        tool_id: str,
        tool_name: str,
        arguments: dict,
    ) -> None:
        """Handle tool call from model.

        Args:
            tool_id: The tool call ID.
            tool_name: The name of the tool.
            arguments: The tool arguments.
        """
        if not self.toolkit:
            logger.warning(
                "Tool call received but no toolkit: %s",
                tool_name,
            )
            return

        logger.info("Executing tool: %s(%s)", tool_name, tool_id)
        await self._execute_tool(tool_id, tool_name, arguments)

    async def _execute_tool(
        self,
        tool_id: str,
        tool_name: str,
        arguments: dict,
    ) -> None:
        """Execute a tool and send the result back to the model.

        Args:
            tool_id: The tool call ID.
            tool_name: The name of the tool.
            arguments: The tool arguments.
        """
        try:
            tool_use_block = ToolUseBlock(
                type="tool_use",
                id=tool_id,
                name=tool_name,
                input=arguments,
            )
            # Save tool call to memory
            tool_call_msg = Msg(
                name=self.name,
                content=[
                    tool_use_block,
                ],
                role="assistant",
            )
            await self.memory.add(tool_call_msg)

            result = await self.toolkit.call_tool_function(tool_use_block)

            # Convert result to string
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result, ensure_ascii=False)
            else:
                result_str = str(result)

            # Save tool result to memory
            tool_result_msg = Msg(
                name="system",
                content=[
                    ToolResultBlock(
                        type="tool_result",
                        id=tool_id,
                        name=tool_name,
                        output=[TextBlock(type="text", text=result_str)],
                    ),
                ],
                role="system",
            )
            await self.memory.add(tool_result_msg)

            # Send result back to model
            await self.model.send_tool_result(tool_id, tool_name, result_str)
            logger.info("Tool %s executed successfully", tool_name)

        except Exception as e:
            logger.error("Tool execution error: %s - %s", tool_name, e)
            # Send error result
            await self.model.send_tool_result(
                tool_id,
                tool_name,
                f"Error: {e}",
            )

    # pylint: disable=too-many-nested-blocks, too-many-branches
    # pylint: disable=too-many-statements
    async def _listen_loop(self) -> None:
        """Listen for model events and forward to MsgStream."""
        logger.info("WebSocketVoiceAgent %s: Listen loop started", self.name)

        try:
            async for event in self.model.iter_events():
                if self._stop_event.is_set():
                    break

                event_type = event.type

                # Handle text delta
                if event_type == LiveEventType.TEXT_DELTA:
                    if event.message:
                        for block in event.message.get_content_blocks():
                            if block.get("type") == "text":
                                text = block["text"]
                                self._response_text += text
                                await self._push_streaming_msg(
                                    text=self._response_text,
                                )

                # Handle output transcription
                elif event_type == LiveEventType.OUTPUT_TRANSCRIPTION:
                    if event.message:
                        for block in event.message.get_content_blocks():
                            if block.get("type") == "text":
                                text = block["text"]
                                self._response_text += text
                                await self._push_streaming_msg(
                                    text=self._response_text,
                                )

                # Handle audio delta
                elif event_type == LiveEventType.AUDIO_DELTA:
                    if event.message:
                        for block in event.message.get_content_blocks():
                            if block.get("type") == "audio":
                                source = block.get("source", {})
                                if source.get("type") == "base64":
                                    audio_data = source.get("data", "")
                                    media_type = source.get("media_type", "")
                                    sample_rate = 24000
                                    if "rate=" in media_type:
                                        try:
                                            rate_str = media_type.split(
                                                "rate=",
                                            )[1].split(";")[0]
                                            sample_rate = int(rate_str)
                                        except (ValueError, IndexError):
                                            pass
                                    await self._push_audio_msg(
                                        audio_data,
                                        sample_rate,
                                    )

                # Handle input transcription (save user input to memory)
                elif event_type == LiveEventType.INPUT_TRANSCRIPTION:
                    if event.message:
                        for block in event.message.get_content_blocks():
                            if block.get("type") == "text":
                                text = block["text"]
                                await self._handle_input_transcription(text)

                # Handle tool call (execute tool and send result)
                elif event_type == LiveEventType.TOOL_CALL:
                    if event.message:
                        for block in event.message.get_content_blocks():
                            if block.get("type") == "tool_use":
                                tool_id = block["id"]
                                tool_name = block["name"]
                                tool_input = block["input"]
                                if isinstance(tool_input, dict):
                                    await self._handle_tool_call(
                                        tool_id,
                                        tool_name,
                                        tool_input,
                                    )

                # Handle response complete
                elif event_type in (
                    LiveEventType.RESPONSE_DONE,
                    LiveEventType.TURN_COMPLETE,
                ):
                    await self._finalize_response()

                # Handle errors
                elif event_type == LiveEventType.ERROR:
                    error_msg = event.metadata.get(
                        "error_message",
                        "Unknown error",
                    )
                    logger.error("Model error: %s", error_msg)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Listen loop error: %s", e)
        finally:
            logger.info("WebSocketVoiceAgent %s: Listen loop ended", self.name)

    async def _push_streaming_msg(self, text: str) -> None:
        """Push a streaming text message to MsgStream."""
        msg = create_msg(
            name=self.name,
            text=text,
            is_partial=True,
        )
        await self.msg_stream.push(msg)

    async def _push_audio_msg(self, audio_b64: str, sample_rate: int) -> None:
        """Push an audio message to MsgStream."""
        from ....message import AudioBlock, Base64Source

        # Directly create AudioBlock with base64 data to avoid encode/decode
        msg = Msg(
            name=self.name,
            role="assistant",
            content=[
                AudioBlock(
                    type="audio",
                    source=Base64Source(
                        type="base64",
                        media_type=f"audio/pcm;rate={sample_rate}",
                        data=audio_b64,
                    ),
                ),
            ],
            metadata={"is_partial": True, "sample_rate": sample_rate},
        )
        await self.msg_stream.push(msg)

    async def _finalize_response(self) -> None:
        """Finalize the current response and save to memory."""
        if self._response_text:
            # Push final message
            msg = create_msg(
                name=self.name,
                text=self._response_text,
                is_partial=False,
            )
            await self.msg_stream.push(msg)

            # Save to memory
            assistant_msg = Msg(
                name=self.name,
                content=[TextBlock(type="text", text=self._response_text)],
                role="assistant",
            )
            await self.memory.add(assistant_msg)

            logger.info(
                "WebSocketVoiceAgent %s: Response complete (%d chars)",
                self.name,
                len(self._response_text),
            )

        # Reset for next response
        self._response_text = ""

    async def reply(self) -> Msg | None:
        """Wait for and return the next complete response.

        This method waits for events from the model and returns
        when a complete response is received.

        Returns:
            The assistant's response message, or None if stopped.
        """
        if not self._initialized:
            raise RuntimeError(
                "Agent not initialized. Call initialize() first.",
            )

        # Wait for response to complete
        response_complete = asyncio.Event()
        saved_text = ""

        async def wait_for_response() -> None:
            nonlocal saved_text
            async for msg in self.msg_stream.subscribe(f"{self.name}_reply"):
                if self._stop_event.is_set():
                    break
                if msg.role == "user":
                    continue
                is_partial = (
                    msg.metadata.get("is_partial", True)
                    if msg.metadata
                    else True
                )
                if not is_partial:
                    saved_text = msg.get_text_content() or ""
                    response_complete.set()
                    break

        wait_task = asyncio.create_task(wait_for_response())

        try:
            await asyncio.wait_for(response_complete.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("Response timeout")
        finally:
            wait_task.cancel()
            try:
                await wait_task
            except asyncio.CancelledError:
                pass

        if saved_text:
            return Msg(
                name=self.name,
                content=[TextBlock(type="text", text=saved_text)],
                role="assistant",
            )
        return None

    def stop(self) -> None:
        """Stop the agent."""
        self._stop_event.set()

    async def close(self) -> None:
        """Close the agent and release resources."""
        self.stop()

        # Cancel listen task
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        # Close model
        if self._initialized:
            await self.model.close()
            self._initialized = False

        # Close MsgStream
        await self.msg_stream.close()

        logger.info("WebSocketVoiceAgent %s closed", self.name)

    async def get_memory(self) -> list[Msg]:
        """Get the conversation history from memory."""
        return await self.memory.get_memory()
