# -*- coding: utf-8 -*-
# pylint: disable=too-many-statements, too-many-branches,
# pylint: disable=too-many-nested-blocks
"""Voice Agent implementation based on MsgStream."""

import asyncio
import base64
import io
import json

import numpy as np
import shortuuid

from agentscope.message import (
    Msg,
    TextBlock,
    AudioBlock,
    Base64Source,
    ToolUseBlock,
    ToolResultBlock,
    ImageBlock,
    VideoBlock,
)
from agentscope._logging import logger
from agentscope.module import StateModule

from .._utils._msg_stream import (
    MsgStream,
    create_msg,
    get_audio_from_msg,
)
from ..model._voice_model_base import RealtimeVoiceModelBase


class RealtimeVoiceAgent(StateModule):
    """Voice Agent implementation based on MsgStream.

    Uses MsgStream for message passing, supports streaming Msg push.
    Audio playback is implemented through AgentBase.print()'s speech parameter.
    """

    id: str
    """The agent's unique identifier, generated using shortuuid."""

    def __init__(
        self,
        name: str,
        model: RealtimeVoiceModelBase,
        sys_prompt: str = "You are a helpful assistant.",
        msg_stream: MsgStream | None = None,
    ) -> None:
        """Initialize the VoiceAgent.

        Args:
            name (`str`):
                The name of the agent.
            model (`RealtimeVoiceModelBase`):
                The real-time voice model instance.
            sys_prompt (`str`, defaults to `"You are a helpful assistant."`):
                The system prompt for the model.
            msg_stream (`MsgStream | None`, defaults to `None`):
                The message stream for communication. Can be set later via
                VoiceMsgHub.
        """
        super().__init__()
        self.id = shortuuid.uuid()

        self.name = name
        self._msg_stream = msg_stream  # Can be set later via VoiceMsgHub
        self._model = model
        self._model.sys_prompt = sys_prompt

        self._initialized = False
        self._response_text = ""
        self._response_audio = bytearray()
        self._stop_event = asyncio.Event()
        self._listening_event = asyncio.Event()
        self._interrupt_requested = False  # Thread-safe interruption flag

        # Current playing message ID (for interruption)
        self._current_msg_id: str | None = None

        self._stream_prefix = {}

        self._listen_task = None
        self._streaming_msg = None

    def set_msg_stream(self, msg_stream: MsgStream) -> None:
        """Set the message stream (called by VoiceMsgHub).

        Args:
            msg_stream (`MsgStream`):
                The message stream instance.

        Raises:
            `RuntimeError`:
                If the agent is already initialized.
        """
        if self._initialized:
            raise RuntimeError(
                f"Cannot set msg_stream after VoiceAgent '{self.name}' is "
                f"initialized",
            )
        self._msg_stream = msg_stream

    def _on_audio_delta(self, audio_bytes: bytes) -> None:
        """Handle audio delta callback from model - accumulate for playback.

        Args:
            audio_bytes (`bytes`):
                The incremental audio data.
        """
        self._response_audio.extend(audio_bytes)

    def _on_speech_started(self) -> None:
        """Handle speech started callback from model - stop audio playback.

        This is called by the model when it detects user speech (VAD).
        Uses thread-safe flag to signal interruption to the audio streaming
        task.
        """
        logger.info(
            "%s: 🛑 User speech detected, requesting interruption",
            self.name,
        )

        # Thread-safe: set flag for _stream_audio_fragments to check
        self._interrupt_requested = True

        # Also try to stop the player directly (for immediate effect)
        if hasattr(self, "_streaming_msg") and self._streaming_msg:
            streaming_msg_id = self._streaming_msg.id
            if (
                streaming_msg_id in self._stream_prefix
                and "audio_player" in self._stream_prefix[streaming_msg_id]
            ):
                player = self._stream_prefix[streaming_msg_id]["audio_player"]
                try:
                    player.stop()
                    logger.info("%s: ✅ Audio playback stopped", self.name)
                except Exception as e:
                    logger.debug(
                        "%s: Player stop error (may already be stopped): %s",
                        self.name,
                        e,
                    )

    def _on_response_done(self) -> None:
        """Handle response done callback from model."""

    async def initialize(self) -> None:
        """Initialize the agent and model.

        Raises:
            `RuntimeError`:
                If msg_stream is not set.
        """
        if not self._initialized:
            if self._msg_stream is None:
                raise RuntimeError(
                    f"VoiceAgent '{self.name}' requires msg_stream. "
                    "Either pass it to constructor or use VoiceMsgHub.",
                )
            self._model.set_audio_callbacks(
                on_audio_delta=self._on_audio_delta,
                on_speech_started=self._on_speech_started,
                on_response_done=self._on_response_done,
            )
            await self._model.initialize()
            # Start background listening task
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._initialized = True
            logger.info("VoiceAgent %s initialized", self.name)

    async def _listen_loop(self) -> None:
        """Background task that continuously listens and responds to messages.

        This loop keeps the agent responsive to incoming messages without
        requiring explicit reply() calls.
        """
        logger.info("%s: Starting listen loop", self.name)
        try:
            while not self._stop_event.is_set():
                try:
                    await self.reply()
                except Exception as e:
                    logger.error("%s: Error in listen loop: %s", self.name, e)
                    await asyncio.sleep(0.1)
        finally:
            logger.info("%s: Listen loop stopped", self.name)

    async def reply(self) -> Msg:
        """Listen to messages from other participants in the queue and
        generate a reply.

        Returns:
            `Msg`:
                The response message containing the agent's reply.
        """

        self._response_text = ""
        self._response_audio = bytearray()

        # Create a streaming message with fixed ID for this response
        self._streaming_msg = Msg(
            name=self.name,
            content=[TextBlock(type="text", text="")],
            role="assistant",
        )

        self._model.callback.reset()
        logger.info("%s: Starting realtime forward...", self.name)

        queue: asyncio.Queue[Msg | None] = asyncio.Queue(maxsize=1000)

        # Only receive user messages to avoid queue being filled with
        # agent's own partial messages
        await self._msg_stream.register_queue(
            self.id,
            queue,
            msg_filter=lambda msg: msg.name == "user",
        )

        response_started = False
        got_audio = False
        streaming_tasks = []
        audio_chunks_during_response = 0

        try:
            while (
                not self._stop_event.is_set()
                and not self._msg_stream.is_closed
            ):
                if self._model.callback.is_responding and not response_started:
                    response_started = True
                    audio_chunks_during_response = 0  # Reset counter
                    logger.info("%s: Model is responding...", self.name)

                    # Start streaming tasks immediately when model starts
                    # responding
                    text_task = asyncio.create_task(
                        self._stream_text_fragments(),
                    )
                    audio_task = asyncio.create_task(
                        self._stream_audio_fragments(),
                    )
                    streaming_tasks = [text_task, audio_task]

                if (
                    response_started
                    and self._model.callback.complete_event.is_set()
                ):
                    # Don't break immediately - continue processing user audio
                    # to support interruption during audio playback
                    if streaming_tasks:
                        # Check if streaming tasks are done
                        if all(task.done() for task in streaming_tasks):
                            logger.info(
                                "%s: Model response complete (received %d "
                                "audio chunks during response)",
                                self.name,
                                audio_chunks_during_response,
                            )
                            break
                        # Else continue loop to process more user audio
                        # while streaming
                    else:
                        logger.info(
                            "%s: Model response complete "
                            "(received %d audio chunks during response)",
                            self.name,
                            audio_chunks_during_response,
                        )
                        break

                # Batch process all available messages in queue
                messages_processed = 0
                try:
                    # First, wait for at least one message (with timeout)
                    received_msg = await asyncio.wait_for(
                        queue.get(),
                        timeout=0.05,  # Shorter timeout for faster response
                    )

                    # Process all available messages
                    messages_to_process = [received_msg]
                    while True:
                        try:
                            msg = queue.get_nowait()
                            messages_to_process.append(msg)
                        except asyncio.QueueEmpty:
                            break

                    for msg in messages_to_process:
                        if msg is None:
                            return  # Exit the entire method

                        messages_processed += 1

                        # Check for response end (is_partial=False)
                        # This triggers creating a new response
                        is_partial = msg.metadata.get("is_partial", True)
                        if not is_partial and got_audio:
                            if self._model.conversation:
                                self._model.conversation.commit()
                                from dashscope.audio.qwen_omni import (
                                    MultiModality,
                                )

                                self._model.conversation.create_response(
                                    output_modalities=[
                                        MultiModality.AUDIO,
                                        MultiModality.TEXT,
                                    ],
                                )
                            continue

                        audio_data = get_audio_from_msg(msg)
                        if audio_data:
                            got_audio = True
                            sample_rate = (
                                msg.metadata.get("sample_rate", 16000)
                                if msg.metadata
                                else 16000
                            )
                            try:
                                self._model.append_audio(
                                    audio_data,
                                    sample_rate=sample_rate,
                                )
                                if response_started:
                                    audio_chunks_during_response += 1
                            except Exception as e:
                                logger.warning(
                                    "%s: Failed to append audio: %s",
                                    self.name,
                                    e,
                                )
                                if not self._model.is_connected:
                                    await self._model.reconnect()
                                    got_audio = False

                except asyncio.TimeoutError:
                    continue
        finally:
            await self._msg_stream.unregister_queue(self.id)

        # Wait for streaming tasks to complete
        if streaming_tasks:
            await asyncio.gather(*streaming_tasks, return_exceptions=True)

        if self._response_text or self._response_audio:
            # Update the streaming message with final content
            self._streaming_msg.content = [
                TextBlock(type="text", text=self._response_text),
            ]

            # Text and audio have been streamed during
            # _collect_and_stream_response
            # This final print is mainly for resource cleanup and newline
            self._current_msg_id = self._streaming_msg.id

            # Final print with last=True for cleanup
            # Using the same streaming_msg.id ensures text won't be reprinted
            # due to _stream_prefix tracking
            # No speech parameter needed as audio was played in real-time
            await self.print(self._streaming_msg, last=True)

            self._current_msg_id = None
            return self._streaming_msg
        else:
            logger.info("%s: No response", self.name)
            return Msg(name=self.name, content="", role="assistant")

    async def _collect_and_stream_response(
        self,
        timeout: float = 30.0,
    ) -> None:
        """Collect model response and stream push to MsgStream.

        Streams both text and audio fragments as they arrive for real-time
        display and playback.

        Args:
            timeout (`float`, defaults to `30.0`):
                Maximum time to wait for response in seconds.
        """
        # Create concurrent tasks to stream text and audio fragments
        text_task = asyncio.create_task(self._stream_text_fragments())
        audio_task = asyncio.create_task(self._stream_audio_fragments())

        # Wait for response completion
        start_time = asyncio.get_event_loop().time()
        while not self._model.callback.complete_event.is_set():
            await asyncio.sleep(0.1)
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning(
                    "%s: Response timeout after %0.1fs",
                    self.name,
                    timeout,
                )
                break

        # Wait for streaming tasks to complete
        await text_task
        await audio_task

        # Push final complete message to MsgStream
        if self._response_audio:
            await self._msg_stream.push(
                create_msg(
                    name=self.name,
                    text=self._response_text,
                    audio_data=bytes(self._response_audio),
                    sample_rate=24000,
                    is_partial=False,
                ),
            )

    async def _stream_text_fragments(self) -> None:
        """Stream text fragments as they arrive from the model.

        Pushes incremental text to MsgStream and prints in real-time.
        Uses the fixed streaming_msg_id created in reply() for consistent
        prefix tracking across fragments.
        """
        streaming_msg_id = self._streaming_msg.id
        logger.info(
            "%s: Starting text streaming (msg_id: %s)",
            self.name,
            streaming_msg_id,
        )

        try:
            fragment_count = 0
            async for text_fragment in self._model.iter_text_fragments():
                if not text_fragment:
                    continue

                fragment_count += 1
                # Accumulate the text
                self._response_text += text_fragment

                if fragment_count <= 3 or fragment_count % 10 == 0:
                    logger.info(
                        "%s: Text fragment #%d: %r (total length: %d)",
                        self.name,
                        fragment_count,
                        text_fragment[:20] + "..."
                        if len(text_fragment) > 20
                        else text_fragment,
                        len(self._response_text),
                    )

                # Push incremental text to MsgStream
                await self._msg_stream.push(
                    create_msg(
                        name=self.name,
                        text=self._response_text,
                        is_partial=True,
                    ),
                )

                # Print streaming text using the same msg_id for prefix
                # tracking
                # Directly call _print_text_block to maintain streaming state
                thinking_and_text_to_print = []
                self._print_text_block(
                    streaming_msg_id,
                    name_prefix=self.name,
                    text_content=self._response_text,
                    thinking_and_text_to_print=thinking_and_text_to_print,
                )

            logger.info(
                "%s: Text streaming ended (%d fragments)",
                self.name,
                fragment_count,
            )
        except Exception as e:
            logger.error("%s: Error streaming text: %s", self.name, e)

    async def _stream_audio_fragments(self) -> None:
        """Stream audio fragments as they arrive from the model.

        Plays audio in real-time using sounddevice.OutputStream.
        Uses non-blocking write to allow interruption.
        """
        import sounddevice as sd

        streaming_msg_id = self._streaming_msg.id
        loop = asyncio.get_event_loop()

        # Initialize audio player
        player = None
        fragment_count = 0
        self._interrupt_requested = False  # Reset interrupt flag

        try:
            player = sd.OutputStream(
                samplerate=24000,
                channels=1,
                dtype=np.float32,
                blocksize=1024,
                latency="low",
            )
            # Store player reference BEFORE starting for interruption handling
            if streaming_msg_id not in self._stream_prefix:
                self._stream_prefix[streaming_msg_id] = {}
            self._stream_prefix[streaming_msg_id]["audio_player"] = player

            player.start()
            logger.info(
                "%s: Audio streaming started (msg_id: %s)",
                self.name,
                streaming_msg_id,
            )

            async for audio_fragment in self._model.iter_audio_fragments():
                # Check for interruption
                if self._interrupt_requested:
                    logger.info("%s: 🛑 Audio interrupted by user", self.name)
                    break

                if not audio_fragment:
                    continue

                fragment_count += 1

                # Accumulate audio data
                self._response_audio.extend(audio_fragment)

                # Convert and play the new audio fragment (non-blocking)
                try:
                    audio_np = np.frombuffer(audio_fragment, dtype=np.int16)
                    audio_float = audio_np.astype(np.float32) / 32768.0

                    if fragment_count == 1:
                        logger.info(
                            "%s: ▶️ Audio playback started (can interrupt "
                            "now)",
                            self.name,
                        )

                    # Non-blocking write using executor
                    await loop.run_in_executor(None, player.write, audio_float)

                except Exception as e:
                    logger.warning(
                        "%s: Error playing audio fragment: %s",
                        self.name,
                        e,
                    )
                    break

                # Push incremental audio to MsgStream
                await self._msg_stream.push(
                    create_msg(
                        name=self.name,
                        audio_data=audio_fragment,
                        sample_rate=24000,
                        is_partial=True,
                    ),
                )

        except Exception as e:
            logger.error("%s: Error streaming audio: %s", self.name, e)
        finally:
            # Close the player
            if player:
                try:
                    player.stop()
                    player.close()
                    logger.info(
                        "%s: Real-time audio streaming ended "
                        "(%d fragments, %d bytes total)",
                        self.name,
                        fragment_count,
                        len(self._response_audio),
                    )
                except Exception:
                    pass
            # Remove from _stream_prefix
            if (
                streaming_msg_id in self._stream_prefix
                and "audio_player" in self._stream_prefix[streaming_msg_id]
            ):
                del self._stream_prefix[streaming_msg_id]["audio_player"]

    def _build_speech(self) -> AudioBlock | None:
        """Build AudioBlock for print playback.

        Returns:
            `AudioBlock | None`:
                The audio block for speech playback, or None if no audio.
        """
        if not self._response_audio:
            return None
        return AudioBlock(
            type="audio",
            source=Base64Source(
                type="base64",
                media_type="audio/pcm;rate=24000",
                data=base64.b64encode(bytes(self._response_audio)).decode(
                    "ascii",
                ),
            ),
        )

    def stop(self) -> None:
        """Stop the agent's operation."""
        self._stop_event.set()

    async def close(self) -> None:
        """Close the agent and release resources."""
        self.stop()
        if self._initialized:
            await self._model.close()
            self._initialized = False
            logger.info("VoiceAgent %s closed", self.name)

    @property
    def is_initialized(self) -> bool:
        """Check if the agent is initialized.

        Returns:
            `bool`:
                True if initialized, False otherwise.
        """
        return self._initialized

    async def print(
        self,
        msg: Msg,
        last: bool = True,
        speech: AudioBlock | list[AudioBlock] | None = None,
    ) -> None:
        """The function to display the message.

        Args:
            msg (`Msg`):
                The message object to be printed.
            last (`bool`, defaults to `True`):
                Whether this is the last one in streaming messages. For
                non-streaming message, this should always be `True`.
            speech (`AudioBlock | list[AudioBlock] | None`, optional):
                The audio content block(s) to be played along with the
                message.
        """

        # The accumulated textual content to print, including the text blocks
        # and the thinking blocks
        thinking_and_text_to_print = []

        for block in msg.get_content_blocks():
            if block["type"] == "text":
                self._print_text_block(
                    msg.id,
                    name_prefix=msg.name,
                    text_content=block["text"],
                    thinking_and_text_to_print=thinking_and_text_to_print,
                )

            elif block["type"] == "thinking":
                self._print_text_block(
                    msg.id,
                    name_prefix=f"{msg.name}(thinking)",
                    text_content=block["thinking"],
                    thinking_and_text_to_print=thinking_and_text_to_print,
                )

            elif last:
                self._print_last_block(block, msg)

        # Play audio block if exists
        if isinstance(speech, list):
            for audio_block in speech:
                self._process_audio_block(msg.id, audio_block)
        elif isinstance(speech, dict):
            self._process_audio_block(msg.id, speech)

        # Clean up resources if this is the last message in streaming
        if last and msg.id in self._stream_prefix:
            # Note: audio_player is already closed in
            # _stream_audio_fragments finally block
            # Only clean up _process_audio_block style audio if it exists
            if "audio" in self._stream_prefix[msg.id]:
                audio_info = self._stream_prefix[msg.id]["audio"]
                # audio_info is (player, base64_str) from _process_audio_block
                if isinstance(audio_info, tuple) and len(audio_info) >= 1:
                    player = audio_info[0]
                    try:
                        player.close()
                    except Exception:
                        pass
            stream_prefix = self._stream_prefix.pop(msg.id)
            if "text" in stream_prefix and not stream_prefix["text"].endswith(
                "\n",
            ):
                print()

    def _process_audio_block(
        self,
        msg_id: str,
        audio_block: AudioBlock,
    ) -> None:
        """Process audio block content.

        Args:
            msg_id (`str`):
                The unique identifier of the message
            audio_block (`AudioBlock`):
                The audio content block
        """
        if "source" not in audio_block:
            raise ValueError(
                "The audio block must contain the 'source' field.",
            )

        if audio_block["source"]["type"] == "url":
            import urllib.request
            import wave
            import sounddevice as sd

            url = audio_block["source"]["url"]
            try:
                with urllib.request.urlopen(url) as response:
                    audio_data = response.read()

                with wave.open(io.BytesIO(audio_data), "rb") as wf:
                    samplerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    audio_frames = wf.readframes(n_frames)

                    # Convert byte data to numpy array
                    audio_np = np.frombuffer(audio_frames, dtype=np.int16)

                    # Play audio
                    sd.play(audio_np, samplerate)
                    sd.wait()

            except Exception as e:
                logger.error(
                    "Failed to play audio from url %s: %s",
                    url,
                    str(e),
                )

        elif audio_block["source"]["type"] == "base64":
            data = audio_block["source"]["data"]

            if msg_id not in self._stream_prefix:
                self._stream_prefix[msg_id] = {}

            audio_prefix = self._stream_prefix[msg_id].get("audio", None)

            import sounddevice as sd

            # The player and the prefix data is cached for streaming audio
            if audio_prefix:
                player, audio_prefix_data = audio_prefix
            else:
                player = sd.OutputStream(
                    samplerate=24000,
                    channels=1,
                    dtype=np.float32,
                    blocksize=1024,
                    latency="low",
                )
                player.start()
                audio_prefix_data = ""

            # play the audio data
            new_audio_data = data[len(audio_prefix_data) :]
            if new_audio_data:
                audio_bytes = base64.b64decode(new_audio_data)
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_np.astype(np.float32) / 32768.0

                # Write to the audio output stream
                player.write(audio_float)

            # save the player and the prefix data
            self._stream_prefix[msg_id]["audio"] = (
                player,
                data,
            )

        else:
            raise ValueError(
                "Unsupported audio source type: "
                f"{audio_block['source']['type']}",
            )

    def _print_text_block(
        self,
        msg_id: str,
        name_prefix: str,
        text_content: str,
        thinking_and_text_to_print: list[str],
    ) -> None:
        """Print the text block and thinking block content.

        Args:
            msg_id (`str`):
                The unique identifier of the message
            name_prefix (`str`):
                The prefix for the message, e.g. "{name}: " for text block and
                "{name}(thinking): " for thinking block.
            text_content (`str`):
                The textual content to be printed.
            thinking_and_text_to_print (`list[str]`):
                A list of textual content to be printed together. Here we
                gather the text and thinking blocks to print them together.
        """
        thinking_and_text_to_print.append(
            f"{name_prefix}: {text_content}",
        )
        # The accumulated text and thinking blocks to print
        to_print = "\n".join(thinking_and_text_to_print)

        # The text prefix that has been printed
        if msg_id not in self._stream_prefix:
            self._stream_prefix[msg_id] = {}

        text_prefix = self._stream_prefix[msg_id].get("text", "")

        # Only print when there is new text content
        if len(to_print) > len(text_prefix):
            print(to_print[len(text_prefix) :], end="")

            # Save the printed text prefix
            self._stream_prefix[msg_id]["text"] = to_print

    def _print_last_block(
        self,
        block: ToolUseBlock
        | ToolResultBlock
        | ImageBlock
        | VideoBlock
        | AudioBlock,
        msg: Msg,
    ) -> None:
        """Process and print the last content block, and the block type
        is not text, or thinking.

        Args:
            block (`ToolUseBlock | ToolResultBlock | ImageBlock | VideoBlock \
            | AudioBlock`):
                The content block to be printed
            msg (`Msg`):
                The message object
        """
        # TODO: We should consider how to handle the multimodal blocks in the
        #  terminal, since the base64 data may be too long to display.
        if block.get("type") in ["image", "video", "audio"]:
            return

        text_prefix = self._stream_prefix.get(msg.id, {}).get("text", "")

        if text_prefix:
            # Add a newline to separate from previous text content
            print_newline = "" if text_prefix.endswith("\n") else "\n"
            print(
                f"{print_newline}"
                f"{json.dumps(block, indent=4, ensure_ascii=False)}",
            )
        else:
            print(
                f"{msg.name}:"
                f" {json.dumps(block, indent=4, ensure_ascii=False)}",
            )
