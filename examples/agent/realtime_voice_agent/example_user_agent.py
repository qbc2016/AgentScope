# -*- coding: utf-8 -*-
"""User-Agent conversation example using WebSocketVoiceAgent.

Demonstrates how to use WebSocketVoiceAgent with local audio input/output
for testing purposes.
"""

import asyncio
import base64
import os

import numpy as np
import sounddevice as sd

from agentscope.agent.realtime_voice_agent import (
    WebSocketVoiceAgent,
    DashScopeWebSocketModel,
    RealtimeVoiceInput,
    MsgStream,
)
from agentscope.memory import InMemoryMemory


def select_audio_device() -> int | None:
    """Let user select an audio input device.

    Returns:
        Selected device index, or None for default device.
    """
    voice_input = RealtimeVoiceInput()
    devices = voice_input.list_devices()

    print("\nAvailable input devices:")
    for device in devices:
        print(f"  [{device['index']}] {device['name']}")

    if len(devices) <= 1:
        return None

    try:
        choice = input(
            "\nEnter device number (press Enter for default): ",
        ).strip()
        if choice:
            device_index = int(choice)
            if device_index < len(devices):
                print(f"Using device: {devices[device_index]['name']}")
                return device_index
    except (ValueError, IndexError):
        pass

    print("Using default device")
    return None


class LocalAudioPlayer:
    """Local audio player for WebSocketVoiceAgent output."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._stream: sd.OutputStream | None = None
        self._playing = False

    def start(self) -> None:
        """Start the audio output stream."""
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
            )
            self._stream.start()
            self._playing = True

    def play(self, audio_data: bytes) -> None:
        """Play audio data."""
        if not self._playing or self._stream is None:
            self.start()

        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        self._stream.write(audio_array)

    def stop(self) -> None:
        """Stop the audio output stream."""
        self._playing = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


async def audio_playback_loop(
    msg_stream: MsgStream,
    player: LocalAudioPlayer,
    stop_event: asyncio.Event,
) -> None:
    """Loop to play audio from MsgStream."""
    try:
        async for msg in msg_stream.subscribe("audio_player"):
            if stop_event.is_set():
                break

            # Only play assistant's audio
            if msg.role != "assistant":
                continue

            for block in msg.get_content_blocks():
                if block.get("type") == "audio":
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        audio_b64 = source.get("data", "")
                        if audio_b64:
                            audio_data = base64.b64decode(audio_b64)
                            player.play(audio_data)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Audio playback error: {e}")


# pylint: disable=too-many-statements
async def main() -> None:
    """Run the main user-agent voice conversation demo.

    This function sets up a real-time voice conversation between a user
    and an AI agent using WebSocketVoiceAgent with local audio I/O.

    The conversation supports:
    - Automatic voice activity detection (VAD)
    - Speech interruption (user can interrupt agent's response)
    - Continuous dialogue until interrupted
    """
    # Select audio device
    device_index = select_audio_device()

    # Get API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY environment variable not set")
        return

    # Create components
    msg_stream = MsgStream()
    player = LocalAudioPlayer(sample_rate=24000)
    stop_event = asyncio.Event()

    # Create WebSocket model
    model = DashScopeWebSocketModel(
        api_key=api_key,
        model_name="qwen3-omni-flash-realtime",
        voice="Cherry",
        instructions="You are a friendly assistant. Please answer questions "
        "concisely.",
        vad_enabled=True,
    )

    # Create WebSocket Agent
    agent = WebSocketVoiceAgent(
        name="assistant",
        model=model,
        sys_prompt="You are a friendly assistant. Please answer questions "
        "concisely.",
        msg_stream=msg_stream,
        memory=InMemoryMemory(),
    )

    # Create VoiceInput with same msg_stream
    # Agent will consume audio from MsgStream and send to model
    voice_input = RealtimeVoiceInput(
        msg_stream=msg_stream,
        device_index=device_index,
    )

    print("\n" + "=" * 60)
    print("User-Agent Conversation Example (WebSocket Mode)")
    print("=" * 60)
    print("\nInstructions:")
    print("  - Speak directly, the model will detect when you stop")
    print(
        "  - Interruption supported: speaking will stop the assistant's "
        "audio",
    )
    print("  - Press Ctrl+C to exit")
    print()

    tasks: list[asyncio.Task] = []

    try:
        # Initialize agent
        await agent.initialize()

        # Start voice input
        await voice_input.start()

        print("✅ Voice conversation started\n")

        # Start audio playback loop
        tasks.append(
            asyncio.create_task(
                audio_playback_loop(msg_stream, player, stop_event),
            ),
        )

        # Wait for agent events (main loop)
        while not stop_event.is_set():
            try:
                response = await asyncio.wait_for(
                    agent.reply(),
                    timeout=60.0,
                )
                if response:
                    text = response.get_text_content()
                    if text:
                        print(f"assistant: {text}")
            except asyncio.TimeoutError:
                continue

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Stop all tasks
        stop_event.set()
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Cleanup
        await voice_input.stop()
        player.stop()
        await agent.close()
        await msg_stream.close()
        print("\nConversation ended")


if __name__ == "__main__":
    asyncio.run(main())
