# -*- coding: utf-8 -*-
"""User-Agent conversation example using VoiceMsgHub.

Demonstrates how VoiceMsgHub simplifies setting up conversations between
users and agents.
"""

import asyncio
import os

from agentscope.agent.realtime_voice_agent import (
    DashScopeRealtimeVoiceModel,
    RealtimeVoiceAgent,
    RealtimeVoiceInput,
    VoiceMsgHub,
)


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


async def main() -> None:
    """Run the main user-agent voice conversation demo.

    This function sets up a real-time voice conversation between a user
    and an AI agent using VoiceMsgHub to manage the participants and
    message flow.

    The conversation supports:
    - Automatic voice activity detection (VAD)
    - Speech interruption (user can interrupt agent's response)
    - Continuous dialogue until interrupted
    """
    # Select audio device
    device_index = select_audio_device()

    # Create model
    model = DashScopeRealtimeVoiceModel(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        voice="Cherry",
    )

    # Create VoiceInput and Agent
    voice_input = RealtimeVoiceInput(device_index=device_index)
    agent = RealtimeVoiceAgent(
        name="assistant",
        model=model,
        sys_prompt="You are a friendly assistant. Please answer questions "
        "concisely.",
    )

    # Create hub
    hub = VoiceMsgHub(participants=[voice_input, agent])

    print("\n" + "=" * 60)
    print("User-Agent Conversation Example")
    print("=" * 60)
    print("\nInstructions:")
    print("  - Speak directly, the model will detect when you stop")
    print(
        "  - Interruption supported: speaking will stop the assistant's "
        "audio",
    )
    print("  - Press Ctrl+C to exit")
    print()

    try:
        # Start the hub
        await hub.start()
        print("✅ Voice conversation started\n")

        # Wait until interrupted
        await hub.join()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Stop and cleanup
        await hub.stop()
        print("\nConversation ended")


if __name__ == "__main__":
    asyncio.run(main())
