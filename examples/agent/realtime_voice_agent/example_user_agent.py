# -*- coding: utf-8 -*-
"""User-Agent conversation example using VoiceMsgHub.

Demonstrates how VoiceMsgHub simplifies setting up conversations between
users and agents.
"""

import asyncio
import os
import base64
from aioconsole import ainput  # type: ignore[import-untyped]


from agentscope.agent.realtime_voice_agent import (
    DashScopeRealtimeVoiceModel,
    RealtimeVoiceAgent,
    RealtimeVoiceInput,
    VoiceMsgHub,
)
from agentscope.message import Msg, AudioBlock, Base64Source, URLSource
from agentscope.formatter._openai_formatter import _to_openai_audio_data


async def wait_for_exit_key() -> None:
    """Wait for user to press Ctrl+C to exit."""
    await ainput("\nPress Enter to exit...\n")


def download_audio_to_pcm_base64(url: str) -> str:
    """Download audio from URL and convert to PCM base64 format.

    DashScope API requires PCM format: 16kHz, mono, 16-bit.
    Directly downloading and base64-encoding MP3 won't work because:
    - MP3 is compressed format, not raw PCM
    - DashScope can't decode MP3, it only accepts PCM

    Args:
        url: The URL of the audio file to download.

    Returns:
        Base64-encoded PCM audio data (16kHz, mono, 16-bit).

    Requires:
        pydub package (to convert MP3/WAV to PCM format)
    """
    from pydub import AudioSegment  # type: ignore[import-untyped]
    import io

    # Download audio using existing formatter function
    audio_data = _to_openai_audio_data(URLSource(type="url", url=url))

    # Decode the base64 data (still in MP3/WAV format)
    audio_bytes = base64.b64decode(audio_data["data"])

    # Convert to PCM format (pydub decodes MP3 and extracts raw PCM)
    audio = AudioSegment.from_file(
        io.BytesIO(audio_bytes),
        format=audio_data["format"],
    )
    # Resample to 16kHz, mono, 16-bit (DashScope requirement)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    # audio.raw_data is pure PCM bytes (no header, no compression)
    return base64.b64encode(audio.raw_data).decode("ascii")


async def main() -> None:
    """Run the main user-agent voice conversation demo.

    This function sets up a real-time voice conversation between a user
    and an AI agent using VoiceMsgHub to manage the participants and
    message flow.

    The conversation supports:
    - Automatic voice activity detection (VAD)
    - Speech interruption (user can interrupt agent's response)
    - Continuous dialogue until interrupted

    Raises:
        `KeyboardInterrupt`:
            When user presses Ctrl+C to exit.
        `Exception`:
            Any other errors during conversation.
    """

    # Create model with automatic VAD enabled
    model = DashScopeRealtimeVoiceModel(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        voice="Cherry",
    )

    # Create VoiceInput and Agent (no need to pass msg_stream)
    voice_input = RealtimeVoiceInput()
    # devices = voice_input.list_devices()
    #
    # print("\n可用的输入设备:")
    # for device in devices:
    #     print(f"  [{device['index']}] {device['name']}")
    #
    # device_index = None
    # if len(devices) > 1:
    #     try:
    #         choice = input("\n请输入设备编号（直接回车使用默认设备）: ").strip()
    #         if choice:
    #             device_index = int(choice)
    #             print(
    #                 f"使用设备: {devices[device_index]['name'] if
    #                 device_index < len(devices) else device_index}")
    #     except (ValueError, IndexError):
    #         print("使用默认设备")
    #
    # if device_index is not None:
    #     voice_input = RealtimeVoiceInput(device_index=device_index)

    agent = RealtimeVoiceAgent(
        name="assistant",
        model=model,
        sys_prompt="You are a friendly assistant. Please answer questions "
        "concisely.",
    )

    print("\n" + "=" * 60)
    print("User-Agent Conversation Example Using VoiceMsgHub")
    print("=" * 60 + "\n")

    # Use VoiceMsgHub to manage participants
    async with VoiceMsgHub(participants=[voice_input, agent]) as hub:
        try:
            print("Real-time voice conversation started")
            print(
                "- Speak directly, the model will automatically detect "
                "speech end",
            )
            print(
                "- Supports interruption: Speaking will stop agent's audio "
                "output",
            )
            print("- Press Ctrl+C to exit")
            print()

            # Download and convert audio to PCM base64
            print("Downloading and converting audio...")
            audio_url = (
                "https://dashscope.oss-cn-beijing.aliyuncs.com"
                "/audios/welcome.mp3"
            )
            audio_base64 = download_audio_to_pcm_base64(audio_url)
            print(f"Audio converted: {len(audio_base64)} base64 chars")

            msg = Msg(
                "user",
                [
                    AudioBlock(
                        type="audio",
                        source=Base64Source(
                            type="base64",
                            media_type="audio/pcm;rate=16000",
                            data=audio_base64,
                        ),
                    ),
                ],
                role="user",
                metadata={
                    "sample_rate": 16000,
                    "is_partial": True,
                },
            )

            # Wait for agent to start listening
            print("Waiting for agent to start listening...")
            await asyncio.sleep(0.5)

            print("Pushing audio message to stream...")
            await hub.msg_stream.push(msg)
            print("Message pushed, agent will process automatically...")

            await wait_for_exit_key()

        except KeyboardInterrupt:
            print("\n\nUser interrupted")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()
    print("\nConversation ended")


if __name__ == "__main__":
    asyncio.run(main())
