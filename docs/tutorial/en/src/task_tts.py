# -*- coding: utf-8 -*-
"""
.. _tts:

TTS
====================

AgentScope provides a unified TTS (Text-to-Speech) module that supports multiple TTS providers,
enabling agents to convert text responses into audio output. This tutorial demonstrates how to use
TTS models in AgentScope.

The supported TTS providers include:

.. list-table::
    :header-rows: 1

    * - Provider
      - Class
      - Streaming Input
    * - DashScope Realtime
      - ``DashScopeRealtimeTTSModel``
      - ✅
    * - DashScope
      - ``DashScopeTTSModel``
      - ❌
    * - OpenAI
      - ``OpenAITTSModel``
      - ❌
    * - Gemini
      - ``GeminiTTSModel``
      - ❌

All TTS models inherit from ``TTSModelBase`` and provide a unified interface:

- For **realtime TTS models** (supporting streaming input):

  - ``connect()``: Establish connection to the TTS service

  - ``push(msg)``: Append text chunks incrementally (non-blocking)

  - ``synthesize(msg=None)``: Synthesize speech and block until complete

  - ``close()``: Close the connection and clean up resources

- For **non-realtime TTS models**:

  - ``synthesize(msg)``: Synthesize speech from complete text

The TTS models return ``TTSResponse`` objects containing ``AudioBlock`` instances with base64-encoded audio data.
"""

# %%
# Basic Usage - Realtime TTS Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For realtime TTS models (like ``DashScopeRealtimeTTSModel``), you need to:
#
# 1. Initialize the TTS model with appropriate parameters
# 2. Connect to the TTS service using ``connect()``
# 3. Use ``synthesize()`` to synthesize complete text, or ``push()`` for incremental text
# 4. Close the connection using ``close()``
#
# Let's start with a simple example using DashScope Realtime TTS:

import asyncio
import os
from typing import AsyncGenerator

from agentscope.message import Msg
from agentscope.tts import (
    DashScopeRealtimeTTSModel,
    DashScopeTTSModel,
    OpenAITTSModel,
    GeminiTTSModel,
    TTSResponse,
)


async def example_basic_realtime_tts() -> None:
    """A basic example of using DashScope Realtime TTS."""
    # Initialize the TTS model
    tts_model = DashScopeRealtimeTTSModel(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        model_name="qwen3-tts-flash-realtime",
        voice="Cherry",
        stream=False,  # Set to False for simpler example
    )

    # Connect to the TTS service
    await tts_model.connect()

    # Create a message with text content
    msg = Msg(
        name="assistant",
        content="Hello, this is a test of TTS functionality.",
        role="assistant",
    )

    # Synthesize the text (blocking until complete)
    tts_response = await tts_model.synthesize(msg)

    # The response contains audio blocks
    print(f"TTS Response: {tts_response}")
    print(f"Number of audio blocks: {len(tts_response.content)}")

    # Clean up
    await tts_model.close()


# asyncio.run(example_basic_realtime_tts())

# %%
# Basic Usage - Non-Realtime TTS Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For non-realtime TTS models (like ``DashScopeTTSModel``, ``OpenAITTSModel``, ``GeminiTTSModel``),
# you can directly call ``synthesize()`` without needing to connect first:


async def example_basic_non_realtime_tts() -> None:
    """A basic example of using non-realtime TTS models."""
    # Example with DashScope TTS
    if os.environ.get("DASHSCOPE_API_KEY"):
        tts_model = DashScopeTTSModel(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            model_name="qwen3-tts-flash",
            voice="Cherry",
        )

        msg = Msg(
            name="assistant",
            content="Hello, this is DashScope TTS.",
            role="assistant",
        )

        # Directly synthesize without connecting
        tts_response = await tts_model.synthesize(msg)

        print(f"TTS Response: {tts_response}")
        print(f"Audio blocks: {len(tts_response.content)}")

    # Example with OpenAI TTS
    if os.environ.get("OPENAI_API_KEY"):
        tts_model = OpenAITTSModel(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model_name="gpt-4o-mini-tts",
            voice="alloy",
        )

        msg = Msg(
            name="assistant",
            content="Hello, this is OpenAI TTS.",
            role="assistant",
        )

        tts_response = await tts_model.synthesize(msg)

        print(f"TTS Response: {tts_response}")
        print(f"Audio blocks: {len(tts_response.content)}")

    # Example with Gemini TTS
    if os.environ.get("GEMINI_API_KEY"):
        tts_model = GeminiTTSModel(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            model_name="gemini-2.5-flash-preview-tts",
            voice="Kore",
        )

        msg = Msg(
            name="assistant",
            content="Hello, this is Gemini TTS.",
            role="assistant",
        )

        tts_response = await tts_model.synthesize(msg)

        print(f"TTS Response: {tts_response}")
        print(f"Audio blocks: {len(tts_response.content)}")


# asyncio.run(example_basic_non_realtime_tts())

# %%
# Using TTS with Agents
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The most common use case is integrating TTS with agents. AgentScope's ``ReActAgent``
# supports TTS models through the ``tts_model`` parameter. When a TTS model is provided,
# the agent will automatically synthesize its text responses into audio.
#
# .. note:: The TTS model will be called automatically during agent execution, handling
#           streaming text incrementally for models that support streaming input.


async def example_agent_with_tts() -> None:
    """An example of using TTS with ReActAgent."""
    from agentscope.agent import ReActAgent, UserAgent
    from agentscope.formatter import DashScopeChatFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.model import DashScopeChatModel

    # Create a TTS model
    tts_model = DashScopeRealtimeTTSModel(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        model_name="qwen3-tts-flash-realtime",
        voice="Cherry",
    )

    # Create an agent with TTS enabled
    agent = ReActAgent(
        name="Assistant",
        sys_prompt="You are a helpful assistant.",
        model=DashScopeChatModel(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            model_name="qwen-max",
            stream=True,
        ),
        formatter=DashScopeChatFormatter(),
        memory=InMemoryMemory(),
        tts_model=tts_model,  # Enable TTS
    )

    user = UserAgent("User")

    # The agent will automatically synthesize its responses
    msg = await user("Tell me a short story.")
    response = await agent(msg)

    print(f"Agent response: {response.get_text_content()}")

    # Clean up
    await tts_model.close()


# asyncio.run(example_agent_with_tts())

# %%
# Streaming Input with Push and Synthesize
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For realtime TTS models that support streaming input (like ``DashScopeRealtimeTTSModel``),
# you can use ``push()`` to incrementally send text chunks as they arrive, and then
# call ``synthesize()`` to get the final audio output.
#
# - ``push(msg)``: Non-blocking method that appends text and returns any available audio
# - ``synthesize(msg=None)``: Blocking method that waits for all audio to be synthesized
#
# .. note:: The ``push()`` method uses the message ID (``msg.id``) to track streaming
#           input requests. All chunks for the same message must have the same ID.


async def example_streaming_push_synthesize() -> None:
    """An example of using push() and synthesize() for streaming input."""
    tts_model = DashScopeRealtimeTTSModel(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        model_name="qwen3-tts-flash-realtime",
        voice="Cherry",
        stream=False,  # Set to False for simpler example
    )

    await tts_model.connect()

    # Simulate streaming text generation
    text_chunks = [
        "Hello, ",
        "this is ",
        "a streaming ",
        "TTS example.",
    ]

    # Create a message with a consistent ID for all chunks
    msg_id = "streaming_msg_001"
    accumulated_text = ""

    for i, chunk in enumerate(text_chunks):
        # Accumulate text incrementally
        accumulated_text += chunk

        # Create a message with accumulated text and same ID
        msg = Msg(
            name="assistant",
            content=accumulated_text,
            role="assistant",
        )
        msg.id = msg_id  # Important: same ID for all chunks

        # Push the incremental text (non-blocking)
        tts_response = await tts_model.push(msg)
        if tts_response.content:
            print(
                f"Chunk {i+1}: Received {len(tts_response.content)} audio blocks",
            )

    # Finalize synthesis to get all remaining audio
    final_msg = Msg(
        name="assistant",
        content=accumulated_text,
        role="assistant",
    )
    final_msg.id = msg_id

    final_response = await tts_model.synthesize(final_msg)
    # Handle both TTSResponse and AsyncGenerator cases
    if isinstance(final_response, AsyncGenerator):
        async for chunk in final_response:
            if chunk.content:
                print(
                    f"Final synthesis chunk: {len(chunk.content)} audio blocks",
                )
    else:
        print(f"Final synthesis: {len(final_response.content)} audio blocks")

    await tts_model.close()


# asyncio.run(example_streaming_push_synthesize())

# %%
# Streaming Output Mode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TTS models support streaming output mode, where audio is returned as an async generator
# of ``TTSResponse`` objects. This is useful for real-time audio playback.
#
# Set ``stream=True`` when initializing the TTS model to enable streaming output:


async def example_streaming_output() -> None:
    """An example of using streaming output mode."""
    tts_model = DashScopeRealtimeTTSModel(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        model_name="qwen3-tts-flash-realtime",
        voice="Cherry",
        stream=True,  # Enable streaming output
    )

    await tts_model.connect()

    msg = Msg(
        name="assistant",
        content="This is a streaming output example.",
        role="assistant",
    )

    # Synthesize returns an async generator when stream=True
    response_generator = await tts_model.synthesize(msg)

    if isinstance(response_generator, AsyncGenerator):
        # Streaming mode - iterate over audio chunks
        async for chunk in response_generator:
            if chunk.content:
                print(
                    f"Received audio chunk: {len(chunk.content)} blocks, is_last={chunk.is_last}",
                )
                # Process audio chunk here (e.g., play audio)
    else:
        # Non-streaming mode
        print(f"Received {len(response_generator.content)} audio blocks")

    await tts_model.close()


# asyncio.run(example_streaming_output())

# %%
# Context Manager Usage
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TTS models support Python's async context manager protocol, which automatically handles
# connection and cleanup. This is especially useful for realtime TTS models:


async def example_context_manager() -> None:
    """An example of using TTS models as context managers."""
    # For realtime TTS models, the context manager automatically calls connect() and close()
    async with DashScopeRealtimeTTSModel(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        model_name="qwen3-tts-flash-realtime",
        voice="Cherry",
        stream=False,  # Set to False for simpler example
    ) as tts_model:
        msg = Msg(
            name="assistant",
            content="Using context manager for TTS.",
            role="assistant",
        )
        tts_response = await tts_model.synthesize(msg)
        print(f"TTS Response: {tts_response}")

    # Connection is automatically closed when exiting the context


# asyncio.run(example_context_manager())

# %%
# Configuration Options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Different TTS models support various configuration options:
#
# **DashScope Realtime TTS:**
#
# - ``voice``: Voice selection (e.g., "Cherry", "Serena", "Ethan", "Chelsie")
# - ``mode``: TTS mode ("server_commit" or "commit")
# - ``cold_start_length``: Minimum text length (characters) before sending the first request
# - ``cold_start_words``: Minimum word count before sending the first request
#
# **DashScope TTS:**
#
# - ``voice``: Voice selection
# - ``language_type``: Language type (e.g., "Auto", "Chinese", "English")
#
# **OpenAI TTS:**
#
# - ``voice``: Voice selection (e.g., "alloy", "ash", "ballad", "coral")
# - ``model_name``: Model selection ("gpt-4o-mini-tts", "tts-1", "tts-1-hd")
#
# **Gemini TTS:**
#
# - ``voice``: Voice selection (e.g., "Zephyr", "Kore", "Orus", "Autonoe")
# - ``model_name``: Model selection (e.g., "gemini-2.5-flash-preview-tts")


async def example_configuration() -> None:
    """An example showing different configuration options."""
    # DashScope Realtime TTS with custom configuration
    tts_model = DashScopeRealtimeTTSModel(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        model_name="qwen3-tts-flash-realtime",
        voice="Serena",  # Different voice
        mode="server_commit",  # Server manages text segmentation
        cold_start_length=10,  # Wait for 10 characters before sending
        cold_start_words=3,  # Or wait for 3 words
        stream=False,  # Set to False for simpler example
    )

    await tts_model.connect()

    msg = Msg(
        name="assistant",
        content="Custom configuration example.",
        role="assistant",
    )
    tts_response = await tts_model.synthesize(msg)
    print(f"TTS Response with custom config: {tts_response}")

    await tts_model.close()


# asyncio.run(example_configuration())

# %%
# Handling TTS Responses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TTS models return ``TTSResponse`` objects that contain ``AudioBlock`` instances.
# Each ``AudioBlock`` contains base64-encoded audio data that can be decoded and played:


async def example_handling_response() -> None:
    """An example of handling TTS responses and audio data."""
    import base64

    tts_model = DashScopeRealtimeTTSModel(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        model_name="qwen3-tts-flash-realtime",
        voice="Cherry",
        stream=False,  # Set to False for simpler example
    )

    await tts_model.connect()

    msg = Msg(
        name="assistant",
        content="This example shows how to handle TTS responses.",
        role="assistant",
    )

    tts_response = await tts_model.synthesize(msg)

    # Access audio blocks
    for i, audio_block in enumerate(tts_response.content):
        print(f"Audio block {i}:")
        print(f"  Type: {audio_block.type}")
        print(f"  Source type: {audio_block.source.type}")
        print(f"  Media type: {audio_block.source.media_type}")
        print(
            f"  Data length: {len(audio_block.source.data)} characters (base64)",
        )

        # Decode base64 audio data if needed
        # audio_bytes = base64.b64decode(audio_block.source.data)
        # # Now you can save or play the audio

    # Access response metadata
    print(f"Response ID: {tts_response.id}")
    print(f"Created at: {tts_response.created_at}")
    print(f"Is last: {tts_response.is_last}")

    await tts_model.close()


# asyncio.run(example_handling_response())

# %%
# Further Reading
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# - :ref:`agent` - Learn more about agents in AgentScope
# - :ref:`message` - Understand message format in AgentScope
# - API Reference: :class:`agentscope.tts.TTSModelBase`
#
