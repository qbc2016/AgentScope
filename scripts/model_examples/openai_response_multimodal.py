# -*- coding: utf-8 -*-
"""Example of OpenAI Responses API model multimodal (vision) calls using
DataBlock."""
import asyncio
import base64
import os
from pathlib import Path

from _utils import stream_and_collect
from agentscope.message import (
    Msg,
    TextBlock,
    DataBlock,
    URLSource,
    Base64Source,
)
from agentscope.model import OpenAIResponseModel
from agentscope.credential import OpenAICredential

# A publicly accessible test image (a simple cat photo)
TEST_IMAGE_URL = (
    "https://help-static-aliyun-doc.aliyuncs.com/file-manage"
    "-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
)

# A publicly accessible test audio
TEST_AUDIO_URL = (
    "https://help-static-aliyun-doc.aliyuncs.com/file-manage"
    "-files/zh-CN/20250211/tixcef/cherry.wav"
)


async def example_image_url() -> None:
    """Call gpt-4.1 (Responses API) with an image URL and ask what is in
    the image."""
    model = OpenAIResponseModel(
        credential=OpenAICredential(
            api_key=os.environ["OPENAI_API_KEY"],
        ),
        model="gpt-4.1",
        stream=True,
        context_size=1_047_576,
    )

    image_block = DataBlock(
        source=URLSource(
            url=TEST_IMAGE_URL,
            media_type="image/jpeg",
        ),
    )

    msgs = [
        Msg(
            name="user",
            content=[
                TextBlock(
                    text="What animal is in this image? Describe it briefly.",
                ),
                image_block,
            ],
            role="user",
        ),
    ]

    print("=== Multimodal Call (Image URL) ===")
    await stream_and_collect(await model(msgs))


def _build_model() -> OpenAIResponseModel:
    return OpenAIResponseModel(
        credential=OpenAICredential(api_key=os.environ["OPENAI_API_KEY"]),
        model="gpt-4.1",
        stream=True,
        context_size=1_047_576,
    )


async def example_image_local_path() -> None:
    """Call gpt-4.1 (Responses API) with a local image using a ``file://`` URL.

    The formatter reads the file from disk and converts it to a base64 data
     URI.
    """
    model = _build_model()

    abs_path = str(Path(__file__).parent / "test.jpeg")
    msgs = [
        Msg(
            name="user",
            content=[
                TextBlock(
                    text="What is happening in this image? Describe it "
                    "briefly.",
                ),
                DataBlock(
                    source=URLSource(
                        url=f"file://{abs_path}",
                        media_type="image/jpeg",
                    ),
                ),
            ],
            role="user",
        ),
    ]

    print("=== Local Path Call (file://) ===")
    await stream_and_collect(await model(msgs))


async def example_image_base64() -> None:
    """Call gpt-4.1 (Responses API) with a local image using explicit base64.

    Use ``Base64Source`` when you already have the binary data in memory or
    want full control over the encoding step.
    """
    model = _build_model()

    with open(Path(__file__).parent / "test.jpeg", "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    msgs = [
        Msg(
            name="user",
            content=[
                TextBlock(
                    text="What is happening in this image? Describe it "
                    "briefly.",
                ),
                DataBlock(
                    source=Base64Source(data=data, media_type="image/jpeg"),
                ),
            ],
            role="user",
        ),
    ]

    print("=== Explicit Base64 Call ===")
    await stream_and_collect(await model(msgs))


async def example_audio_input() -> None:
    """Call gpt-4o-audio-preview (Responses API) with an audio URL.

    Audio understanding requires an audio-capable model such as
    ``gpt-audio-mini``.  The Responses API does not have a dedicated
    audio input type; audio is sent as ``input_file`` (the formatter handles
    the conversion from the Chat-Completions-style ``input_audio`` format
    automatically).

    Note: audio *output* is not yet supported by the Responses API.
    See https://developers.openai.com/api/docs/guides/migrate-to-responses
    """
    model = OpenAIResponseModel(
        credential=OpenAICredential(
            api_key=os.environ["OPENAI_API_KEY"],
        ),
        model="gpt-audio-mini",
        stream=True,
    )

    audio_block = DataBlock(
        source=URLSource(
            url=TEST_AUDIO_URL,
            media_type="audio/wav",
        ),
    )

    msgs = [
        Msg(
            name="user",
            content=[
                TextBlock(text="What is being said in this audio clip?"),
                audio_block,
            ],
            role="user",
        ),
    ]

    print("=== Multimodal Call (Audio Input) ===")
    await stream_and_collect(await model(msgs))


if __name__ == "__main__":
    asyncio.run(example_image_url())
    asyncio.run(example_image_local_path())
    asyncio.run(example_image_base64())
    asyncio.run(example_audio_input())
