# -*- coding: utf-8 -*-
"""Audio model"""
# https://help.aliyun.com/zh/model-studio/realtime
import sys
import threading
from dashscope.audio.qwen_omni import OmniRealtimeCallback

from agentscope.message import TextBlock, AudioBlock, Base64Source
from agentscope._logging import logger


class OmniCallback(OmniRealtimeCallback):
    """Callback handler for Omni realtime conversation"""

    def __init__(self) -> None:
        super().__init__()
        self.complete_event: threading.Event = threading.Event()
        self.conversation = None
        self.response_text = ""
        self.response_audio = ""

        self.chunks = []

    def wait_for_complete(self) -> None:
        """Wait for response completion"""
        self.complete_event.wait()
        self.complete_event = threading.Event()

    def on_open(self) -> None:
        """Handle connection open"""
        logger.info("Connection opened, initializing audio...")

    def on_close(self, close_status_code: int, close_msg: str) -> None:
        """Handle connection close"""
        logger.info(f"Connection closed: {close_status_code}, {close_msg}")
        sys.exit(0)

    def on_event(self, response: dict) -> None:
        """Handle various event types"""
        try:
            event_type = response["type"]
            # print(response)
            if event_type == "session.created":
                logger.info(f'Session started: {response["session"]["id"]}')
            elif (
                event_type
                == "conversation.item.input_audio_transcription.completed"
            ):
                logger.info(f'Question: {response["transcript"]}')
            elif event_type == "response.audio_transcript.delta":
                text = response["delta"]
                # print(f"Got LLM response delta: {text}")
                self.response_text += text
                self.chunks.append(
                    ([TextBlock(type="text", text=self.response_text)], False),
                )

            elif event_type == "response.audio.delta":
                audio_data = response["delta"]
                self.response_audio += audio_data
                self.chunks.append(
                    (
                        [
                            TextBlock(
                                type="text",
                                text=self.response_text,
                            ),
                            AudioBlock(
                                type="audio",
                                source=Base64Source(
                                    type="base64",
                                    media_type="audio/wav",
                                    data=self.response_audio,
                                ),
                            ),
                        ],
                        False,
                    ),
                )
            elif event_type == "response.done":
                logger.info("======RESPONSE DONE======")
                logger.info(
                    logger.info(
                        f"[Metric] response: "
                        f"{self.conversation.get_last_response_id()}, "
                        f"first text delay: "
                        f"{self.conversation.get_last_first_text_delay()}, "
                        f"first audio delay: "
                        f"{self.conversation.get_last_first_audio_delay()}",
                    ),
                )

                # print(self.response_audio)
                self.chunks.append(
                    (
                        [
                            TextBlock(
                                type="text",
                                text=self.response_text,
                            ),
                            AudioBlock(
                                type="audio",
                                source=Base64Source(
                                    type="base64",
                                    media_type="audio/wav",
                                    data=self.response_audio,
                                ),
                            ),
                        ],
                        True,
                    ),
                )

                self.complete_event.set()
        except Exception as e:
            logger.info(f"[Error] {e}")
