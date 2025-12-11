# -*- coding: utf-8 -*-
"""The studio related hook functions in agentscope."""
from typing import Any

import requests
import shortuuid

from ..agent import AgentBase, UserAgent
from ..message import AudioBlock


def _serialize_speech(
    speech: AudioBlock | list[AudioBlock] | None,
) -> list[AudioBlock] | None:
    """Serialize speech data to a list of audio block dictionaries.

    Args:
        speech (`AudioBlock | list[AudioBlock] | None`):
         The speech data, can be a single AudioBlock dict or a list of
         AudioBlock dicts.

    Returns:
        A list of audio block dictionaries, or None if no valid speech data.
    """
    if speech is None:
        return None

    # Normalize to list
    if isinstance(speech, dict):
        speech_list = [speech]
    elif isinstance(speech, list):
        speech_list = speech
    else:
        return None

    # Filter and validate audio blocks
    result = []
    for block in speech_list:
        if isinstance(block, dict) and block.get("type") == "audio":
            result.append(block)

    return result if result else None


def as_studio_forward_message_pre_print_hook(
    self: AgentBase,
    kwargs: dict[str, Any],
    studio_url: str,
    run_id: str,
) -> None:
    """The pre-speak hook to forward messages to the studio."""
    msg = kwargs["msg"]
    speech = kwargs.get("speech", None)

    message_data = msg.to_dict()

    if hasattr(self, "_reply_id"):
        reply_id = getattr(self, "_reply_id")
    else:
        reply_id = shortuuid.uuid()

    # Serialize speech data
    speech_data = _serialize_speech(speech)

    n_retry = 0
    while True:
        try:
            payload = {
                "runId": run_id,
                "replyId": reply_id,
                "replyName": getattr(self, "name", msg.name),
                "replyRole": "user"
                if isinstance(self, UserAgent)
                else "assistant",
                "msg": message_data,
            }

            # Add speech data if available
            if speech_data:
                payload["speech"] = speech_data

            res = requests.post(
                f"{studio_url}/trpc/pushMessage",
                json=payload,
            )
            res.raise_for_status()
            break
        except Exception as e:
            if n_retry < 3:
                n_retry += 1
                continue

            raise e from None
