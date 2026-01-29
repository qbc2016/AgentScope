# -*- coding: utf-8 -*-
"""The client events for web-to-backend communication."""
from enum import Enum


class ClientEventType(str, Enum):
    """Types of client events for web-to-backend communication."""

    # ============== Session control ================
    CLIENT_SESSION_CREATE = "client.session.create"
    """The user creates a new session in the frontend."""

    CLIENT_SESSION_END = "client.session.end"
    """The user ends the current session in the frontend."""

    # ============== Response control ================
    CLIENT_RESPONSE_CREATE = "client.response.create"
    """The user requests the agent to generate a response immediately."""

    CLIENT_RESPONSE_CANCEL = "client.response.cancel"
    """The user interrupts the agent's current response generation."""

    CLIENT_IMAGE_APPEND = "client.image.append"
    """The user appends an image input to the current session."""

    CLIENT_TEXT_APPEND = "client.text.append"
    """The user appends a text input to the current session."""

    CLIENT_AUDIO_APPEND = "client.audio.append"
    """The user appends an audio input to the current session."""

    CLIENT_AUDIO_COMMIT = "client.audio.commit"
    """The user commits the audio input to signal end of input."""

    CLIENT_TOOL_RESULT = "client.tool.result"
    """The tool result executed in the frontend is sent back to the backend."""


class ClientEvent:
    """Realtime client events."""
