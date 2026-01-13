# -*- coding: utf-8 -*-
"""The realtime voice agent modules.

Provides:
- WebSocketVoiceAgent: WebSocket-based voice agent for real-time voice
- WebSocketVoiceModelBase: Base class for all WebSocket voice models
- DashScopeWebSocketModel, GeminiWebSocketModel, OpenAIWebSocketModel
- RealtimeVoiceInput, MsgStream, VoiceMsgHub
"""

from .agent import WebSocketVoiceAgent
from .model import (
    WebSocketVoiceModelBase,
    DashScopeWebSocketModel,
    # GeminiWebSocketModel,
    # OpenAIWebSocketModel,
    LiveEvent,
    LiveEventType,
)
from ._utils import RealtimeVoiceInput, MsgStream, VoiceMsgHub

__all__ = [
    # Agents
    "WebSocketVoiceAgent",
    # Base class
    "WebSocketVoiceModelBase",
    # Event types
    "LiveEvent",
    "LiveEventType",
    # WebSocket models
    "DashScopeWebSocketModel",
    # "GeminiWebSocketModel",
    # "OpenAIWebSocketModel",
    # Utils
    "RealtimeVoiceInput",
    "MsgStream",
    "VoiceMsgHub",
]
