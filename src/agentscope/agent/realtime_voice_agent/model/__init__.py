# -*- coding: utf-8 -*-
"""The realtime voice models.

Provides WebSocket-based real-time voice models:
- WebSocketVoiceModelBase: Base class for all WebSocket models
- DashScopeWebSocketModel: DashScope implementation
- GeminiWebSocketModel: Gemini implementation
- OpenAIWebSocketModel: OpenAI implementation
"""

from ._voice_model_base import (
    WebSocketVoiceModelBase,
    LiveEvent,
    LiveEventType,
)
from ._dashscope_websocket_model import DashScopeWebSocketModel

# from ._gemini_websocket_model import GeminiWebSocketModel
# from ._openai_websocket_model import OpenAIWebSocketModel


__all__ = [
    # Base class
    "WebSocketVoiceModelBase",
    # Event types
    "LiveEvent",
    "LiveEventType",
    # WebSocket implementations
    "DashScopeWebSocketModel",
    # "GeminiWebSocketModel",
    # "OpenAIWebSocketModel",
]
