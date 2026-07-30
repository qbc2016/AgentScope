# -*- coding: utf-8 -*-
"""The TTS (Text-to-Speech) module in AgentScope."""

from ._tts_base import TTSModelBase
from ._tts_model_card import TTSModelCard
from ._tts_response import TTSResponse, TTSUsage
from ._dashscope import (
    DashScopeCosyVoiceTTSModel,
    DashScopeTTSModel,
    DashScopeRealtimeTTSModel,
)
from ._gemini import GeminiTTSModel
from ._local import (
    ChatterboxTTSModel,
    KokoroTTSModel,
    LuxTTSModel,
    TadaTTSModel,
)
from ._openai import OpenAITTSModel
from ._remote import RemoteTTSError, RemoteTTSModel
from ._voicebox import VoiceboxTTSModel

__all__ = [
    "TTSModelBase",
    "TTSModelCard",
    "TTSResponse",
    "TTSUsage",
    "ChatterboxTTSModel",
    "DashScopeCosyVoiceTTSModel",
    "DashScopeTTSModel",
    "DashScopeRealtimeTTSModel",
    "GeminiTTSModel",
    "KokoroTTSModel",
    "LuxTTSModel",
    "OpenAITTSModel",
    "RemoteTTSError",
    "RemoteTTSModel",
    "TadaTTSModel",
    "VoiceboxTTSModel",
]
