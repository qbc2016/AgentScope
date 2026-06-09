# -*- coding: utf-8 -*-
"""The TTS (Text-to-Speech) module in AgentScope."""

from ._tts_base import TTSModelBase
from ._tts_response import TTSResponse, TTSUsage
from ._dashscope_tts_model import DashScopeTTSModel

__all__ = [
    "TTSModelBase",
    "TTSResponse",
    "TTSUsage",
    "DashScopeTTSModel",
]
