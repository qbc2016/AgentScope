# -*- coding: utf-8 -*-
"""The TTS (Text-to-Speech) module."""

from ._tts_base import TTSModelBase
from ._dashscope_tts_model import DashScopeTTSModel
from ._gemini_tts_model import GeminiTTSModel
from ._openai_tts_model import OpenAITTSModel

__all__ = [
    "TTSModelBase",
    "DashScopeTTSModel",
    "GeminiTTSModel",
    "OpenAITTSModel",
]
