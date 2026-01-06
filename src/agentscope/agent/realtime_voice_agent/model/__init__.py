# -*- coding: utf-8 -*-
"""The realtime voice models."""

from ._voice_model_base import RealtimeVoiceModelBase
from ._dashscope_realtime_voice_model import DashScopeRealtimeVoiceModel


__all__ = [
    "RealtimeVoiceModelBase",
    "DashScopeRealtimeVoiceModel",
]
