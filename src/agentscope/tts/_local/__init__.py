# -*- coding: utf-8 -*-
"""Local TTS engines module."""
from ._kokoro import KokoroTTSModel
from ._chatterbox import ChatterboxTTSModel
from ._luxtts import LuxTTSModel
from ._tada import TadaTTSModel

__all__ = [
    "KokoroTTSModel",
    "ChatterboxTTSModel",
    "LuxTTSModel",
    "TadaTTSModel",
]
