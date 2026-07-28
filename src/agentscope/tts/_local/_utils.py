# -*- coding: utf-8 -*-
"""Shared utilities for local TTS models."""
import base64
import importlib.util
import os
import tempfile
from typing import Any

import numpy as np


_LOCAL_ENGINE_MODULES: dict[str, tuple[str, ...]] = {
    "kokoro": ("kokoro", "soundfile"),
    "chatterbox": ("chatterbox", "soundfile"),
    "luxtts": ("zipvoice.luxvoice", "soundfile"),
    "tada": ("tada", "dac", "soundfile"),
}


def is_local_tts_engine_available(engine: str) -> bool:
    """Return whether the optional dependencies for an engine exist."""
    modules = _LOCAL_ENGINE_MODULES.get(engine)
    if modules is None:
        return True
    try:
        return all(
            importlib.util.find_spec(module) is not None for module in modules
        )
    except (ImportError, AttributeError, ValueError):
        return False


def audio_to_numpy(audio: Any) -> np.ndarray:
    """Convert a generated tensor/array to non-empty NumPy audio."""
    value = audio
    for method_name in ("detach", "squeeze", "cpu", "float"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    result = np.asarray(value).squeeze()
    if result.size == 0:
        raise ValueError("TTS engine returned empty audio.")
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError("TTS engine returned non-numeric audio.")
    return result


def decode_to_tempfile(audio_base64: str) -> str:
    """Decode base64 audio to a temporary WAV file.

    The caller is responsible for cleaning up via
    :func:`cleanup_tempfile` after use.

    Args:
        audio_base64 (`str`):
            Base64-encoded audio data.

    Returns:
        `str`:
            Path to the temporary WAV file.
    """
    audio_bytes = base64.b64decode(audio_base64, validate=True)
    fd, path = tempfile.mkstemp(
        suffix=".wav",
        prefix="agentscope_ref_",
    )
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(audio_bytes)
    except Exception:
        cleanup_tempfile(path)
        raise
    return path


def cleanup_tempfile(path: str | None) -> None:
    """Remove a temporary file if it exists.

    Args:
        path (`str | None`):
            File path to remove, or ``None`` (no-op).
    """
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass
