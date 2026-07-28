# -*- coding: utf-8 -*-
"""Shared utilities for local TTS models."""
import base64
import os
import tempfile


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
    audio_bytes = base64.b64decode(audio_base64)
    fd, path = tempfile.mkstemp(
        suffix=".wav",
        prefix="agentscope_ref_",
    )
    try:
        os.write(fd, audio_bytes)
    finally:
        os.close(fd)
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
