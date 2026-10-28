# -*- coding: utf-8 -*-
"""Credential endpoint resolution shared by enrollment and synthesis."""
from urllib.parse import urlsplit


def _dashscope_url(
    base_url: str,
    path: str,
    *,
    websocket: bool = False,
) -> str:
    """Replace the API suffix, preserving the host and proxy path prefix."""
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("DashScope base_url must be an absolute HTTP(S) URL")
    prefix = parsed.path.rstrip("/")
    for suffix in ("/compatible-mode/v1", "/api/v1"):
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break
    scheme = parsed.scheme
    if websocket:
        scheme = "wss" if scheme == "https" else "ws"
    return parsed._replace(
        scheme=scheme,
        path=prefix + path,
        fragment="",
    ).geturl()
