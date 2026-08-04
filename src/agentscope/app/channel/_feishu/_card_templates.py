# -*- coding: utf-8 -*-
"""Feishu interactive-card helpers for the tool-approval flow.

The card round-trips only an opaque ``request_id`` (plus the click's
approve/deny) — the gateway holds all business context.
"""
import json
from typing import Any

_ACTION_TYPE = "tool_guard_approval"
_APPROVE = "approve"
_DENY = "deny"


def _build_approval_card(
    request_id: str,
    tool_name: str,
    summary: str,
) -> str:
    """Build the approval card (JSON string) for a pending tool call.

    Args:
        request_id (`str`):
            Opaque token embedded in both buttons so the click can be
            routed back to the parked request.
        tool_name (`str`):
            Name of the tool awaiting approval, shown in the card body.
        summary (`str`):
            A rendering of the tool arguments; truncated when long.

    Returns:
        `str`: The card as a JSON string.
    """
    base = {"type": _ACTION_TYPE, "request_id": request_id}
    body = f"**Tool:** `{tool_name}`"
    if summary:
        shown = summary if len(summary) <= 800 else summary[:799] + "…"
        body += f"\n**Arguments:** {shown}"
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "orange",
            "title": {
                "tag": "plain_text",
                "content": "🛡️ Tool execution needs approval",
            },
        },
        "elements": [
            {"tag": "markdown", "content": body},
            {"tag": "hr"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "✅ Allow"},
                        "type": "primary",
                        "value": {**base, "action": _APPROVE},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "❌ Deny"},
                        "type": "danger",
                        "value": {**base, "action": _DENY},
                    },
                ],
            },
        ],
    }
    return json.dumps(card, ensure_ascii=False)


def _build_resolved_card(outcome: str) -> str:
    """Build the post-decision card that replaces the approval card.

    Args:
        outcome (`str`):
            ``"approved"`` or ``"denied"`` — selects the card's colour
            and text.

    Returns:
        `str`: The card as a JSON string.
    """
    approved = outcome == "approved"
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "green" if approved else "red",
            "title": {
                "tag": "plain_text",
                "content": "✅ Allowed" if approved else "🚫 Denied",
            },
        },
        "elements": [
            {
                "tag": "markdown",
                "content": (
                    "The tool was allowed to run."
                    if approved
                    else "The tool was denied."
                ),
            },
        ],
    }
    return json.dumps(card, ensure_ascii=False)


def _parse_action(value: Any) -> tuple[str, bool] | None:
    """Parse a card button's value into ``(request_id, approved)``.

    Args:
        value (`Any`):
            The clicked button's ``value`` field — a dict (or a JSON
            string) carrying ``type`` / ``request_id`` / ``action``.

    Returns:
        `tuple[str, bool] | None`:
            ``(request_id, approved)`` for a valid approval button, or
            ``None`` if the value is not one of ours.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(value, dict) or value.get("type") != _ACTION_TYPE:
        return None
    request_id = str(value.get("request_id") or "").strip()
    action = str(value.get("action") or "").strip().lower()
    if not request_id or action not in (_APPROVE, _DENY):
        return None
    return request_id, action == _APPROVE


def _build_toast(approved: bool) -> Any:
    """Build the synchronous card-callback response (a toast).

    Args:
        approved (`bool`):
            The decision, selecting the toast style and text.

    Returns:
        `Any`:
            A ``P2CardActionTriggerResponse`` when lark_oapi is
            importable, else a plain dict with the same shape.
    """
    toast = {
        "type": "success" if approved else "info",
        "content": "Allowed" if approved else "Denied",
    }
    try:
        from lark_oapi.event.callback.model.p2_card_action_trigger import (
            P2CardActionTriggerResponse,
        )

        return P2CardActionTriggerResponse({"toast": toast})
    except (ImportError, AttributeError):
        return {"toast": toast}
