# -*- coding: utf-8 -*-
"""Feishu interactive-card helpers for the tool-approval flow.

The card round-trips only an opaque ``request_id`` (plus the click's
approve/deny) — the gateway holds all business context. See
``docs/design_channel_redesign.md`` §6.2.
"""
import json
from typing import Any

ACTION_TYPE = "tool_guard_approval"
APPROVE = "approve"
DENY = "deny"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_approval_card(request_id: str, tool_name: str, summary: str) -> str:
    """Build the approval card (JSON string) for a pending tool call."""
    base = {"type": ACTION_TYPE, "request_id": request_id}
    body = f"**工具:** `{tool_name}`"
    if summary:
        body += f"\n**参数:** {_truncate(summary, 800)}"
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "orange",
            "title": {"tag": "plain_text", "content": "🛡️ 工具执行需要确认"},
        },
        "elements": [
            {"tag": "markdown", "content": body},
            {"tag": "hr"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "✅ 允许执行"},
                        "type": "primary",
                        "value": {**base, "action": APPROVE},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "❌ 拒绝"},
                        "type": "danger",
                        "value": {**base, "action": DENY},
                    },
                ],
            },
        ],
    }
    return json.dumps(card, ensure_ascii=False)


def build_resolved_card(outcome: str) -> str:
    """Build the post-decision card (``"approved"`` / ``"denied"``)."""
    approved = outcome == "approved"
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "green" if approved else "red",
            "title": {
                "tag": "plain_text",
                "content": "✅ 已允许执行" if approved else "🚫 已拒绝",
            },
        },
        "elements": [
            {
                "tag": "markdown",
                "content": "工具已被允许执行。" if approved else "工具已被拒绝执行。",
            },
        ],
    }
    return json.dumps(card, ensure_ascii=False)


def parse_action(value: Any) -> tuple[str, bool] | None:
    """Parse a card button value into ``(request_id, approved)`` or None."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(value, dict) or value.get("type") != ACTION_TYPE:
        return None
    request_id = str(value.get("request_id") or "").strip()
    action = str(value.get("action") or "").strip().lower()
    if not request_id or action not in (APPROVE, DENY):
        return None
    return request_id, action == APPROVE


def build_toast(approved: bool) -> Any:
    """Build the synchronous card-callback response (a toast)."""
    toast = {
        "type": "success" if approved else "info",
        "content": "已允许" if approved else "已拒绝",
    }
    try:
        from lark_oapi.event.callback.model.p2_card_action_trigger import (
            P2CardActionTriggerResponse,
        )

        return P2CardActionTriggerResponse({"toast": toast})
    except (ImportError, AttributeError):
        return {"toast": toast}
