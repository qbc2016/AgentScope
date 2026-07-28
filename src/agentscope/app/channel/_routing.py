# -*- coding: utf-8 -*-
"""Pure routing: resolve an inbound event to ``(agent_id, session_id)``.

There is no persisted channel→session mapping table. Given the routing
rules on the channel record, both the target agent and the session id
are computed deterministically from the event — so every node derives
the same result with zero coordination, and session creation is
idempotent (``get_or_create`` on a derived id).

See ``docs/design_channel_redesign.md`` §3.
"""
from uuid import NAMESPACE_URL, uuid5

from ..storage import ChannelRecord, ChannelBinding, SessionScope
from ._base import ChannelEvent


# Fixed namespace so derived session ids are stable across processes and
# restarts. Do not change — it would orphan every existing session.
_SESSION_NAMESPACE = uuid5(NAMESPACE_URL, "agentscope.channel.session")


def _event_field(event: ChannelEvent, match_key: str) -> str | None:
    """Return the event value a binding's ``match_key`` refers to.

    ``chat_id`` / ``user_id`` map to first-class event fields; any other
    key is looked up in ``event.metadata`` (e.g. ``chat_type``).
    """
    if match_key == "chat_id":
        return event.chat_id
    if match_key == "user_id":
        return event.channel_user_id
    value = event.metadata.get(match_key)
    return str(value) if value is not None else None


def _match(event: ChannelEvent, binding: ChannelBinding) -> bool:
    """Whether ``binding`` matches ``event`` (``"*"`` matches anything)."""
    if binding.match_value == "*":
        return True
    return _event_field(event, binding.match_key) == binding.match_value


def _first_match(event: ChannelEvent, record: ChannelRecord) -> ChannelBinding:
    """Return the first matching binding.

    ``RoutingConfig`` guarantees a trailing catch-all, so a match always
    exists; the fallback is defensive only.
    """
    for binding in record.routing.bindings:
        if _match(event, binding):
            return binding
    return record.routing.bindings[-1]


def _scope_key(event: ChannelEvent, scope: SessionScope) -> str:
    """Project the ``(chat_id, user_id)`` pair per the session scope."""
    if scope is SessionScope.PER_CHAT_USER:
        return f"{event.chat_id}:{event.channel_user_id}"
    # PER_CHAT — a DM is naturally per-user (its chat has only that user).
    return event.chat_id


def resolve(event: ChannelEvent, record: ChannelRecord) -> tuple[str, str]:
    """Resolve an event to ``(agent_id, session_id)``.

    The session id embeds ``agent_id``, so different agents never share
    a session even at the same scope.

    Args:
        event (`ChannelEvent`): The inbound event.
        record (`ChannelRecord`): The channel's configuration.

    Returns:
        `tuple[str, str]`: ``(agent_id, session_id)``.
    """
    binding = _first_match(event, record)
    scope_key = _scope_key(event, binding.session_scope)
    session_id = str(
        uuid5(
            _SESSION_NAMESPACE,
            f"{record.id}:{binding.agent_id}:{scope_key}",
        ),
    )
    return binding.agent_id, session_id
