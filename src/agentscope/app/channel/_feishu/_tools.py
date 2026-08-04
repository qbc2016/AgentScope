# -*- coding: utf-8 -*-
"""Feishu channel tools exposed to the agent.

Two families, designed to form a closed chain:

* **discovery** (``ListChats`` / ``ListChatMembers``) — read-only lookups
  that hand back a ``receive_id`` + ``receive_id_type`` pair;
* **send** (``SendMessage`` / ``SendFile`` / ``SendImage``) — write actions
  that consume such a pair to reach a chat/user *other* than the current
  conversation. Replies to the current conversation are delivered by the
  channel automatically and must not go through these tools.

Each tool closes over the running :class:`FeishuChannel`, so it acts with
the same bot credentials and connection the channel already holds.
"""
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import Field

from ....message import TextBlock, ToolResultState
from ....permission import (
    PermissionBehavior,
    PermissionContext,
    PermissionDecision,
)
from ....tool import ParamsBase, ToolBase, ToolChunk

if TYPE_CHECKING:
    from ....workspace import WorkspaceBase
    from ._channel import FeishuChannel


def build_feishu_tools(
    channel: "FeishuChannel",
    workspace: "WorkspaceBase",
) -> list[ToolBase]:
    """Instantiate every Feishu agent tool bound to ``channel``.

    File-sending tools also take the session's ``workspace`` so they read
    their payload from the agent's workspace, not the host filesystem.
    """
    return [
        ListChats(channel),
        ListChatMembers(channel),
        SendMessage(channel),
        SendFile(channel, workspace),
        SendImage(channel, workspace),
    ]


def _ack(data: dict | None, what: str) -> ToolChunk:
    """Turn a Feishu send response into a success/error chunk."""
    if data and data.get("code") == 0:
        return ToolChunk(content=[TextBlock(text=f"Sent {what}.")])
    msg = (data or {}).get("msg") or "the platform rejected the request"
    return ToolChunk(
        content=[TextBlock(text=f"Failed to send {what}: {msg}")],
        state=ToolResultState.ERROR,
    )


class _FeishuTool(ToolBase):
    """Shared base: holds the channel and a read/write permission split.

    Discovery tools (``is_read_only=True``) are allowed outright. Send
    tools reach people/groups outside the current conversation, so they
    default to ASK — routing through the channel's own confirmation UI.
    """

    is_concurrency_safe: bool = False
    is_state_injected: bool = False
    is_external_tool: bool = False
    is_mcp: bool = False
    mcp_name: str | None = None

    def __init__(self, channel: "FeishuChannel") -> None:
        """Bind the running channel the tool acts through.

        Args:
            channel (`FeishuChannel`): The live channel to send / query.
        """
        super().__init__()
        self._channel = channel

    async def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        """Allow reads; ask before sending to another chat/user."""
        if self.is_read_only:
            return PermissionDecision(
                behavior=PermissionBehavior.ALLOW,
                message=f"{self.name} is a read-only lookup.",
            )
        return PermissionDecision(
            behavior=PermissionBehavior.ASK,
            message="Sending to another Feishu chat/user needs the user's "
            "confirmation.",
        )


class _FeishuFileTool(_FeishuTool):
    """A send tool that reads its payload from the agent's workspace."""

    def __init__(
        self,
        channel: "FeishuChannel",
        workspace: "WorkspaceBase",
    ) -> None:
        """Bind the channel and the session workspace to read files from.

        Args:
            channel (`FeishuChannel`): The live channel to send through.
            workspace (`WorkspaceBase`): The session workspace the file
                payload is read from.
        """
        super().__init__(channel)
        self._workspace = workspace

    def _resolve(self, path: str) -> str:
        """Map an agent-supplied path to a backend-side workspace path.

        Accepts a ``workspace://`` reference, a workspace-relative path,
        or an absolute in-sandbox path — all resolved against the
        session's workspace, never the host.
        """
        backend = self._workspace.get_backend()
        if path.startswith("workspace://"):
            rel = path[len("workspace://") :].lstrip("/")
            return backend.join_path(self._workspace.workdir, rel)
        if path.startswith("/"):
            return path
        return backend.join_path(self._workspace.workdir, path)

    async def _read(self, path: str) -> bytes:
        """Read ``path`` from the workspace as bytes."""
        return await self._workspace.get_backend().read_file(
            self._resolve(path),
        )


# -- Discovery (read-only) --


class _ListChatsParams(ParamsBase):
    query: str | None = Field(
        default=None,
        description="Optional case-insensitive substring to filter groups "
        "by name. Omit to list all.",
    )


_LIST_CHATS_DESC = """List the Feishu groups this bot belongs to, to obtain \
a target for sending.

## When to Use
- You need to message a *group* other than the current conversation and \
must first find its id.

## Output
A JSON array of ``{receive_id, receive_id_type, name}``. ``receive_id_type`` \
is always ``"chat_id"``. Copy ``receive_id`` + ``receive_id_type`` verbatim \
into a Send* tool. To reach a specific *person* in a group, take that \
group's ``receive_id`` and call ``ListChatMembers`` next."""


class ListChats(_FeishuTool):
    """List the bot's Feishu groups as ready-to-send address pairs."""

    name: str = "ListChats"
    description: str = _LIST_CHATS_DESC
    is_read_only: bool = True
    input_schema: dict = _ListChatsParams.model_json_schema()

    async def __call__(self, query: str | None = None) -> ToolChunk:
        """Return the bot's chats filtered by ``query``."""
        chats = await self._channel.list_bot_chats()
        needle = (query or "").lower()
        items = [
            {
                "receive_id": chat.get("chat_id", ""),
                "receive_id_type": "chat_id",
                "name": chat.get("name", ""),
            }
            for chat in chats
            if not needle or needle in (chat.get("name", "") or "").lower()
        ]
        return ToolChunk(
            content=[TextBlock(text=json.dumps(items, ensure_ascii=False))],
        )


class _ListChatMembersParams(ParamsBase):
    chat_id: str = Field(
        description="The group's chat_id, taken from a ListChats result.",
    )


_LIST_MEMBERS_DESC = """List the members of a Feishu group, to obtain a \
person's id for a direct message.

## When to Use
- You need to message a *specific person* directly and must first find \
their id. Get the group's ``chat_id`` from ``ListChats``, then call this.

## Output
A JSON array of ``{receive_id, receive_id_type, name}``. ``receive_id_type`` \
is always ``"open_id"``. Copy the ``receive_id`` + ``receive_id_type`` of \
the person you want into a Send* tool to message them directly."""


class ListChatMembers(_FeishuTool):
    """List a group's members as ready-to-send address pairs."""

    name: str = "ListChatMembers"
    description: str = _LIST_MEMBERS_DESC
    is_read_only: bool = True
    input_schema: dict = _ListChatMembersParams.model_json_schema()

    async def __call__(self, chat_id: str) -> ToolChunk:
        """Return the members of ``chat_id`` as address pairs."""
        members = await self._channel.list_chat_members(chat_id)
        items = [
            {
                "receive_id": member.get("open_id", ""),
                "receive_id_type": "open_id",
                "name": member.get("name", ""),
            }
            for member in members
        ]
        return ToolChunk(
            content=[TextBlock(text=json.dumps(items, ensure_ascii=False))],
        )


# -- Send (write; ASK-gated) --


class _SendMessageParams(ParamsBase):
    receive_id: str = Field(
        description="Target id, taken verbatim from a ListChats / "
        "ListChatMembers result.",
    )
    receive_id_type: str = Field(
        description="Must match the id: 'chat_id' for a group, 'open_id' "
        "for a person. Copy it from the same discovery result.",
        json_schema_extra={"enum": ["chat_id", "open_id"]},
    )
    text: str = Field(description="The message text to send.")


_SEND_MESSAGE_DESC = """Send a text message to a Feishu chat or person \
OTHER than the current conversation.

## When to Use
- The user asks you to notify or relay something to a *different* group or \
person (e.g. "tell the finance group ...", "let Li Si know ...").

## When NOT to Use
- To answer the person you are talking with now — that reply is sent \
automatically. Never use this tool for the current conversation.

## How to Use
Obtain ``receive_id`` first: a group's via ``ListChats``, a person's via \
``ListChatMembers``. Pass ``receive_id`` and ``receive_id_type`` exactly as \
returned. Sending requires the user's confirmation."""


class SendMessage(_FeishuTool):
    """Send text to another Feishu chat/user."""

    name: str = "SendMessage"
    description: str = _SEND_MESSAGE_DESC
    is_read_only: bool = False
    input_schema: dict = _SendMessageParams.model_json_schema()

    async def __call__(
        self,
        receive_id: str,
        receive_id_type: str,
        text: str,
    ) -> ToolChunk:
        """Send ``text`` to ``receive_id``."""
        data = await self._channel.send_message_to(
            receive_id,
            receive_id_type,
            text,
        )
        return _ack(data, f"message to {receive_id}")


class _SendFileParams(ParamsBase):
    path: str = Field(
        description="Path to the file in your workspace, e.g. one you just "
        "created. Workspace-relative (recommended) or a workspace:// "
        "reference.",
    )
    receive_id: str = Field(
        description="Target id, taken verbatim from a ListChats / "
        "ListChatMembers result.",
    )
    receive_id_type: str = Field(
        description="Must match the id: 'chat_id' for a group, 'open_id' "
        "for a person.",
        json_schema_extra={"enum": ["chat_id", "open_id"]},
    )


_SEND_FILE_DESC = """Send a file to a Feishu chat or person OTHER than the \
current conversation.

## When to Use
- The user asks you to deliver a file (a report, export, ...) to a \
*different* group or person.

## How to Use
Give ``path`` — a file in your workspace (the one you produced it in). \
Obtain ``receive_id`` via ``ListChats`` (group) or ``ListChatMembers`` \
(person) and pass ``receive_id`` + ``receive_id_type`` verbatim. Sending \
requires the user's confirmation.

To send an image so it renders inline, use ``SendImage`` instead."""


class SendFile(_FeishuFileTool):
    """Upload and send a file to another Feishu chat/user."""

    name: str = "SendFile"
    description: str = _SEND_FILE_DESC
    is_read_only: bool = False
    input_schema: dict = _SendFileParams.model_json_schema()

    async def __call__(
        self,
        path: str,
        receive_id: str,
        receive_id_type: str,
    ) -> ToolChunk:
        """Read ``path`` from the workspace and send it to ``receive_id``."""
        try:
            raw = await self._read(path)
        except Exception as e:  # pylint: disable=broad-except
            return ToolChunk(
                content=[
                    TextBlock(text=f"SendFile: cannot read {path!r}: {e}"),
                ],
                state=ToolResultState.ERROR,
            )
        data = await self._channel.send_file_to(
            receive_id,
            receive_id_type,
            raw,
            Path(path).name,
        )
        return _ack(data, f"file {Path(path).name} to {receive_id}")


class _SendImageParams(ParamsBase):
    path: str = Field(
        description="Path to the image file in your workspace. "
        "Workspace-relative (recommended) or a workspace:// reference.",
    )
    receive_id: str = Field(
        description="Target id, taken verbatim from a ListChats / "
        "ListChatMembers result.",
    )
    receive_id_type: str = Field(
        description="Must match the id: 'chat_id' for a group, 'open_id' "
        "for a person.",
        json_schema_extra={"enum": ["chat_id", "open_id"]},
    )


_SEND_IMAGE_DESC = """Send an image to a Feishu chat or person OTHER than \
the current conversation, rendered inline.

## When to Use
- The user asks you to send a picture/chart to a *different* group or \
person, and you want it shown inline (not as a file attachment).

## How to Use
Give ``path`` to the image file. Obtain ``receive_id`` via ``ListChats`` \
(group) or ``ListChatMembers`` (person) and pass ``receive_id`` + \
``receive_id_type`` verbatim. Sending requires the user's confirmation."""


class SendImage(_FeishuFileTool):
    """Upload and send an image to another Feishu chat/user."""

    name: str = "SendImage"
    description: str = _SEND_IMAGE_DESC
    is_read_only: bool = False
    input_schema: dict = _SendImageParams.model_json_schema()

    async def __call__(
        self,
        path: str,
        receive_id: str,
        receive_id_type: str,
    ) -> ToolChunk:
        """Read the image at ``path`` from the workspace and send it."""
        try:
            raw = await self._read(path)
        except Exception as e:  # pylint: disable=broad-except
            return ToolChunk(
                content=[
                    TextBlock(text=f"SendImage: cannot read {path!r}: {e}"),
                ],
                state=ToolResultState.ERROR,
            )
        data = await self._channel.send_image_to(
            receive_id,
            receive_id_type,
            raw,
        )
        return _ack(data, f"image to {receive_id}")
