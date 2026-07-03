# -*- coding: utf-8 -*-
"""Channel type registry — maps channel_type names to metadata and
JSON Schema definitions for frontend form generation."""
from typing import Any
from pydantic import BaseModel, Field


class ChannelTypeSchema(BaseModel):
    """Metadata and credential schema for a channel type."""

    channel_type: str
    """Type identifier (e.g. 'feishu', 'discord')."""

    display_name: str
    """Human-readable name for the UI."""

    description: str = ""
    """Brief description of the platform."""

    credentials_schema: dict[str, Any] = Field(default_factory=dict)
    """JSON Schema describing required credentials fields."""

    config_schema: dict[str, Any] = Field(default_factory=dict)
    """JSON Schema describing optional platform-specific config."""

    platform_bot_id_field: str = ""
    """Which credential field holds the platform_bot_id."""


class ChannelTypeRegistry:
    """Registry of supported channel types and their schemas."""

    def __init__(self) -> None:
        self._types: dict[str, ChannelTypeSchema] = {}
        self._register_builtin_types()

    def register(self, schema: ChannelTypeSchema) -> None:
        """Register a new channel type."""
        self._types[schema.channel_type] = schema

    def get(self, channel_type: str) -> ChannelTypeSchema | None:
        """Get schema for a channel type."""
        return self._types.get(channel_type)

    def list_types(self) -> list[ChannelTypeSchema]:
        """Return all registered types."""
        return list(self._types.values())

    def extract_platform_bot_id(
        self,
        channel_type: str,
        credentials: dict,
    ) -> str:
        """Extract platform_bot_id from credentials using the schema."""
        schema = self._types.get(channel_type)
        if not schema or not schema.platform_bot_id_field:
            raise ValueError(
                f"Cannot extract platform_bot_id for type '{channel_type}'",
            )
        bot_id = credentials.get(schema.platform_bot_id_field)
        if not bot_id:
            raise ValueError(
                f"Missing '{schema.platform_bot_id_field}' in credentials",
            )
        return str(bot_id)

    def _register_builtin_types(self) -> None:
        """Register built-in platform types."""
        self.register(
            ChannelTypeSchema(
                channel_type="feishu",
                display_name="飞书 (Feishu/Lark)",
                description="飞书机器人，通过长连接模式接收消息",
                credentials_schema={
                    "type": "object",
                    "required": ["app_id", "app_secret"],
                    "properties": {
                        "app_id": {
                            "type": "string",
                            "title": "App ID",
                            "description": "飞书应用的 App ID",
                        },
                        "app_secret": {
                            "type": "string",
                            "title": "App Secret",
                            "description": "飞书应用的 App Secret",
                            "format": "password",
                        },
                    },
                },
                config_schema={
                    "type": "object",
                    "properties": {
                        "only_at_reply": {
                            "type": "boolean",
                            "title": "仅@时回复",
                            "description": "群聊中仅在被@时回复",
                            "default": True,
                        },
                    },
                },
                platform_bot_id_field="app_id",
            ),
        )

        self.register(
            ChannelTypeSchema(
                channel_type="discord",
                display_name="Discord",
                description="Discord bot via Gateway WebSocket",
                credentials_schema={
                    "type": "object",
                    "required": ["bot_token", "application_id"],
                    "properties": {
                        "bot_token": {
                            "type": "string",
                            "title": "Bot Token",
                            "format": "password",
                        },
                        "application_id": {
                            "type": "string",
                            "title": "Application ID",
                        },
                    },
                },
                config_schema={
                    "type": "object",
                    "properties": {},
                },
                platform_bot_id_field="application_id",
            ),
        )

        self.register(
            ChannelTypeSchema(
                channel_type="dingtalk",
                display_name="钉钉 (DingTalk)",
                description="钉钉机器人，通过 Stream 模式接收消息",
                credentials_schema={
                    "type": "object",
                    "required": ["client_id", "client_secret"],
                    "properties": {
                        "client_id": {
                            "type": "string",
                            "title": "Client ID (AppKey)",
                        },
                        "client_secret": {
                            "type": "string",
                            "title": "Client Secret (AppSecret)",
                            "format": "password",
                        },
                    },
                },
                config_schema={
                    "type": "object",
                    "properties": {},
                },
                platform_bot_id_field="client_id",
            ),
        )

        self.register(
            ChannelTypeSchema(
                channel_type="wecom",
                display_name="企业微信 (WeCom)",
                description="企业微信应用机器人",
                credentials_schema={
                    "type": "object",
                    "required": ["corp_id", "agent_id", "secret"],
                    "properties": {
                        "corp_id": {
                            "type": "string",
                            "title": "Corp ID",
                        },
                        "agent_id": {
                            "type": "string",
                            "title": "Agent ID",
                        },
                        "secret": {
                            "type": "string",
                            "title": "Secret",
                            "format": "password",
                        },
                    },
                },
                config_schema={
                    "type": "object",
                    "properties": {},
                },
                platform_bot_id_field="corp_id",
            ),
        )
