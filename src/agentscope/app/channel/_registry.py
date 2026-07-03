# -*- coding: utf-8 -*-
"""Channel type registry — maps channel_type names to metadata and
JSON Schema definitions for frontend form generation."""
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ._base import ChannelBase

ChannelFactory = Callable[..., "ChannelBase"]


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
        self._factories: dict[str, ChannelFactory] = {}
        self._register_builtin_types()

    def register(
        self,
        schema: ChannelTypeSchema,
        factory: ChannelFactory | None = None,
    ) -> None:
        """Register a new channel type with optional factory function."""
        self._types[schema.channel_type] = schema
        if factory is not None:
            self._factories[schema.channel_type] = factory

    def register_factory(
        self,
        channel_type: str,
        factory: ChannelFactory,
    ) -> None:
        """Register a factory function for an existing channel type."""
        self._factories[channel_type] = factory

    def create_channel(
        self,
        channel_type: str,
        channel_id: str,
        credentials: dict,
        config: dict,
    ) -> ChannelBase:
        """Create a channel instance using the registered factory.

        Raises ValueError if no factory is registered for the type.
        """
        factory = self._factories.get(channel_type)
        if factory is None:
            raise ValueError(
                f"No factory registered for channel type '{channel_type}'. "
                f"Use registry.register(schema, factory=...) to register.",
            )
        return factory(
            channel_id=channel_id,
            credentials=credentials,
            config=config,
        )

    def has_factory(self, channel_type: str) -> bool:
        """Check if a factory is registered for the given type."""
        return channel_type in self._factories

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

        self._register_builtin_factories()

    def _register_builtin_factories(self) -> None:
        """Register factory functions for built-in channel types."""

        def _feishu_factory(
            channel_id: str,
            credentials: dict,
            config: dict,
        ) -> ChannelBase:
            from .feishu import FeishuChannel

            return FeishuChannel(
                channel_id=channel_id,
                app_id=credentials["app_id"],
                app_secret=credentials["app_secret"],
                **config,
            )

        def _discord_factory(
            channel_id: str,
            credentials: dict,
            config: dict,
        ) -> ChannelBase:
            from .discord import DiscordChannel

            return DiscordChannel(
                channel_id=channel_id,
                bot_token=credentials["bot_token"],
                **config,
            )

        def _dingtalk_factory(
            channel_id: str,
            credentials: dict,
            config: dict,
        ) -> ChannelBase:
            from .dingtalk import DingTalkChannel

            return DingTalkChannel(
                channel_id=channel_id,
                client_id=credentials["client_id"],
                client_secret=credentials["client_secret"],
                **config,
            )

        def _wecom_factory(
            channel_id: str,
            credentials: dict,
            config: dict,
        ) -> ChannelBase:
            from .wecom import WeChatWorkChannel

            return WeChatWorkChannel(
                channel_id=channel_id,
                corp_id=credentials["corp_id"],
                agent_id=credentials["agent_id"],
                secret=credentials["secret"],
                **config,
            )

        self.register_factory("feishu", _feishu_factory)
        self.register_factory("discord", _discord_factory)
        self.register_factory("dingtalk", _dingtalk_factory)
        self.register_factory("wecom", _wecom_factory)
