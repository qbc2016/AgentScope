# -*- coding: utf-8 -*-
"""Feishu channel JSON Schema definitions.

These schemas are used by the frontend for form generation when
configuring a Feishu channel.
"""

FEISHU_CREDENTIALS_SCHEMA = {
    "type": "object",
    "required": ["app_id", "app_secret"],
    "properties": {
        "app_id": {
            "type": "string",
            "title": "App ID",
            "description": "飞书开放平台应用的 App ID",
        },
        "app_secret": {
            "type": "string",
            "title": "App Secret",
            "description": "飞书开放平台应用的 App Secret",
            "format": "password",
        },
    },
}

FEISHU_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "only_at_reply": {
            "type": "boolean",
            "title": "仅@时回复",
            "description": "在群聊中，仅当机器人被@时才回复消息",
            "default": True,
        },
    },
}
