# -*- coding: utf-8 -*-
"""Unit tests for :func:`get_model` — the chat model factory."""
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from agentscope.app._service._model import get_model
from agentscope.app.storage import ChatModelConfig

_CREDENTIAL = {"type": "deepseek_credential", "api_key": "test"}


class GetModelContextSizeTest(IsolatedAsyncioTestCase):
    """The factory must forward the model card's ``context_size``."""

    async def test_uses_card_context_size(self) -> None:
        """A matching card's value wins over the class default."""
        access = MagicMock(
            resolve_credential=AsyncMock(
                return_value=SimpleNamespace(data=_CREDENTIAL),
            ),
        )

        model = await get_model(
            "user-1",
            ChatModelConfig(
                type="deepseek",
                credential_id="cred-1",
                model="deepseek-v4-flash",
                parameters={},
            ),
            access,
        )

        self.assertEqual(1000000, model.context_size)

    async def test_keeps_default_for_unknown_model(self) -> None:
        """A model without a card keeps the constructor default."""
        access = MagicMock(
            resolve_credential=AsyncMock(
                return_value=SimpleNamespace(data=_CREDENTIAL),
            ),
        )

        model = await get_model(
            "user-1",
            ChatModelConfig(
                type="deepseek",
                credential_id="cred-1",
                model="my-custom-model",
                parameters={},
            ),
            access,
        )

        self.assertEqual(65536, model.context_size)

    async def test_classifier_only_credential_rejects_chat_model(self) -> None:
        """A classifier-only credential should fail with a clear error."""
        access = MagicMock(
            resolve_credential=AsyncMock(
                return_value=SimpleNamespace(
                    data={
                        "type": "typesafe_credential",
                        "api_key": "test",
                    },
                ),
            ),
        )

        with self.assertRaises(HTTPException) as context:
            await get_model(
                "user-1",
                ChatModelConfig(
                    type="typesafe_credential",
                    credential_id="cred-1",
                    model="jev-latest",
                    parameters={},
                ),
                access,
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(
            context.exception.detail,
            "Provider 'typesafe_credential' does not support chat models.",
        )
