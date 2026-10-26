# -*- coding: utf-8 -*-
"""Unit tests for :func:`get_model` — the chat model factory."""
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from agentscope.app._service._model import get_model
from agentscope.app.storage import ChatModelConfig
from agentscope.credential import CredentialFactory, OpenAICredential
from agentscope.model import OpenAIChatModel, OpenAIResponseModel

_CREDENTIAL = {"type": "deepseek_credential", "api_key": "test"}


class _CustomOpenAIChatModel(OpenAIChatModel):
    """A custom adapter selected through the legacy credential getter."""


class _CustomOpenAICredential(OpenAICredential):
    """An existing credential extension using the single-class getter."""

    @classmethod
    def get_chat_model_class(cls) -> type[OpenAIChatModel]:
        return _CustomOpenAIChatModel


class OpenAICredentialModelTest(IsolatedAsyncioTestCase):
    """Check model resolution for built-in and extended credentials."""

    def test_default_models(self) -> None:
        """Standard credentials continue to expose both API surfaces."""
        self.assertListEqual(
            OpenAICredential.get_chat_model_classes(),
            [OpenAIChatModel, OpenAIResponseModel],
        )

    def test_legacy_model_override(self) -> None:
        """A legacy override keeps its model without adding other APIs."""
        for model_cls in (_CustomOpenAIChatModel, OpenAIResponseModel):
            with self.subTest(model=model_cls.__name__):
                with patch.object(
                    _CustomOpenAICredential,
                    "get_chat_model_class",
                    return_value=model_cls,
                ):
                    self.assertListEqual(
                        _CustomOpenAICredential.get_chat_model_classes(),
                        [model_cls],
                    )

    async def test_factory_uses_legacy_model_override(self) -> None:
        """Saved configs without model_class still use the custom adapter."""
        access = MagicMock(
            resolve_credential=AsyncMock(
                return_value=SimpleNamespace(data={}),
            ),
        )
        credential = _CustomOpenAICredential(api_key="test")
        with patch.object(
            CredentialFactory,
            "from_dict",
            return_value=credential,
        ):
            model = await get_model(
                "user-1",
                ChatModelConfig(
                    type="openai_credential",
                    credential_id="cred-1",
                    model="custom-model",
                    parameters={"max_tokens": 123},
                ),
                access,
            )
        try:
            self.assertEqual(
                (type(model), model.model, model.parameters.max_tokens),
                (_CustomOpenAIChatModel, "custom-model", 123),
            )
        finally:
            await model.client.close()


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
