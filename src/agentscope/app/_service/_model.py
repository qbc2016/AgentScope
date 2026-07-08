# -*- coding: utf-8 -*-
"""Model service: builds a ChatModelBase from stored credential + config."""
from typing import Type

from fastapi import HTTPException, status

from ..storage import StorageBase, ChatModelConfig
from ...credential import CredentialFactory
from ...model import ChatModelBase
from ..._logging import logger


def _get_model_input_types(
    model_cls: Type[ChatModelBase],
    model_name: str,
) -> list[str] | None:
    """Look up ``input_types`` from the built-in model card for *model_name*.

    Returns ``None`` when no matching card is found (e.g. custom models)
    so callers can fall back to the formatter's default.
    """
    try:
        for card in model_cls.list_models():
            if card.name == model_name:
                return card.input_types
    except Exception:
        logger.debug(
            "Failed to look up model card for %s, "
            "using formatter defaults.",
            model_name,
        )
    return None


async def get_model(
    user_id: str,
    config: ChatModelConfig,
    storage: StorageBase,
) -> ChatModelBase:
    """Get the model instance from the configuration and storage.

    Args:
        user_id (`str`):
            The user id.
        config (`ChatModelConfig`):
            The chat model configuration.
        storage (`StorageBase`):
            The storage instance.

    Returns:
        `ChatModelBase`:
            The model instance.
    """
    credential_record = await storage.get_credential(
        user_id,
        config.credential_id,
    )
    if credential_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Credential {config.credential_id!r} not found.",
        )

    credential = CredentialFactory.from_dict(credential_record.data)
    model_cls = credential.get_chat_model_class()
    parameters = (
        model_cls.Parameters(**config.parameters)
        if config.parameters
        else None
    )
    model = model_cls(
        credential=credential,
        model=config.model,
        parameters=parameters,
    )

    input_types = _get_model_input_types(model_cls, config.model)
    if input_types is not None and hasattr(model, "formatter"):
        model.formatter.input_types = input_types

    return model
