# -*- coding: utf-8 -*-
"""The TTS model router."""

from fastapi import APIRouter, Depends, HTTPException, status

from ._schema import ListTTSModelsResponse, ListTTSModelsRequest
from .._service import ResourceAccessService, discover_tts_models
from ..deps import get_current_user_id, get_resource_access_service
from ...credential import CredentialFactory

tts_model_router = APIRouter(
    prefix="/tts-model",
    tags=["tts-model"],
    responses={404: {"description": "Not found"}},
)


@tts_model_router.get(
    "/",
    response_model=ListTTSModelsResponse,
    summary="List all candidate TTS models under the given credential type",
)
async def list_tts_models(
    body: ListTTSModelsRequest = Depends(),
    user_id: str = Depends(get_current_user_id),
    access: ResourceAccessService = Depends(get_resource_access_service),
) -> ListTTSModelsResponse:
    """Return static or credential-specific candidate TTS models.

    Args:
        body (ListTTSModelsRequest): The request body.

    Returns:
        `ListTTSModelsResponse`: The response body.
    """
    if body.credential_id is not None:
        record = await access.resolve_credential(user_id, body.credential_id)
        credential = CredentialFactory.from_dict(record.data)
        if body.provider is not None and credential.type != body.provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Credential {body.credential_id!r} has type "
                    f"{credential.type!r}, not {body.provider!r}."
                ),
            )
        models = await discover_tts_models(record, credential)
        return ListTTSModelsResponse(models=models, total=len(models))

    assert body.provider is not None
    credential_cls = CredentialFactory.get_credential_class(body.provider)
    if credential_cls is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{body.provider}' not found.",
        )

    models = credential_cls.list_tts_models()
    return ListTTSModelsResponse(models=models, total=len(models))
