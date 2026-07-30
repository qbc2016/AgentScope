# -*- coding: utf-8 -*-
"""The TTS model router."""

from fastapi import APIRouter, Depends, HTTPException, status

from ._schema import (
    ListTTSModelsResponse,
    ListTTSModelsRequest,
    VoiceboxClientSetupResponse,
)
from ..deps import get_current_user_id, get_resource_access_service
from .._service import ResourceAccessService, get_voicebox_client_id
from ...credential import CredentialFactory, VoiceboxCredential
from ...tts import VoiceboxTTSModel

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
) -> ListTTSModelsResponse:
    """Return all candidate TTS models under the given credential type.

    Args:
        body (ListTTSModelsRequest): The request body.

    Returns:
        `ListTTSModelsResponse`: The response body.
    """
    credential_cls = CredentialFactory.get_credential_class(body.provider)
    if credential_cls is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{body.provider}' not found.",
        )

    models = credential_cls.list_tts_models()
    return ListTTSModelsResponse(models=models, total=len(models))


@tts_model_router.get(
    "/voicebox-setup",
    response_model=VoiceboxClientSetupResponse,
    summary="Inspect Voicebox connection and client binding",
)
async def get_voicebox_setup(
    credential_id: str,
    user_id: str = Depends(get_current_user_id),
    access: ResourceAccessService = Depends(get_resource_access_service),
) -> VoiceboxClientSetupResponse:
    """Return setup information needed by the Voicebox frontend panel.

    The read request also presents the stable AgentScope client id to
    Voicebox, allowing a previously unseen client to appear in Voicebox
    Settings -> MCP without first failing a chat request.
    """
    record = await access.resolve_credential(user_id, credential_id)
    credential = CredentialFactory.from_dict(record.data)
    if not isinstance(credential, VoiceboxCredential):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The selected credential is not a Voicebox credential.",
        )

    client_id = get_voicebox_client_id(user_id, credential_id)
    model = VoiceboxTTSModel(
        credential=credential,
        client_id=client_id,
    )
    try:
        binding = await model.get_client_binding()
    except (ImportError, RuntimeError) as e:
        return VoiceboxClientSetupResponse(
            client_id=client_id,
            endpoint=credential.endpoint,
            reachable=False,
            error=str(e),
        )

    return VoiceboxClientSetupResponse(
        client_id=client_id,
        endpoint=credential.endpoint,
        reachable=True,
        profile_id=binding.get("profile_id") if binding else None,
        default_engine=binding.get("default_engine") if binding else None,
    )
