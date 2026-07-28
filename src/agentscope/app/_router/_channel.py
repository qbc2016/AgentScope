# -*- coding: utf-8 -*-
"""Channel HTTP API.

    GET    /channels/types              List channel types + schemas
    GET    /channels/                   List the user's channels
    POST   /channels/                   Create a channel
    GET    /channels/{id}               Channel details
    PATCH  /channels/{id}               Update routing/session/presentation
    DELETE /channels/{id}               Delete a channel
    POST   /channels/{id}/enable        Enable
    POST   /channels/{id}/disable       Disable
    GET    /channels/{id}/status        Aggregated runtime status
    GET    /channels/{id}/chat_ids      Known chats (for routing config)
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..channel import (
    ChannelLifecycleDispatcher,
    ChannelNotFoundError,
    ChannelService,
    ChannelTypeRegistry,
    DuplicateBotError,
)
from ..deps import get_current_user_id
from ..storage import (
    ChannelRecord,
    ReplyPresentation,
    RoutingConfig,
    SessionSettings,
)

channel_router = APIRouter(prefix="/channels", tags=["channels"])


# -- Schemas --


class CreateChannelRequest(BaseModel):
    """Body for creating a channel."""

    channel_type: str
    credentials: dict
    platform_config: dict = Field(default_factory=dict)
    routing: RoutingConfig
    session: SessionSettings
    presentation: ReplyPresentation = Field(default_factory=ReplyPresentation)
    enabled: bool = True


class UpdateChannelRequest(BaseModel):
    """Body for updating a channel (immutable: type/credentials)."""

    routing: RoutingConfig | None = None
    session: SessionSettings | None = None
    presentation: ReplyPresentation | None = None
    enabled: bool | None = None


class ChannelResponse(BaseModel):
    """Channel details (credentials omitted)."""

    id: str
    channel_type: str
    user_id: str
    platform_bot_id: str
    enabled: bool
    platform_config: dict
    routing: RoutingConfig
    session: SessionSettings
    presentation: ReplyPresentation
    created_at: str
    updated_at: str


# -- Dependencies --


def _service(request: Request) -> ChannelService:
    svc = getattr(request.app.state, "channel_service", None)
    if svc is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Channel feature is not enabled.",
        )
    return svc


def _runtime(request: Request) -> ChannelLifecycleDispatcher:
    rt = getattr(request.app.state, "channel_runtime", None)
    if rt is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Channel feature is not enabled.",
        )
    return rt


def _type_registry(request: Request) -> ChannelTypeRegistry:
    reg = getattr(request.app.state, "channel_type_registry", None)
    if reg is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Channel feature is not enabled.",
        )
    return reg


def _to_response(
    record: ChannelRecord,
    registry: ChannelTypeRegistry,
) -> ChannelResponse:
    try:
        bot_id = registry.extract_platform_bot_id(
            record.channel_type,
            record.credentials,
        )
    except ValueError:
        bot_id = ""
    return ChannelResponse(
        id=record.id,
        channel_type=record.channel_type,
        user_id=record.user_id,
        platform_bot_id=bot_id,
        enabled=record.enabled,
        platform_config=record.platform_config,
        routing=record.routing,
        session=record.session,
        presentation=record.presentation,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


async def _owned(
    channel_id: str,
    user_id: str,
    service: ChannelService,
) -> ChannelRecord:
    record = await service.get(channel_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Channel not found.")
    if record.user_id != user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied.")
    return record


# -- Endpoints --


@channel_router.get("/types")
async def list_channel_types(
    registry: ChannelTypeRegistry = Depends(_type_registry),
) -> list[dict]:
    """List supported channel types with their JSON schemas."""
    return [t.model_dump() for t in registry.list_types()]


@channel_router.get("/")
async def list_channels(
    service: ChannelService = Depends(_service),
    registry: ChannelTypeRegistry = Depends(_type_registry),
    user_id: str = Depends(get_current_user_id),
) -> list[ChannelResponse]:
    """List channels owned by the current user."""
    records = await service.list_for_user(user_id)
    return [_to_response(r, registry) for r in records]


@channel_router.post("/", status_code=status.HTTP_201_CREATED)
async def create_channel(
    body: CreateChannelRequest,
    service: ChannelService = Depends(_service),
    registry: ChannelTypeRegistry = Depends(_type_registry),
    user_id: str = Depends(get_current_user_id),
) -> ChannelResponse:
    """Create a channel."""
    try:
        record = await service.create(
            user_id=user_id,
            channel_type=body.channel_type,
            credentials=body.credentials,
            platform_config=body.platform_config,
            routing=body.routing,
            session=body.session,
            presentation=body.presentation,
            enabled=body.enabled,
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    except DuplicateBotError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    return _to_response(record, registry)


@channel_router.get("/{channel_id}")
async def get_channel(
    channel_id: str,
    service: ChannelService = Depends(_service),
    registry: ChannelTypeRegistry = Depends(_type_registry),
    user_id: str = Depends(get_current_user_id),
) -> ChannelResponse:
    """Get channel details."""
    record = await _owned(channel_id, user_id, service)
    return _to_response(record, registry)


@channel_router.patch("/{channel_id}")
async def update_channel(
    channel_id: str,
    body: UpdateChannelRequest,
    service: ChannelService = Depends(_service),
    registry: ChannelTypeRegistry = Depends(_type_registry),
    user_id: str = Depends(get_current_user_id),
) -> ChannelResponse:
    """Update routing / session / presentation / enabled."""
    await _owned(channel_id, user_id, service)
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No fields to update.",
        )
    try:
        record = await service.update(channel_id, updates)
    except ChannelNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return _to_response(record, registry)


@channel_router.delete(
    "/{channel_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_channel(
    channel_id: str,
    service: ChannelService = Depends(_service),
    user_id: str = Depends(get_current_user_id),
) -> None:
    """Delete a channel."""
    await _owned(channel_id, user_id, service)
    await service.delete(channel_id)


@channel_router.post("/{channel_id}/enable")
async def enable_channel(
    channel_id: str,
    service: ChannelService = Depends(_service),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """Enable a channel."""
    await _owned(channel_id, user_id, service)
    await service.set_enabled(channel_id, True)
    return {"status": "enabled"}


@channel_router.post("/{channel_id}/disable")
async def disable_channel(
    channel_id: str,
    service: ChannelService = Depends(_service),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """Disable a channel."""
    await _owned(channel_id, user_id, service)
    await service.set_enabled(channel_id, False)
    return {"status": "disabled"}


@channel_router.get("/{channel_id}/status")
async def channel_status(
    channel_id: str,
    service: ChannelService = Depends(_service),
    runtime: ChannelLifecycleDispatcher = Depends(_runtime),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """Aggregated multi-node runtime status."""
    await _owned(channel_id, user_id, service)
    return await runtime.get_status(channel_id)


@channel_router.get("/{channel_id}/chat_ids")
async def list_chat_ids(
    channel_id: str,
    service: ChannelService = Depends(_service),
    runtime: ChannelLifecycleDispatcher = Depends(_runtime),
    user_id: str = Depends(get_current_user_id),
) -> list[dict]:
    """Known chats for routing config: platform list ∪ passively seen."""
    await _owned(channel_id, user_id, service)
    results: list[dict] = []
    platform_ids: set[str] = set()
    for chat in await runtime.list_bot_chats(channel_id):
        cid = chat.get("chat_id", "")
        if cid:
            platform_ids.add(cid)
            results.append(
                {
                    "chat_id": cid,
                    "name": chat.get("name", ""),
                    "source": "platform",
                },
            )
    for cid in await runtime.list_seen_chat_ids(channel_id):
        if cid not in platform_ids:
            results.append({"chat_id": cid, "name": "", "source": "recorded"})
    return results
