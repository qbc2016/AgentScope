# -*- coding: utf-8 -*-
"""Feishu (Lark) channel adapter — new ChannelBase interface.

Translates the Feishu platform to/from normalised events and emits them
via the injected gateway callback. Confirmation is two-phase: a card
click is emitted as a ``ConfirmDecisionEvent`` (same entry as messages);
the gateway later resolves the card via :meth:`update_confirm`. No
in-process approval futures or attachment buffers — media aggregation
and pending state live in the gateway / shared storage.

The WebSocket runs in a background thread (the lark SDK owns its own
event loop); inbound events are bridged to the app loop with
``run_coroutine_threadsafe``. Connection/reconnect is driven via the
SDK's public ``start()`` — the one place to adapt if the SDK changes.
"""
import asyncio
import base64
import json
import threading
from typing import Any

from ...._logging import logger
from ....message import Base64Source, DataBlock, TextBlock
from .._base import (
    ChannelBase,
    ChannelCapability,
    ChannelEvent,
    ConfirmDecisionEvent,
    ConfirmPrompt,
)
from ._card_templates import (
    build_approval_card,
    build_resolved_card,
    build_toast,
    parse_action,
)

_API = "https://open.feishu.cn/open-apis"
_TOKEN_EXPIRED_CODES = frozenset({99991663, 99991664})
_MEDIA_TYPES = frozenset({"image", "audio", "media", "file"})


class FeishuChannel(ChannelBase):
    """Feishu platform adapter (SDK long-connection mode)."""

    capabilities = ChannelCapability(
        text=True,
        markdown=True,
        image=True,
        interactive=True,
        max_message_length=4000,
    )

    def __init__(
        self,
        channel_id: str,
        app_id: str,
        app_secret: str,
        *,
        only_at_reply: bool = True,
    ) -> None:
        self._channel_id = channel_id
        self._app_id = app_id
        self._app_secret = app_secret
        self._only_at_reply = only_at_reply
        self._http: Any = None
        self._token: str | None = None
        self._ws_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def channel_id(self) -> str:
        return self._channel_id

    # -- Lifecycle --

    async def on_start(self) -> None:
        import httpx

        self._http = httpx.AsyncClient(timeout=30.0)
        await self._refresh_token()

    async def on_stop(self) -> None:
        self._stop.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)
        if self._http:
            await self._http.aclose()
            self._http = None

    async def start_listening(self) -> None:
        """Run the WS client, reconnecting with backoff if it exits."""
        self._loop = asyncio.get_running_loop()
        backoff = 1.0
        while not self._stop.is_set():
            self._ws_thread = self._launch_ws_thread()
            uptime = 0.0
            while not self._stop.is_set() and self._ws_thread.is_alive():
                await asyncio.sleep(5.0)
                uptime += 5.0
            if self._stop.is_set():
                break
            backoff = 1.0 if uptime >= 60.0 else min(backoff * 2, 30.0)
            logger.warning(
                "Feishu WS '%s' exited, reconnecting in %.1fs",
                self._channel_id,
                backoff,
            )
            await asyncio.sleep(backoff)

    def _launch_ws_thread(self) -> threading.Thread:
        try:
            import lark_oapi as lark
        except ImportError as e:
            raise ImportError(
                "Feishu channel requires 'lark-oapi' "
                "(pip install lark-oapi).",
            ) from e

        loop = self._loop
        assert loop is not None  # set in start_listening before this runs

        def on_message(data: Any) -> None:
            asyncio.run_coroutine_threadsafe(self._on_message(data), loop)

        def on_card_action(data: Any) -> Any:
            return self._on_card_action(data, loop)

        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(on_message)
            .register_p2_card_action_trigger(on_card_action)
            .build()
        )
        client = lark.ws.Client(
            self._app_id,
            self._app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )

        def run() -> None:
            try:
                client.start()  # public entry; owns its own event loop
            except Exception:  # pylint: disable=broad-except
                if not self._stop.is_set():
                    logger.exception(
                        "Feishu WS '%s' crashed",
                        self._channel_id,
                    )

        thread = threading.Thread(
            target=run,
            name=f"feishu-ws:{self._channel_id}",
            daemon=True,
        )
        thread.start()
        return thread

    # -- Inbound (WS thread → app loop) --

    async def _on_message(self, data: Any) -> None:
        """Normalise an inbound message and emit it (media or text)."""
        try:
            event = await self._normalize(data)
            if event and self._emit:
                await self._emit(event)
        except Exception:  # pylint: disable=broad-except
            logger.exception(
                "Feishu '%s' message handling failed",
                self._channel_id,
            )

    async def _normalize(self, data: Any) -> ChannelEvent | None:
        message = getattr(data.event, "message", None)
        sender = getattr(data.event, "sender", None)
        if message is None or sender is None:
            return None

        user_id = ""
        if sender.sender_id:
            user_id = sender.sender_id.open_id or ""
        chat_id = message.chat_id or ""
        chat_type = message.chat_type or ""
        message_id = message.message_id or ""
        meta = {
            "chat_type": chat_type,
            "tenant_key": data.header.tenant_key if data.header else "",
        }

        msg_type = message.message_type
        if msg_type in _MEDIA_TYPES:
            block = await self._download_media(message, msg_type)
            if block is None:
                return None
            return ChannelEvent(
                channel_id=self._channel_id,
                channel_user_id=user_id,
                chat_id=chat_id,
                channel_message_id=message_id,
                content=[block],
                metadata=meta,
            )
        if msg_type != "text":
            await self._reply(message_id, chat_id, f"暂不支持 {msg_type} 消息。")
            return None

        content = json.loads(message.content or "{}")
        text = (content.get("text") or "").strip()
        if chat_type == "group" and self._only_at_reply:
            mentions = message.mentions or []
            if not mentions and "@_user_" not in (message.content or ""):
                return None
            for mention in mentions:
                text = text.replace(mention.key or "", "").strip()
        if not text:
            return None

        return ChannelEvent(
            channel_id=self._channel_id,
            channel_user_id=user_id,
            chat_id=chat_id,
            channel_message_id=message_id,
            content=[TextBlock(text=text)],
            metadata=meta,
        )

    def _on_card_action(self, data: Any, loop: Any) -> Any:
        """Emit the click as a ConfirmDecisionEvent; ack with a toast."""
        action = getattr(getattr(data.event, "action", None), "value", None)
        parsed = parse_action(action)
        if parsed is None:
            return build_toast(False)
        request_id, approved = parsed
        if self._emit:
            asyncio.run_coroutine_threadsafe(
                self._emit(
                    ConfirmDecisionEvent(
                        channel_id=self._channel_id,
                        request_id=request_id,
                        approved=approved,
                    ),
                ),
                loop,
            )
        return build_toast(approved)

    # -- Outbound (gateway → platform) --

    async def send_response(
        self,
        event: ChannelEvent,
        content: list[TextBlock | DataBlock],
    ) -> None:
        text = "".join(b.text for b in content if isinstance(b, TextBlock))
        if not text:
            return
        for part in self._split_long_message(text):
            await self._send_text(
                event.channel_message_id,
                event.chat_id,
                part,
            )

    async def present_confirm(
        self,
        event: ChannelEvent,
        prompt: ConfirmPrompt,
    ) -> str | None:
        card = build_approval_card(
            prompt.request_id,
            prompt.tool_name,
            prompt.summary,
        )
        return await self._send_card(
            event.channel_message_id,
            event.chat_id,
            card,
        )

    async def update_confirm(self, ref: str, outcome: str) -> None:
        await self._api(
            "PATCH",
            f"{_API}/im/v1/messages/{ref}",
            {
                "msg_type": "interactive",
                "content": build_resolved_card(outcome),
            },
        )

    async def add_reaction(
        self,
        event: ChannelEvent,
        emoji_type: str,
    ) -> str | None:
        if not event.channel_message_id:
            return None
        data = await self._api(
            "POST",
            f"{_API}/im/v1/messages/{event.channel_message_id}/reactions",
            {"reaction_type": {"emoji_type": emoji_type}},
        )
        if data and data.get("code") == 0:
            return data.get("data", {}).get("reaction_id")
        return None

    async def remove_reaction(
        self,
        event: ChannelEvent,
        reaction_id: str,
    ) -> None:
        if not event.channel_message_id:
            return
        await self._api(
            "DELETE",
            f"{_API}/im/v1/messages/{event.channel_message_id}"
            f"/reactions/{reaction_id}",
        )

    async def list_bot_chats(self) -> list[dict]:
        results: list[dict] = []
        page_token = ""
        while True:
            url = f"{_API}/im/v1/chats?page_size=50"
            if page_token:
                url += f"&page_token={page_token}"
            data = await self._api("GET", url)
            if not data or data.get("code") != 0:
                break
            payload = data.get("data", {})
            for item in payload.get("items", []):
                results.append(
                    {
                        "chat_id": item.get("chat_id", ""),
                        "name": item.get("name", ""),
                        "chat_type": item.get("chat_type", ""),
                    },
                )
            if not payload.get("has_more"):
                break
            page_token = payload.get("page_token", "")
        return results

    # -- Feishu API helpers --

    async def _refresh_token(self) -> None:
        resp = await self._http.post(
            f"{_API}/auth/v3/tenant_access_token/internal",
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        data = resp.json()
        if data.get("code") == 0:
            self._token = data.get("tenant_access_token")
        else:
            logger.error("Feishu token refresh failed: %s", data)

    async def _api(
        self,
        method: str,
        url: str,
        body: dict | None = None,
        *,
        _retried: bool = False,
    ) -> dict | None:
        """Authenticated Feishu request; refreshes the token once on expiry."""
        if not self._http or not self._token:
            return None
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        try:
            resp = await self._http.request(
                method,
                url,
                headers=headers,
                json=body,
            )
            data = resp.json()
            if data.get("code") == 0:
                return data
            if not _retried and data.get("code") in _TOKEN_EXPIRED_CODES:
                await self._refresh_token()
                return await self._api(method, url, body, _retried=True)
            logger.warning("Feishu API %s failed: %s", method, data.get("msg"))
            return data
        except Exception:  # pylint: disable=broad-except
            logger.debug("Feishu API %s request failed", method)
            return None

    async def _send_text(
        self,
        reply_to: str | None,
        chat_id: str,
        text: str,
    ) -> None:
        await self._send(reply_to, chat_id, "text", json.dumps({"text": text}))

    async def _send_card(
        self,
        reply_to: str | None,
        chat_id: str,
        card: str,
    ) -> str | None:
        data = await self._send(reply_to, chat_id, "interactive", card)
        if data and data.get("code") == 0:
            return data.get("data", {}).get("message_id")
        return None

    async def _send(
        self,
        reply_to: str | None,
        chat_id: str,
        msg_type: str,
        content: str,
    ) -> dict | None:
        """Send a message — as a reply when possible, else to the chat."""
        if reply_to:
            return await self._api(
                "POST",
                f"{_API}/im/v1/messages/{reply_to}/reply",
                {"msg_type": msg_type, "content": content},
            )
        return await self._api(
            "POST",
            f"{_API}/im/v1/messages?receive_id_type=chat_id",
            {"receive_id": chat_id, "msg_type": msg_type, "content": content},
        )

    async def _reply(
        self,
        message_id: str,
        chat_id: str,
        text: str,
    ) -> None:
        if message_id or chat_id:
            await self._send_text(message_id, chat_id, text)

    async def _download_media(
        self,
        message: Any,
        msg_type: str,
    ) -> DataBlock | None:
        """Download a media resource into a base64 ``DataBlock``."""
        content = json.loads(getattr(message, "content", None) or "{}")
        key = content.get("image_key") or content.get("file_key") or ""
        if not key:
            return None
        resource_type = "image" if msg_type == "image" else "file"
        default_mime = {
            "image": "image/png",
            "audio": "audio/ogg",
            "media": "video/mp4",
        }.get(msg_type, "application/octet-stream")
        url = (
            f"{_API}/im/v1/messages/{message.message_id}"
            f"/resources/{key}?type={resource_type}"
        )
        try:
            resp = await self._http.get(
                url,
                headers={"Authorization": f"Bearer {self._token}"},
            )
            if resp.status_code != 200:
                return None
            return DataBlock(
                source=Base64Source(
                    data=base64.b64encode(resp.content).decode("ascii"),
                    media_type=resp.headers.get("content-type", default_mime),
                ),
                name=msg_type,
            )
        except Exception:  # pylint: disable=broad-except
            logger.debug("Feishu media download failed")
            return None
