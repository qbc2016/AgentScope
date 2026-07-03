# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Feishu (Lark) channel adapter using the official lark-oapi SDK.

Uses lark_oapi.ws.Client for WebSocket long-connection mode. This is the
recommended approach as it handles authentication, reconnection, keep-alive,
and the internal Protobuf frame protocol automatically.
"""
import asyncio
import json
import threading
from typing import Any

from ...._logging import logger
from .._base import ChannelBase, ChannelCapability, ChannelEvent

_WS_MODULE_LOCK = threading.Lock()


class FeishuChannel(ChannelBase):
    """Feishu platform adapter using the official SDK long-connection mode."""

    capabilities = ChannelCapability(
        text=True,
        markdown=True,
        image=True,
        file=False,
        streaming=False,
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
        self._http_client: Any = None
        self._tenant_access_token: str | None = None
        self._ws_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._pending_approvals: dict[str, asyncio.Future] = {}

    @property
    def channel_id(self) -> str:
        return self._channel_id

    async def on_start(self) -> None:
        """Initialise HTTP client for sending messages."""
        import httpx

        self._http_client = httpx.AsyncClient(timeout=30.0)
        await self._refresh_token()

    async def on_stop(self) -> None:
        """Stop WebSocket client and close HTTP client."""
        self._stop_event.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def start_listening(self) -> None:
        """Start the official Feishu WebSocket client in a background thread.

        The lark-oapi SDK's ws.Client runs its own asyncio event loop
        internally, so we run it in a separate thread and bridge events
        back to our main loop. Implements automatic reconnection with
        exponential backoff when the thread dies unexpectedly.
        """
        self._main_loop = asyncio.get_running_loop()
        backoff = 1.0
        token_refresh_interval = 1800  # 30 minutes
        while not self._stop_event.is_set():
            self._ws_thread = self._launch_ws_thread()
            logger.info(
                "Feishu channel '%s' WebSocket client started (thread: %s).",
                self._channel_id,
                self._ws_thread.name,
            )
            backoff = 1.0
            elapsed_since_refresh = 0.0

            while not self._stop_event.is_set():
                if not self._ws_thread.is_alive():
                    break
                await asyncio.sleep(5.0)
                elapsed_since_refresh += 5.0
                if elapsed_since_refresh >= token_refresh_interval:
                    elapsed_since_refresh = 0.0
                    try:
                        await self._refresh_token()
                    except Exception:
                        pass

            if self._stop_event.is_set():
                break

            logger.warning(
                "Feishu WS thread for '%s' died, reconnecting in %.1fs.",
                self._channel_id,
                backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _launch_ws_thread(self) -> threading.Thread:
        """Create and start a new WebSocket thread.

        Uses asyncio.run() to create a completely isolated event loop
        context. This avoids the "Future attached to a different loop"
        issues that occur when manually overriding the SDK's module-level
        loop variable with websockets >= 13.
        """
        try:
            import lark_oapi as lark

        except ImportError as e:
            raise ImportError(
                "Feishu channel requires 'lark-oapi'. "
                "Install it with: pip install lark-oapi",
            ) from e

        main_loop = self._main_loop or asyncio.get_running_loop()
        app_id = self._app_id
        app_secret = self._app_secret
        channel_id = self._channel_id
        stop_event = self._stop_event
        gateway = self._gateway
        normalize = self._normalize_sdk_event
        handle_card_action = self._handle_card_action_from_thread

        def run_ws() -> None:
            import lark_oapi.ws.client as ws_module

            ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ws_loop)

            def on_event(
                data: "lark.im.v1.P2ImMessageReceiveV1",
            ) -> None:
                try:
                    event = normalize(data)
                    if event and gateway:
                        asyncio.run_coroutine_threadsafe(
                            gateway.handle_event(event),
                            main_loop,
                        )
                except Exception:
                    logger.exception(
                        "Error normalizing Feishu event in '%s'",
                        channel_id,
                    )

            def on_card_action(data: Any) -> Any:
                """Handle card.action.trigger callback."""
                try:
                    return handle_card_action(data, main_loop)
                except Exception:
                    logger.exception(
                        "Error handling card action in '%s'",
                        channel_id,
                    )

                    from lark_oapi.event.callback.model.p2_card_action_trigger import (  # noqa
                        P2CardActionTriggerResponse,
                    )

                    return P2CardActionTriggerResponse({})

            event_handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(on_event)
                .register_p2_card_action_trigger(on_card_action)
                .build()
            )

            ws_client = lark.ws.Client(
                app_id,
                app_secret,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO,
            )

            async def _drive() -> None:
                # Patch module-level loop right before _connect() to
                # ensure _receive_message_loop is scheduled on ws_loop.
                # Use lock to prevent race with other FeishuChannel threads.
                with _WS_MODULE_LOCK:
                    ws_module.loop = ws_loop
                    await ws_client._connect()
                ws_loop.create_task(ws_client._ping_loop())
                while not stop_event.is_set():
                    await asyncio.sleep(1.0)
                await ws_client._disconnect()

            try:
                ws_loop.run_until_complete(_drive())
            except Exception:
                if not stop_event.is_set():
                    logger.exception(
                        "Feishu WS client for '%s' exited unexpectedly",
                        channel_id,
                    )
            finally:
                try:
                    ws_loop.close()
                except Exception:
                    pass

        thread = threading.Thread(
            target=run_ws,
            name=f"feishu-ws:{self._channel_id}",
            daemon=True,
        )
        thread.start()
        return thread

    def _normalize_sdk_event(
        self,
        data: Any,
    ) -> ChannelEvent | None:
        """Convert SDK event object to ChannelEvent."""
        event = data.event
        if event is None:
            return None

        message = event.message
        sender = event.sender

        if message is None or sender is None:
            return None

        msg_type = message.message_type
        if msg_type != "text":
            return None

        content_str = message.content or "{}"
        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            content = {}

        text = content.get("text", "").strip()
        if not text:
            return None

        chat_id = message.chat_id or ""
        chat_type = message.chat_type or ""
        channel_user_id = ""
        if sender.sender_id:
            channel_user_id = sender.sender_id.open_id or ""
        message_id = message.message_id or ""

        # Group chat: check if bot is mentioned
        if chat_type == "group" and self._only_at_reply:
            mentions = message.mentions or []
            if not mentions:
                if "@_user_" not in content_str:
                    return None
            for mention in mentions:
                key = mention.key or ""
                if key:
                    text = text.replace(key, "").strip()

        if not text:
            return None

        return ChannelEvent(
            channel_id=self._channel_id,
            channel_user_id=channel_user_id,
            channel_message_id=message_id,
            message=text,
            metadata={
                "chat_id": chat_id,
                "chat_type": chat_type,
                "tenant_key": data.header.tenant_key if data.header else "",
            },
        )

    async def normalize(self, raw_payload: dict) -> ChannelEvent | None:
        """Not used in SDK mode — events come via the SDK callback."""
        return None

    async def send_response(
        self,
        event: ChannelEvent,
        response: str,
    ) -> None:
        """Send reply message back to Feishu."""
        if not self._http_client or not self._tenant_access_token:
            logger.error("Feishu channel not initialised, cannot send.")
            return

        parts = self._split_long_message(response)
        for part in parts:
            await self._send_message(event, part)

    def build_approval_card(
        self,
        *,
        request_id: str,
        tool_name: str,
        tool_input_summary: str = "",
        session_id: str = "",
        agent_id: str = "",
        user_id: str = "",
    ) -> str:
        """Build a Feishu interactive card for tool approval."""
        from ._card_templates import build_approval_card

        return build_approval_card(
            request_id=request_id,
            tool_name=tool_name,
            tool_input_summary=tool_input_summary,
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
        )

    def build_resolved_card(
        self,
        *,
        tool_name: str,
        action: str,
    ) -> str:
        """Build a Feishu resolved card."""
        from ._card_templates import build_resolved_card

        return build_resolved_card(tool_name=tool_name, action=action)

    async def send_interactive_card(
        self,
        event: ChannelEvent,
        card_content: str,
    ) -> str | None:
        """Send an interactive card message to Feishu."""
        if not self._http_client or not self._tenant_access_token:
            return None

        message_id = event.channel_message_id
        if message_id:
            url = (
                "https://open.feishu.cn/open-apis"
                f"/im/v1/messages/{message_id}/reply"
            )
            body: dict = {
                "msg_type": "interactive",
                "content": card_content,
            }
        else:
            chat_id = event.metadata.get("chat_id", "")
            url = (
                "https://open.feishu.cn/open-apis"
                "/im/v1/messages?receive_id_type=chat_id"
            )
            body = {
                "receive_id": chat_id,
                "msg_type": "interactive",
                "content": card_content,
            }

        try:
            resp = await self._http_client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._tenant_access_token}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            data = resp.json()
            if data.get("code") == 0:
                return data.get("data", {}).get("message_id")
            logger.warning(
                "Feishu send_interactive_card failed: %s",
                data.get("msg"),
            )
        except Exception:
            logger.exception("Failed to send interactive card to Feishu.")
        return None

    async def update_card(
        self,
        message_id: str,
        card_content: str,
    ) -> None:
        """Update an existing interactive card message."""
        if not self._http_client or not self._tenant_access_token:
            return

        url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}"
        try:
            await self._http_client.patch(
                url,
                headers={
                    "Authorization": f"Bearer {self._tenant_access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "msg_type": "interactive",
                    "content": card_content,
                },
            )
        except Exception:
            logger.debug("Failed to update Feishu card %s.", message_id)

    # ── Approval mechanism ──

    def register_approval(self, request_id: str) -> asyncio.Future:
        """Register a pending approval and return the Future to await."""
        loop = self._main_loop or asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_approvals[request_id] = fut
        return fut

    def resolve_approval(self, request_id: str, approved: bool) -> bool:
        """Resolve a pending approval future. Returns True if found."""
        fut = self._pending_approvals.pop(request_id, None)
        if fut and not fut.done():
            fut.set_result(approved)
            return True
        return False

    def _handle_card_action_from_thread(
        self,
        data: Any,
        main_loop: asyncio.AbstractEventLoop,
    ) -> Any:
        """Process card.action.trigger callback from the WS thread.

        ``data`` is a ``P2CardActionTrigger`` instance. The action value
        is at ``data.event.action.value``.
        Must return a ``P2CardActionTriggerResponse`` synchronously.
        """
        from lark_oapi.event.callback.model.p2_card_action_trigger import (
            P2CardActionTriggerResponse,
        )
        from ._card_templates import (
            parse_action_value,
            build_resolved_card,
            build_toast_response,
            ACTION_TYPE,
        )

        event_obj = getattr(data, "event", None)
        action_obj = getattr(event_obj, "action", None) if event_obj else None
        action_value = (
            getattr(action_obj, "value", None) if action_obj else None
        )

        if isinstance(action_value, str):
            try:
                action_value = json.loads(action_value)
            except (json.JSONDecodeError, TypeError):
                return P2CardActionTriggerResponse({})

        if not isinstance(action_value, dict):
            return P2CardActionTriggerResponse({})
        if action_value.get("type") != ACTION_TYPE:
            return P2CardActionTriggerResponse({})

        parsed = parse_action_value(action_value)
        if not parsed:
            return P2CardActionTriggerResponse({})

        user_action = parsed["action"]
        request_id = parsed["request_id"]
        tool_name = parsed["tool_name"]
        approved = user_action == "approve"

        # Resolve the pending Future in the main loop
        asyncio.run_coroutine_threadsafe(
            self._resolve_approval_async(request_id, approved),
            main_loop,
        )

        # Build response: update card + toast
        resolved_card = build_resolved_card(
            tool_name=tool_name,
            action=user_action,
        )
        return build_toast_response(
            action=user_action,
            tool_name=tool_name,
            card_json=resolved_card,
        )

    async def _resolve_approval_async(
        self,
        request_id: str,
        approved: bool,
    ) -> None:
        """Resolve approval from async context."""
        self.resolve_approval(request_id, approved)

    async def add_reaction(
        self,
        event: ChannelEvent,
        emoji_type: str,
    ) -> str | None:
        """Add an emoji reaction to a Feishu message."""
        message_id = event.channel_message_id
        if (
            not message_id
            or not self._http_client
            or not self._tenant_access_token
        ):
            return None

        url = (
            "https://open.feishu.cn/open-apis"
            f"/im/v1/messages/{message_id}/reactions"
        )
        try:
            resp = await self._http_client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._tenant_access_token}",
                    "Content-Type": "application/json",
                },
                json={"reaction_type": {"emoji_type": emoji_type}},
            )
            data = resp.json()
            if data.get("code") == 0:
                return data.get("data", {}).get("reaction_id")
            logger.debug(
                "Feishu add_reaction failed: %s",
                data.get("msg"),
            )
        except Exception:
            logger.debug("Failed to add reaction to Feishu message.")
        return None

    async def remove_reaction(
        self,
        event: ChannelEvent,
        reaction_id: str,
    ) -> None:
        """Remove an emoji reaction from a Feishu message."""
        message_id = event.channel_message_id
        if (
            not message_id
            or not self._http_client
            or not self._tenant_access_token
        ):
            return

        url = (
            f"https://open.feishu.cn/open-apis/im/v1/messages"
            f"/{message_id}/reactions/{reaction_id}"
        )
        try:
            await self._http_client.delete(
                url,
                headers={
                    "Authorization": f"Bearer {self._tenant_access_token}",
                },
            )
        except Exception:
            logger.debug("Failed to remove reaction from Feishu message.")

    # ── Internal helpers ──

    async def _refresh_token(self) -> None:
        """Get tenant_access_token from Feishu API."""
        url = (
            "https://open.feishu.cn/open-apis"
            "/auth/v3/tenant_access_token/internal"
        )
        resp = await self._http_client.post(
            url,
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        data = resp.json()
        if data.get("code") == 0:
            self._tenant_access_token = data.get("tenant_access_token")
        else:
            logger.error("Failed to get Feishu token: %s", data)

    async def _send_message(
        self,
        event: ChannelEvent,
        text: str,
        *,
        _retried: bool = False,
    ) -> None:
        """Send a text message via Feishu API (reply to original).

        Automatically refreshes the token and retries once on auth failure.
        """
        message_id = event.channel_message_id

        if message_id:
            url = (
                "https://open.feishu.cn/open-apis"
                f"/im/v1/messages/{message_id}/reply"
            )
            body: dict = {
                "msg_type": "text",
                "content": json.dumps({"text": text}),
            }
        else:
            chat_id = event.metadata.get("chat_id", "")
            url = (
                "https://open.feishu.cn/open-apis"
                "/im/v1/messages?receive_id_type=chat_id"
            )
            body = {
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}),
            }

        try:
            resp = await self._http_client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._tenant_access_token}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            data = resp.json()
            code = data.get("code", 0)
            if code == 0:
                return
            # Token expired/invalid — refresh and retry once
            if not _retried and code in (99991663, 99991664):
                await self._refresh_token()
                await self._send_message(event, text, _retried=True)
                return
            logger.warning(
                "Feishu send_message failed (code=%s): %s",
                code,
                data.get("msg"),
            )
        except Exception:
            logger.exception("Failed to send message to Feishu.")
