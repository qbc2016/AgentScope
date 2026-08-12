# -*- coding: utf-8 -*-
"""Tests for shared channel credential binding behavior."""

import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from agentscope.app._router._channel import (
    cancel_credential_binding,
    channel_router,
)
from agentscope.app._service._channel import ChannelService
from agentscope.app.channel import (
    ChannelError,
    ChannelCredentialBindingState,
    ChannelCredentialBindingStore,
    ChannelTypeRegistry,
    FeishuChannel,
)
from agentscope.app.channel._credential_binding import (
    ChannelCredentialBindingRecord,
)
from agentscope.app.channel._feishu._credential_binding import (
    FeishuCredentialBinding,
)
from agentscope.app.message_bus import InMemoryMessageBus
from agentscope.app.storage import (
    ChannelBinding,
    RoutingConfig,
    SessionSettings,
)


class _PausingMessageBus(InMemoryMessageBus):
    """Pause one registry write to make lock ordering deterministic."""

    def __init__(self) -> None:
        super().__init__()
        self.pause_next_write = False
        self.write_started = asyncio.Event()
        self.resume_write = asyncio.Event()

    async def registry_set(
        self,
        namespace: str,
        field: str,
        value: str,
        *,
        ttl_secs: int | None = None,
    ) -> None:
        if self.pause_next_write:
            self.pause_next_write = False
            self.write_started.set()
            await self.resume_write.wait()
        await super().registry_set(
            namespace,
            field,
            value,
            ttl_secs=ttl_secs,
        )


class _FakeRegistrationSdk:
    """Return a QR code followed by authorized Feishu credentials."""

    def __init__(self) -> None:
        self.arguments: dict = {}

    async def aregister_app(self, **kwargs: object) -> dict[str, str]:
        """Emit one QR callback and return deterministic credentials."""
        self.arguments = kwargs
        on_qr_code = kwargs["on_qr_code"]
        assert callable(on_qr_code)
        on_qr_code(
            {
                "url": "https://accounts.feishu.cn/device",
                "expire_in": 300,
            },
        )
        await asyncio.sleep(0)
        return {
            "client_id": "cli_test",
            "client_secret": "secret_test",
        }


class _ChannelStorage:
    """Store only the channel methods exercised by ChannelService."""

    def __init__(self) -> None:
        self.channels: dict[str, object] = {}
        self.bot_ids: dict[str, str] = {}

    async def get_channel_id_by_platform_bot_id(
        self,
        platform_bot_id: str,
    ) -> str | None:
        """Return the channel currently indexed by a platform bot id."""
        return self.bot_ids.get(platform_bot_id)

    async def upsert_channel(
        self,
        record: object,
        platform_bot_id: str,
    ) -> str:
        """Persist a channel record and its bot-id index."""
        channel_id = str(getattr(record, "id"))
        self.channels[channel_id] = record
        self.bot_ids[platform_bot_id] = channel_id
        return channel_id


class _ServiceBinding:
    """Return the same credentials to concurrent create requests."""

    def __init__(self) -> None:
        self.completed = 0

    async def resolve_credentials(self, *_: object) -> dict[str, str]:
        """Return one authorized credential pair."""
        return {
            "app_id": "cli_test",
            "app_secret": "secret_test",
        }

    async def complete(self, *_: object) -> None:
        """Count successful binding consumption."""
        self.completed += 1


class _ServiceRegistry:
    """Provide the credential operations needed by ChannelService."""

    def __init__(self, binding: _ServiceBinding) -> None:
        self.binding = binding

    def get_credential_binding(self, _: str) -> _ServiceBinding:
        """Return the configured fake binding provider."""
        return self.binding

    def validate_credentials(
        self,
        _: str,
        credentials: dict,
    ) -> dict:
        """Return already-valid fake credentials."""
        return credentials

    def extract_platform_bot_id(
        self,
        _: str,
        credentials: dict,
    ) -> str:
        """Use the fake app id as the unique platform bot id."""
        return str(credentials["app_id"])


class TestChannelCredentialBindingStore(IsolatedAsyncioTestCase):
    """Exercise the minimal shared record and binding lock."""

    async def test_delete_wins_after_in_flight_replace(self) -> None:
        """A cancellation must not be undone by a background write."""
        bus = _PausingMessageBus()
        store = ChannelCredentialBindingStore(bus)
        record = ChannelCredentialBindingRecord(
            id="binding",
            user_id="user",
            provider_id="feishu",
            state=ChannelCredentialBindingState.PENDING,
            expires_at="2099-01-01T00:00:00+00:00",
        )
        await store.create(record, 60)

        authorized = record.model_copy(
            update={
                "state": ChannelCredentialBindingState.AUTHORIZED,
                "credentials": {
                    "app_id": "cli_test",
                    "app_secret": "secret_test",
                },
            },
        )
        bus.pause_next_write = True
        replace_task = asyncio.create_task(
            store.replace(
                authorized,
                60,
                {ChannelCredentialBindingState.PENDING},
            ),
        )
        await bus.write_started.wait()
        delete_task = asyncio.create_task(
            store.delete("user", "binding", "feishu"),
        )
        await asyncio.sleep(0)
        self.assertFalse(delete_task.done())

        bus.resume_write.set()
        self.assertTrue(await replace_task)
        self.assertTrue(await delete_task)
        self.assertIsNone(await store.get("binding"))
        await bus.aclose()


class TestFeishuCredentialBinding(IsolatedAsyncioTestCase):
    """Exercise QR persistence and credential consumption."""

    async def test_start_persists_only_after_qr_and_authorizes(self) -> None:
        """Persist the first record with its QR and then authorize it."""
        bus = InMemoryMessageBus()
        store = ChannelCredentialBindingStore(bus)
        binding = FeishuCredentialBinding()
        sdk = _FakeRegistrationSdk()

        with (
            patch.object(binding, "_sdk", return_value=sdk),
            patch(
                "agentscope.app.channel._feishu._credential_binding."
                "_verification_url_to_qr_data_url",
                return_value="data:image/svg+xml,test",
            ),
        ):
            session = await binding.start("user", store)
            status = await binding.get_status("user", session.id, store)

        self.assertEqual(session.qr_code_url, "data:image/svg+xml,test")
        self.assertEqual(
            status.state,
            ChannelCredentialBindingState.AUTHORIZED,
        )
        self.assertNotIn("on_status_change", sdk.arguments)
        self.assertEqual(
            await binding.resolve_credentials("user", session.id, store),
            {
                "app_id": "cli_test",
                "app_secret": "secret_test",
            },
        )

        await binding.complete("user", session.id, store)
        self.assertIsNone(await store.get(session.id))
        await binding.aclose()
        await bus.aclose()


class TestCredentialBindingConsumption(IsolatedAsyncioTestCase):
    """Exercise bot-lock serialization without a consume lock."""

    async def test_concurrent_consumers_create_only_one_channel(self) -> None:
        """Let the bot lock reject the second concurrent consumer."""
        storage = _ChannelStorage()
        bus = InMemoryMessageBus()
        binding = _ServiceBinding()
        registry = _ServiceRegistry(binding)
        service = ChannelService(  # type: ignore[arg-type]
            storage,
            bus,
            registry,
        )
        routing = RoutingConfig(
            bindings=[ChannelBinding(agent_id="agent")],
        )
        session = SessionSettings(chat_model_config={"model": "test"})

        async def create() -> object:
            return await service.create(
                user_id="user",
                channel_type="feishu",
                credentials=None,
                credential_binding_id="binding",
                platform_config={},
                routing=routing,
                session=session,
                name="channel",
            )

        results = await asyncio.gather(
            create(),
            create(),
            return_exceptions=True,
        )
        failures = [item for item in results if isinstance(item, Exception)]
        self.assertEqual(len(storage.channels), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], ChannelError)
        self.assertEqual(binding.completed, 1)
        await bus.aclose()


class TestCredentialBindingRouting(IsolatedAsyncioTestCase):
    """Check provider dispatch and the simplified HTTP contract."""

    async def test_provider_is_indexed_by_stored_provider_id(self) -> None:
        """Resolve the provider id carried by a shared binding record."""
        registry = ChannelTypeRegistry([FeishuChannel])
        provider = registry.get_credential_binding("feishu")
        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertIs(
            registry.get_credential_binding_by_provider_id(
                provider.provider_id,
            ),
            provider,
        )

    async def test_status_and_cancel_routes_only_use_binding_id(self) -> None:
        """Expose only opaque binding ids in poll and cancel routes."""
        paths = {route.path for route in channel_router.routes}
        self.assertIn("/channels/bindings/{binding_id}", paths)
        self.assertNotIn(
            "/channels/bindings/{channel_type}/{binding_id}",
            paths,
        )

    async def test_cancel_missing_binding_is_idempotent(self) -> None:
        """Treat repeated cancellation as a successful no-op."""
        bus = InMemoryMessageBus()
        registry = ChannelTypeRegistry([FeishuChannel])

        await cancel_credential_binding(
            "missing",
            registry,
            bus,
            "user",
        )

        await registry.close_credential_bindings()
        await bus.aclose()
