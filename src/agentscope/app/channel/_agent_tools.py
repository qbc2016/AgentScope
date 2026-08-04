# -*- coding: utf-8 -*-
"""Wire channel platform tools into channel-originated agent runs.

A channel-created session records the channel it came from
(``source_channel_id``). When such a session's agent is assembled, this
factory resolves that channel's *local* adapter and attaches whatever
:meth:`ChannelBase.list_tools` exposes — so the agent can act back on the
platform (send a file to another user, list groups, ...).

Distribution: every node runs every enabled channel (no sharding), so the
node running the agent holds the adapter locally; no cross-node hop.

It composes: an optional ``inner`` factory (the deployment's own
``extra_agent_tools``) runs first, and the channel tools are appended.
"""
from typing import TYPE_CHECKING, Awaitable, Callable

from ..storage import StorageBase
from ..workspace_manager import WorkspaceManagerBase
from ...tool import ToolBase

if TYPE_CHECKING:
    from ._dispatcher import ChannelLifecycleDispatcher

# ``() -> dispatcher | None`` — read lazily, as the dispatcher is built
# after the ChatService that holds this factory.
RuntimeGetter = Callable[[], "ChannelLifecycleDispatcher | None"]
InnerFactory = Callable[[str, str, str], Awaitable[list[ToolBase]]]


class ChannelAgentToolFactory:
    """An ``AgentToolFactory`` that adds a channel's tools to its agent."""

    def __init__(
        self,
        storage: StorageBase,
        workspace_manager: WorkspaceManagerBase,
        get_runtime: RuntimeGetter,
        inner: InnerFactory | None = None,
    ) -> None:
        """Bind dependencies.

        Args:
            storage (`StorageBase`):
                Application storage, to read the session's source channel.
            workspace_manager (`WorkspaceManagerBase`):
                Resolves the session's workspace, passed to the channel's
                tools so file-sending reads from the workspace.
            get_runtime (`RuntimeGetter`):
                Returns the channel dispatcher (or ``None`` before it is
                built / when channels are disabled).
            inner (`InnerFactory | None`, optional):
                The deployment's own tool factory, run first and composed
                with the channel tools.
        """
        self._storage = storage
        self._workspace_manager = workspace_manager
        self._get_runtime = get_runtime
        self._inner = inner

    async def __call__(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> list[ToolBase]:
        """Return the inner tools plus this session's channel tools."""
        tools: list[ToolBase] = []
        if self._inner is not None:
            tools += await self._inner(user_id, agent_id, session_id)

        session = await self._storage.get_session(
            user_id,
            agent_id,
            session_id,
        )
        if session is None or session.source_channel_id is None:
            return tools

        runtime = self._get_runtime()
        if runtime is None:
            return tools

        adapter = runtime.get_local_adapter(session.source_channel_id)
        if adapter is None:
            return tools

        workspace = await self._workspace_manager.get_workspace(
            user_id,
            agent_id,
            session_id,
            session.config.workspace_id,
        )
        tools += await adapter.list_tools(workspace)
        return tools
