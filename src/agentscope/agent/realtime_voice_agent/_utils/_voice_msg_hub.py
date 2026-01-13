# -*- coding: utf-8 -*-
"""VoiceMsgHub for managing voice conversation participants, similar to
MsgHub."""

import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Sequence,
    List,
    Optional,
    Type,
)

from agentscope._logging import logger

from ._msg_stream import (
    MsgStream,
    EVENT_SESSION_END,
    get_event_from_msg,
    create_event_msg,
)

# For type hints only, avoiding circular imports
if TYPE_CHECKING:
    from ..agent._websocket_voice_agent import WebSocketVoiceAgent
    from ._voice_user_input import RealtimeVoiceInput


class VoiceMsgHub:
    """VoiceMsgHub manages participants in voice conversations.

    Similar to MsgHub, but designed for voice messages. Supports VoiceAgent
    and RealtimeVoiceInput as participants, providing a centralized message
    stream for all participants to communicate.

    The hub supports both context manager usage and manual start/stop control.

    Examples:
        Usage 1: Context manager (recommended for scripts):

        .. code-block:: python

            agent = VoiceAgent(name="assistant", model=model)
            voice_input = RealtimeVoiceInput()

            async with VoiceMsgHub(participants=[voice_input, agent]) as hub:
                await asyncio.sleep(100)  # Run for 100 seconds

        Usage 2: Manual control (for servers/long-running services):

        .. code-block:: python

            hub = VoiceMsgHub(participants=[voice_input, agent])
            await hub.start()

            # ... your server logic ...

            await hub.stop()

        Usage 3: Wait for completion:

        .. code-block:: python

            hub = VoiceMsgHub(participants=[voice_input, agent])
            await hub.start()
            await hub.join()  # Block until stopped
    """

    def __init__(
        self,
        participants: Sequence,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the VoiceMsgHub.

        Args:
            participants (`Sequence`):
                List of conversation participants.
            name (`Optional[str]`, defaults to `None`):
                Name of the hub for identification. If None, defaults to
                "voice_hub".

        Raises:
            `TypeError`:
                If any participant doesn't match expected types (VoiceAgent or
                RealtimeVoiceInput).
        """
        self.name = name or "voice_hub"
        self._agents: List["WebSocketVoiceAgent"] = []
        self._voice_inputs: List["RealtimeVoiceInput"] = []

        # Import at runtime to avoid circular imports at module load time
        from ..agent._websocket_voice_agent import WebSocketVoiceAgent
        from ._voice_user_input import RealtimeVoiceInput

        # Categorize participants by type using isinstance
        for p in participants:
            if isinstance(p, WebSocketVoiceAgent):
                self._agents.append(p)
            elif isinstance(p, RealtimeVoiceInput):
                self._voice_inputs.append(p)
            else:
                raise TypeError(
                    f"Unsupported participant type: {type(p).__name__}. "
                    f"Expected WebSocketVoiceAgent or RealtimeVoiceInput.",
                )

        self._msg_stream = MsgStream()
        self._initialized = False
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start the hub and initialize all participants.

        Sets up the message stream for all participants and initializes
        all agents. Call this method when not using the context manager.

        Raises:
            `RuntimeError`:
                If the hub is already started.
        """
        if self._initialized:
            raise RuntimeError("VoiceMsgHub is already started")

        self._stop_event.clear()

        # Set msg_stream for all participants
        for voice_input in self._voice_inputs:
            voice_input.set_msg_stream(self._msg_stream)
        for agent in self._agents:
            agent.set_msg_stream(self._msg_stream)

        # Start voice inputs
        for voice_input in self._voice_inputs:
            await voice_input.start()

        # Initialize all agents
        for agent in self._agents:
            await agent.initialize()

        self._initialized = True

        # Start watching for SESSION_END event
        self._tasks.append(
            asyncio.create_task(self._watch_session_end()),
        )

        logger.info(
            "VoiceMsgHub '%s' started with %d agents, %d voice inputs",
            self.name,
            len(self._agents),
            len(self._voice_inputs),
        )

    async def stop(self) -> None:
        """Stop the hub and clean up all resources.

        Stops all voice inputs, closes all agents, and closes the message
        stream. Call this method when not using the context manager.
        """
        if not self._initialized:
            return

        self._stop_event.set()

        # Cancel any running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()

        # Stop all voice inputs
        for voice_input in self._voice_inputs:
            await voice_input.stop()

        # Stop and close all agents
        for agent in self._agents:
            agent.stop()
        for agent in self._agents:
            await agent.close()

        # Close MsgStream
        await self._msg_stream.close()

        self._initialized = False
        logger.info("VoiceMsgHub '%s' stopped", self.name)

    async def request_stop(self, sender: str = "system") -> None:
        """Request the hub to stop by sending a SESSION_END event.

        This is a non-blocking way to request hub termination.
        Any participant can call this to signal the end of a session.

        Args:
            sender (`str`, defaults to `"system"`):
                The name of the sender requesting the stop.

        Example:
            .. code-block:: python

                # From anywhere with access to msg_stream
                await hub.request_stop("user")

                # Or push directly to msg_stream
                await msg_stream.push(
                    create_event_msg("user", EVENT_SESSION_END)
                )
        """
        if not self._initialized:
            return

        msg = create_event_msg(sender, EVENT_SESSION_END, role="system")
        await self._msg_stream.push(msg)
        logger.info(
            "VoiceMsgHub '%s' stop requested by '%s'",
            self.name,
            sender,
        )

    async def join(self, timeout: Optional[float] = None) -> None:
        """Wait until the hub is stopped.

        This method blocks until stop() is called or the timeout expires.

        Args:
            timeout (`Optional[float]`, defaults to `None`):
                Maximum time to wait in seconds. If None, waits indefinitely.

        Returns:
            None

        Raises:
            `asyncio.TimeoutError`:
                If the timeout expires before the hub is stopped.
        """
        if not self._initialized:
            return

        await asyncio.wait_for(
            self._stop_event.wait(),
            timeout=timeout,
        )

    async def _watch_session_end(self) -> None:
        """Watch for SESSION_END event and stop the hub.

        This task runs in the background and monitors for SESSION_END messages.
        When received, it triggers hub.stop().
        """
        try:
            async for msg in self._msg_stream.subscribe(
                f"_hub_watcher_{self.name}",
            ):
                if self._stop_event.is_set():
                    break

                event = get_event_from_msg(msg)
                if event == EVENT_SESSION_END:
                    logger.info(
                        "VoiceMsgHub '%s' received SESSION_END from '%s'",
                        self.name,
                        msg.name,
                    )
                    # Set stop event (don't call stop() to avoid recursion)
                    self._stop_event.set()
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("VoiceMsgHub '%s' watcher error: %s", self.name, e)

    async def run_forever(self) -> None:
        """Start the hub and run until interrupted.

        Convenience method that starts the hub and waits indefinitely.
        Useful for simple scripts.

        Example:
            .. code-block:: python

                hub = VoiceMsgHub(participants=[agent, voice_input])
                try:
                    await hub.run_forever()
                except KeyboardInterrupt:
                    await hub.stop()
        """
        await self.start()
        await self.join()

    async def __aenter__(self) -> "VoiceMsgHub":
        """Enter context and initialize all participants.

        Returns:
            `VoiceMsgHub`:
                Self reference for context manager usage.
        """
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit context and clean up resources.

        Args:
            exc_type (`Optional[Type[BaseException]]`):
                Exception type if an exception occurred.
            exc_val (`Optional[BaseException]`):
                Exception instance if an exception occurred.
            exc_tb (`Optional[Any]`):
                Traceback object if an exception occurred.
        """
        await self.stop()

    @property
    def agents(self) -> List["WebSocketVoiceAgent"]:
        """Get all agents in the hub.

        Returns:
            `List[WebSocketVoiceAgent]`:
                List of all VoiceAgent instances managed by this hub.
        """
        return self._agents

    @property
    def voice_inputs(self) -> List["RealtimeVoiceInput"]:
        """Get all voice inputs in the hub.

        Returns:
            `List[RealtimeVoiceInput]`:
                List of all RealtimeVoiceInput instances managed by this hub.
        """
        return self._voice_inputs

    @property
    def msg_stream(self) -> MsgStream:
        """Get the internal message stream.

        Returns:
            `MsgStream`:
                The message stream instance used for inter-participant
                communication.
        """
        return self._msg_stream

    @property
    def is_initialized(self) -> bool:
        """Check if the hub is initialized and running.

        Returns:
            `bool`:
                True if the hub has been started, False otherwise.
        """
        return self._initialized

    @property
    def is_running(self) -> bool:
        """Check if the hub is currently running.

        Returns:
            `bool`:
                True if the hub is initialized and not stopped.
        """
        return self._initialized and not self._stop_event.is_set()

    @property
    def participant_count(self) -> int:
        """Get the total number of participants.

        Returns:
            `int`:
                Total count of agents and voice inputs in this hub.
        """
        return len(self._agents) + len(self._voice_inputs)
